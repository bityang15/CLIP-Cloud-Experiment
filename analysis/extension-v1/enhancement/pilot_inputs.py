"""Read and verify frozen inputs for the independent enhancement pilot."""

from hashlib import sha256
import importlib
from importlib import metadata
from itertools import product
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[2]
PROTOCOL_PATH = HERE / "enhancement_protocol_v1.json"
INPUTS_PATH = HERE / "pilot_inputs_v1.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


PROTOCOL = read_json(PROTOCOL_PATH)
INPUTS = read_json(INPUTS_PATH)
PILOT = PROTOCOL["pilot"]
CONDITIONS = list(product(PROTOCOL["algorithms"], PROTOCOL["levels"]))
COMBINATIONS = list(product(PILOT["datasets"], PILOT["methods"], PILOT["groups"]))
SEED = PILOT["seed"]


def digest(path):
    # Python 3.10, as in the frozen runtime.
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_digest(path, expected):
    require(digest(path) == expected, f"SHA-256 differs: {path}")


def condition_key(dataset, method, group, corruption, level):
    return f"{dataset}/{method}/seed{SEED}/{group}/{corruption}_L{level}"


def source_identity(expected_commit):
    require(re.fullmatch(r"[a-f0-9]{40}", expected_commit), "Expected a full 40-character Git commit SHA")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(CHECKOUT), *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    require(commit == expected_commit, f"Checked-out commit is {commit}, expected {expected_commit}")
    executable_files = ["pilot.py", "pilot_inputs.py", "enhancement_v1.py", PROTOCOL_PATH.name, INPUTS_PATH.name]
    paths = [(HERE / name).relative_to(CHECKOUT).as_posix() for name in executable_files]
    git("ls-files", "--error-unmatch", "--", *paths)
    require(not git("diff", "HEAD", "--name-only", "--", *paths), "Pilot source or protocol differs from committed bytes")
    for path in paths:
        committed = subprocess.check_output(["git", "-C", str(CHECKOUT), "show", f"HEAD:{path}"])
        require(sha256(committed).hexdigest() == digest(CHECKOUT / path), f"Working-tree bytes differ: {path}")
    return {"commit": commit, "protocol_sha256": digest(PROTOCOL_PATH), "inputs_sha256": digest(INPUTS_PATH),
            "source_sha256": {path: digest(CHECKOUT / path) for path in paths}}


def frozen_source(history, reference):
    path = reference / "run_manifest.json"
    require_digest(path, INPUTS["formal_manifest_sha256"])
    archived = read_json(path)
    require(archived["protocol_version"] == PROTOCOL["corruption_protocol"], "Unexpected history protocol")
    for name, expected in {**archived["sources"], **INPUTS["historical_records"]}.items():
        require_digest(history / name, expected)
    return archived


def import_history(history):
    if str(history) not in sys.path:
        sys.path.insert(0, str(history))
    package = importlib.import_module("course_eval")
    require(Path(package.__file__).resolve() == history / "course_eval" / "__init__.py",
            f"Wrong course_eval imported: {package.__file__}")
    common = importlib.import_module("course_eval.common")
    require(common.REPO == history, "Historical paths must resolve to the checked runtime root")
    return common, importlib.import_module("course_eval.inference")


def select_samples(history, dataset, group):
    path = history / "records" / "manifests" / dataset / f"seed{SEED}" / f"test_{group}.json"
    pinned = INPUTS["samples"][dataset][group]
    require_digest(path, pinned["manifest_sha256"])
    rows = read_json(path)
    positions = PILOT["sample_positions"]
    count = PILOT["images_per_group"]
    require(len(positions) == count and positions == sorted(set(positions))
            and positions[0] == 0 and positions[-1] < PILOT["historical_batch_size"] <= len(rows),
            f"Invalid pilot positions for historical first batch: {path}")
    selected = [rows[index] for index in positions]
    ids = [row["image_id"] for row in selected]
    require(ids == pinned["image_ids"] and len(set(ids)) == count, f"Pilot sample lock differs: {path}")
    return selected


def check_environment(device, history, reference):
    errors = []
    report = {"history": str(history), "reference": str(reference), "requested_device": device, "errors": errors}

    def check(name, operation):
        try:
            report[name] = operation()
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")

    def dependency_versions():
        contract = read_json(history / "course_eval" / "protocol_v1.json")["dependencies"].copy()
        contract.pop("python")
        contract.update(PROTOCOL["dependencies"])
        # These distributions are imported by the archived trainer/utility code.
        additional = ("PyYAML", "regex", "tabulate", "tqdm", "wilds", "tensorboard", "gdown", "future")
        report["python"] = platform.python_version()
        if not report["python"].startswith("3.10."):
            errors.append("Python 3.10.x is required")
        installed = {}
        for name in (*contract, *additional):
            try:
                installed[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                errors.append(f"Missing distribution: {name}")
                continue
            if name in contract and installed[name].split("+")[0] != contract[name]:
                errors.append(f"{name}: expected {contract[name]}, found {installed[name]}")
        return installed

    def gpu():
        info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
                                        "--format=csv,noheader,nounits"], text=True, timeout=15)
        import torch
        require(re.fullmatch(r"cuda:\d+", device), f"Expected a CUDA device, got {device}")
        index = int(device.split(":")[1])
        require(torch.cuda.is_available() and index < torch.cuda.device_count(), f"CUDA unavailable for {device}")
        requested = torch.cuda.get_device_properties(index)
        require("RTX 4090" in requested.name, f"Pilot expects an RTX 4090, found {requested.name}")
        runtime = {"cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        for name, value in runtime.items():
            require(value == report["formal"]["software"][name], f"{name} differs from the frozen runtime: {value}")
        return {"name": requested.name, "total_memory_bytes": requested.total_memory, **runtime,
                "nvidia_smi": info.strip().splitlines()}

    def datasets():
        manifest = read_json(history / "records" / "data_manifest.json")["datasets"]
        result = {}
        data_root = (history / "datasets").resolve()
        for dataset in PILOT["datasets"]:
            ids = set()
            for seed, group in product((1, 2, 3), PILOT["groups"]):
                path = history / "records" / "manifests" / dataset / f"seed{seed}" / f"test_{group}.json"
                ids.update(row["image_id"] for row in read_json(path))
            missing = [image_id for image_id in sorted(ids) if not (data_root / image_id).is_file()]
            if missing:
                errors.append(f"{dataset}: missing {len(missing)}/{len(ids)} test images, first five: {missing[:5]}")
            sampled = list(row["image_id"] for group in PILOT["groups"] for row in select_samples(history, dataset, group))
            for image_id in sampled:
                path = (data_root / image_id).resolve()
                if path.is_file():
                    require(path.is_relative_to(data_root), f"Image path escapes dataset root: {image_id}")
                    require_digest(path, manifest[dataset]["images"][image_id]["sha256"])
            result[dataset] = {"test_images_expected": len(ids), "present": len(ids) - len(missing),
                               "pilot_hashes_checked": len(set(sampled) - set(missing))}
        return result

    def checkpoints():
        rows = read_json(history / "records" / "checkpoints.json")
        specs = [(dataset, method, seed) for dataset, method, seed in product(
            PILOT["datasets"], [m for m in PILOT["methods"] if m != "clip"], (1, 2, 3))]
        results = {}
        for dataset, method, seed in specs:
            prefix = f"outputs/B/{dataset}/{method}/seed{seed}/"
            candidates = [row for row in rows if row["path"].startswith(prefix)]
            require(len(candidates) == 1, f"No unique checkpoint in the archive ledger: {prefix}")
            row = candidates[0]
            path = history / row["path"]
            if not path.is_file() or path.stat().st_size != row["bytes"]:
                errors.append(f"Missing or wrong-size checkpoint: {path}")
                continue
            if seed == SEED:
                require_digest(path, row["sha256"])
            results[row["path"]] = {"bytes": row["bytes"], "sha256_verified": seed == SEED}
        return results

    def shared_weight():
        weight = history / "weights" / "ViT-B-16.pt"
        expected = read_json(history / "records" / "config_manifest.json")["pretrained_sha256"]
        require_digest(weight, expected)
        return {"path": str(weight), "sha256": expected, "bytes": weight.stat().st_size}

    def controls():
        for key, expected in INPUTS["references"].items():
            folder = reference / "predictions" / key
            for field, name in (("metadata_sha256", "metadata.json"), ("predictions_sha256", "predictions.npz")):
                require_digest(folder / name, expected[field])
        return len(INPUTS["references"])

    check("formal", lambda: frozen_source(history, reference))
    check("dependencies", dependency_versions)
    check("gpu", gpu)
    check("dataset_inventory", datasets)
    check("prompt_checkpoints", checkpoints)
    check("shared_weight", shared_weight)
    check("formal_controls", controls)
    if not errors:
        def runtime_imports():
            common, inference = import_history(history)
            import cv2
            from PIL import features
            import scipy.ndimage
            import torchvision

            require(features.version("jpg") == report["formal"]["software"]["libjpeg"], "libjpeg differs from formal run")
            require(features.version("libjpeg_turbo") == report["formal"]["software"]["libjpeg_turbo"],
                    "libjpeg-turbo differs from formal run")
            require(cv2.__version__ == PROTOCOL["dependencies"]["opencv-python-headless"].rsplit(".", 1)[0],
                    f"Imported OpenCV differs: {cv2.__version__}")
            inference.historical_modules()
            for dataset, method, group in COMBINATIONS:
                cfg, _, _ = inference.configuration(dataset, method, SEED, group, CHECKOUT)
                require(cfg.DATALOADER.TEST.BATCH_SIZE == PILOT["historical_batch_size"],
                        f"Historical batch size differs: {dataset}/{method}/{group}")
            return {"cv2": cv2.__version__, "scipy": scipy.__version__, "torchvision": torchvision.__version__,
                    "historical_protocol": common.REPO.as_posix()}

        check("runtime_imports", runtime_imports)
    report["passed"] = not errors
    return report
