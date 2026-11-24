"""Run, resume, or verify the frozen full Enhanced-only evaluation."""

import argparse
from contextlib import contextmanager
import csv
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
import traceback
from uuid import uuid4

from formal_inputs import (BATCH_SIZE, BLOCKS, CHECKOUT, CONDITIONS, LOCK, LOCK_PATH, MATRIX, QUALITY_ROWS,
                           formal_identity, full_preflight, import_history, key_for, load_reference, split_key)
from pilot import EnhancedTransform, check_export, remember_inputs, time_utc, write_json
from pilot_inputs import PROTOCOL, PROTOCOL_PATH, digest, read_json, require, require_digest

ARTIFACTS = ("predictions.npz", "metadata.json", "effective_config.yaml", "historical_export_metadata.json")
METRIC_FIELDS = ("n", "classes", "evaluation_seconds", "export_wall_seconds", "peak_memory_gib",
                 "peak_reserved_memory_gib", "initialization_seconds", "initialization_peak_memory_gib",
                 "predictions_sha256", "config_sha256", "checkpoint_sha256")


@contextmanager
def run_lock(run):
    # The operating system releases the lock on exit, including process termination.
    with (run / "run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError(f"Another writer holds the run lock: {run}") from error
        try:
            yield
        finally:
            unlock()


def atomic_json(path, value):
    tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    write_json(tmp, value)
    os.replace(tmp, path)


def combo_paths(run, key):
    staging = run / ".work" / key
    published = run / "predictions" / key / "enhanced"
    require(staging.resolve() == staging and published.resolve() == published, f"Symlink in output path: {key}")
    return staging, published


def setup_run(namespace, run_id, resume):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id), "Invalid formal run ID")
    require(namespace.resolve() == namespace, "Formal output namespace must be inside the checkout")
    run = namespace / run_id
    require(run.resolve() == run, "Formal run directory cannot be a symlink")
    if resume:
        require(run.is_dir(), f"No existing formal run: {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)
    return run


def bind_run(run, identity, history, reference, device, resume):
    bound = dict(identity, scope="full_enhanced_only", run_id=run.name, history=str(history), reference=str(reference),
                 device=device, expected_exports=len(MATRIX), batch_size=BATCH_SIZE)
    path = run / "run_manifest.json"
    if resume:
        previous = read_json(path)
        require({k: v for k, v in previous.items() if k != "started_at_utc"} == bound,
                "Resume requires the same commit, roots, device, and frozen inputs")
        return previous
    manifest = dict(bound, started_at_utc=time_utc())
    atomic_json(path, manifest)
    return manifest


def expected_config(inference, key, staging):
    cfg, _, _ = inference.configuration(*split_key(key)[:4], staging)
    require(cfg.DATALOADER.TEST.BATCH_SIZE == BATCH_SIZE and cfg.DATALOADER.NUM_WORKERS == 0,
            f"Formal loader differs from the archived batch protocol: {key}")
    return cfg


def export_full(inference, harness, staging, key, cfg, rows, identity):
    """Use the archived full exporter with a per-dataset transformation wrapper."""
    import torch
    from torch.utils.data import DataLoader
    from enhancement_v1 import enhancement_parameters, enhancer_for

    started = time.perf_counter()
    dataset, _, seed, group, corruption, level = split_key(key)
    staging.mkdir(parents=True, exist_ok=False)
    images = inference.ManifestImages(dataset, rows, corruption, level, cfg)
    images.transform = EnhancedTransform(images.transform, enhancer_for(corruption))
    require(str(images.root) == cfg.DATASET.ROOT, f"Image root differs from effective config: {key}")
    config = cfg.dump().encode("utf-8")
    harness.cfg = cfg
    harness.config_sha256 = sha256(config).hexdigest()
    harness.test_loader = DataLoader(images, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    harness.val_loader = None
    harness.experiment.ExperimentMixin.test(harness, split="test")
    raw_metadata = staging / "metadata.json"
    meta = read_json(raw_metadata)
    for record in images.input_records:
        record["corrupted_rgb_uint8_sha256"] = images.transform.corrupted_hashes[record["image_id"]]
        record["enhanced_rgb_uint8_sha256"] = record["rgb_uint8_sha256"]
    manifest, _ = inference.sample_manifest(dataset, seed, group)
    meta.update(sample_scope="full", variant="enhanced", runtime_commit=identity["commit"],
                protocol_version=PROTOCOL["corruption_protocol"], enhancement_protocol=PROTOCOL["version"],
                enhancement_protocol_sha256=identity["protocol_sha256"], formal_inputs_sha256=identity["formal_inputs_sha256"],
                corruption=corruption, level=level, parameters=inference.parameters(corruption, level),
                enhancement=enhancement_parameters(corruption), historical_config_sha256=digest(harness.historical_config),
                sample_manifest_sha256=digest(manifest), sample_records=images.input_records, batch_size=BATCH_SIZE,
                initialization_seconds=harness.initialization_seconds,
                initialization_peak_memory_gib=harness.initialization_peak_memory_gib,
                export_wall_seconds=time.perf_counter() - started,
                peak_reserved_memory_gib=torch.cuda.max_memory_reserved() / 1024**3)
    raw_metadata.rename(staging / "historical_export_metadata.json")
    write_json(raw_metadata, meta)
    (staging / "effective_config.yaml").write_bytes(config)


def inventory_of(root, paths):
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}


def receipt(directory, key, identity):
    files = [directory / name for name in ARTIFACTS]
    require(all(p.is_file() and not p.is_symlink() for p in files), f"Missing or unsafe output files: {key}")
    return {"key": key, "commit": identity["commit"], "formal_inputs_sha256": identity["formal_inputs_sha256"],
            "reference": LOCK["references"][key], "files": inventory_of(directory, files)}


def validate_combination(common, directory, key, identity, expected=None, reference=None):
    from enhancement_v1 import enhancement_parameters

    require(directory.is_dir() and not directory.is_symlink(), f"Missing or unsafe prediction directory: {key}")
    require({path.name for path in directory.iterdir()} == set(ARTIFACTS) | {"receipt.json"},
            f"Unexpected combination inventory: {key}")
    require(read_json(directory / "receipt.json") == receipt(directory, key, identity),
            f"Combination receipt or file hash differs: {key}")
    config = directory / "effective_config.yaml"
    if expected is not None:
        require(config.read_bytes() == expected.dump().encode("utf-8"), f"Archived effective config differs: {key}")
    if reference is None:
        arrays, meta = common.load_predictions(directory)
    else:
        arrays, meta = check_export(common, directory, *reference, "enhanced", key)
    require(meta["n"] == LOCK["references"][key][2] and meta["sample_scope"] == "full" and
            meta["variant"] == "enhanced" and meta["batch_size"] == BATCH_SIZE and
            meta["runtime_commit"] == identity["commit"] and
            meta["enhancement_protocol_sha256"] == identity["protocol_sha256"] and
            meta["formal_inputs_sha256"] == identity["formal_inputs_sha256"] and
            meta["enhancement"] == enhancement_parameters(split_key(key)[4]), f"Wrong formal provenance: {key}")
    require_digest(config, meta["config_sha256"])
    return arrays, meta


def metrics_for(key, metadata):
    fields = ("dataset", "method", "seed", "group", "corruption", "level")
    return dict(zip(fields, split_key(key)), **{field: metadata[field] for field in METRIC_FIELDS},
                output=f"predictions/{key}/enhanced/predictions.npz")


def quarantine(run, staging):
    if staging.exists():
        destination = run / "failed_staging" / uuid4().hex / staging.relative_to(run / ".work")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        return destination.relative_to(run).as_posix()
    return None


def run_outputs(run):
    root = run / "predictions"
    files = sorted(path for path in root.rglob("*") if path.is_file())
    known = {f"{key}/enhanced" for key in MATRIX}
    require(all(path.parent.relative_to(root).as_posix() in known for path in files), "Unexpected prediction path")
    return files


def artifact_paths(run, excluded=("SHA256SUMS.json",)):
    paths = sorted(path for path in run.rglob("*") if path.is_file() and path.relative_to(run).as_posix() not in excluded)
    require(all(not path.is_symlink() and path.resolve().is_relative_to(run) for path in paths), "Unsafe run artifact")
    return paths


def seal_run(run, manifest, completed, shared):
    require(set(completed) == set(MATRIX) and len(shared) == QUALITY_ROWS, "Incomplete formal output")
    path = run / "metrics.csv"
    tmp = run / f"metrics.csv.{uuid4().hex}.tmp"
    with tmp.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(completed[MATRIX[0]]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(completed[key] for key in MATRIX)
    os.replace(tmp, path)
    atomic_json(run / "completion.json", {"passed": True, "exports": len(completed), "commit": manifest["commit"],
                "prediction_rows": sum(row["n"] for row in completed.values()), "unique_enhanced_inputs": len(shared),
                "finished_at_utc": time_utc()})
    atomic_json(run / "SHA256SUMS.json", {"files": inventory_of(run, artifact_paths(run))})


def verify_formal(run, commit, manifest_sha256=None):
    run = run.resolve()
    path = run / "SHA256SUMS.json"
    if manifest_sha256 is not None:
        require_digest(path, manifest_sha256)
    expected = read_json(path)["files"]
    observed = inventory_of(run, artifact_paths(run))
    require(expected == observed, "Formal run inventory or SHA-256 differs")
    identity = read_json(run / "run_manifest.json")
    status = read_json(run / "completion.json")
    require(identity["scope"] == "full_enhanced_only" and commit == identity["commit"] == status["commit"] and
            identity["formal_inputs_sha256"] == digest(LOCK_PATH) and identity["protocol_sha256"] == digest(PROTOCOL_PATH) and
            identity["expected_exports"] == len(MATRIX) and status["passed"] and status["exports"] == len(MATRIX) and
            status["unique_enhanced_inputs"] == QUALITY_ROWS and
            status["prediction_rows"] == sum(LOCK["references"][key][2] for key in MATRIX),
            "Wrong commit, input lock or incomplete formal run")
    expected_exports = {f"predictions/{key}/enhanced/{file}" for key in MATRIX for file in (*ARTIFACTS, "receipt.json")}
    require({p.relative_to(run).as_posix() for p in run_outputs(run)} == expected_exports, "Wrong formal output matrix")
    return dict(status, manifest_sha256=digest(path), verified_files=len(observed))


def runtime_fingerprint(readiness, device):
    runtime = readiness["runtime"]
    gpu = runtime["gpu"]
    # Keep hardware and driver identity while excluding changing free memory.
    nvidia = [row.split(",")[:3] for row in gpu["nvidia_smi"]]
    return {"device": device, "gpu_name": gpu["name"], "drivers": nvidia, "cuda_runtime": gpu["cuda_runtime"],
            "cudnn": gpu["cudnn"], "python": runtime["python"], "dependencies": runtime["dependencies"],
            "runtime_imports": runtime["runtime_imports"]}


def process_run(run, manifest, history, reference, device):
    import torch

    common, inference = import_history(history)
    completed, shared = {}, {}
    current = None
    try:
        run_outputs(run)
        # Revalidate all completed combinations before loading any model on resume.
        for key in MATRIX:
            current = key
            staging, published = combo_paths(run, key)
            if published.exists():
                _, meta = validate_combination(common, published, key, manifest, expected_config(inference, key, staging),
                                               load_reference(common, reference, key))
                dataset, _, _, _, corruption, level = split_key(key)
                remember_inputs(shared, dataset, corruption, level, meta["sample_records"])
                completed[key] = metrics_for(key, meta)
            if staging.exists():
                retained = quarantine(run, staging)
                write_json(run / f"interrupted-{uuid4().hex}.json", {"key": key, "recovered_staging": retained,
                                                                  "resumed_at_utc": time_utc()})
        print(f"Verified {len(completed)}/{len(MATRIX)} existing combinations", flush=True)
        for block in BLOCKS:
            todo = [key_for(*block, *condition) for condition in CONDITIONS if key_for(*block, *condition) not in completed]
            if not todo:
                continue
            current = todo[0]
            dataset, _, seed, group = block
            _, rows = common.sample_manifest(dataset, seed, group)
            with inference.restored_model(*block, combo_paths(run, current)[0], device) as harness:
                for key in todo:
                    current = key
                    stage, published = combo_paths(run, key)
                    cfg = expected_config(inference, key, stage)
                    reference_data = load_reference(common, reference, key)
                    export_full(inference, harness, stage, key, cfg, rows, manifest)
                    write_json(stage / "receipt.json", receipt(stage, key, manifest))
                    _, meta = validate_combination(common, stage, key, manifest, cfg, reference_data)
                    _, _, _, _, corruption, level = split_key(key)
                    remember_inputs(shared, dataset, corruption, level, meta["sample_records"])
                    published.parent.mkdir(parents=True, exist_ok=True)
                    require(not published.exists(), f"Refusing to overwrite an existing combination: {key}")
                    stage.rename(published)
                    completed[key] = metrics_for(key, meta)
                    print(f"{len(completed)}/{len(MATRIX)} verified {key}", flush=True)
            del harness
            gc.collect()
            torch.cuda.empty_cache()
        current = None
        seal_run(run, manifest, completed, shared)
    except BaseException:
        error = traceback.format_exc()
        write_json(run / f"failure-{uuid4().hex}.json", {"key": current, "completed_exports": len(completed),
                   "failed_at_utc": time_utc(), "traceback": error})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--commit", required=True)
        sub.add_argument("--history", type=Path, required=True)
        sub.add_argument("--reference", type=Path, required=True)
        sub.add_argument("--device", default="cuda:0")
    run = commands.choices["run"]
    run.add_argument("--run-id", required=True)
    run.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--commit", required=True)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify_formal(args.run_dir, args.commit, args.manifest_sha256), indent=2))
        return 0
    identity = formal_identity(args.commit)
    history, reference = args.history.resolve(), args.reference.resolve()
    readiness = full_preflight(identity, history, reference, args.device)
    if args.command == "preflight":
        print(json.dumps(readiness, indent=2))
        return int(not readiness["passed"])
    require(readiness["passed"], f"Formal preflight failed: {readiness['errors']}")
    identity["runtime_fingerprint"] = runtime_fingerprint(readiness, args.device)
    namespace = CHECKOUT / "outputs" / "course_enhancement_v1_formal"
    destination = setup_run(namespace, args.run_id, args.resume)
    with run_lock(destination):
        manifest = bind_run(destination, identity, history, reference, args.device, args.resume)
        if (destination / "SHA256SUMS.json").exists():
            print(json.dumps(verify_formal(destination, args.commit), indent=2))
            return 0
        write_json(destination / f"preflight-{uuid4().hex}.json", readiness)
        process_run(destination, manifest, history, reference, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
