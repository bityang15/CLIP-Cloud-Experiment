"""Independent enhancement pilot, remote readiness checks, and local transfer verification."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
import traceback
from unittest.mock import patch

from pilot_inputs import (CHECKOUT, COMBINATIONS, CONDITIONS, INPUTS, INPUTS_PATH, PILOT, PROTOCOL,
                          PROTOCOL_PATH, SEED, check_environment, condition_key, digest,
                          import_history, read_json, require, require_digest, select_samples, source_identity)


def write_json(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def time_utc():
    return datetime.now(timezone.utc).isoformat()


class EnhancedTransform:
    """Wrap only the extended dataset instance. The frozen control exporter stays untouched."""

    def __init__(self, historical, operation):
        self.historical = historical
        self.corruption = historical.corruption
        self.tensor = historical.tensor
        self.operation = operation
        self.corrupted_hashes = {}

    def rgb(self, image, image_id):
        import numpy as np

        corrupted = self.historical.rgb(image, image_id)
        before = sha256(corrupted.tobytes()).hexdigest()
        enhanced = self.operation(corrupted)
        require(enhanced.shape == tuple(PROTOCOL["image_shape"]) and enhanced.dtype == np.uint8,
                f"Enhanced RGB contract violated: {image_id}")
        require(sha256(corrupted.tobytes()).hexdigest() == before, f"Enhancement modified its input: {image_id}")
        self.corrupted_hashes[image_id] = before
        return enhanced


def padded_forward(images, forward):
    """Preserve the archived batch geometry using only the eight selected images."""
    import torch

    positions = PILOT["sample_positions"]
    require(len(images) == len(positions), "Pilot did not supply exactly one eight-image batch")
    order = [0] * PILOT["historical_batch_size"]
    for index, position in enumerate(positions):
        order[position] = index
    full = images.index_select(0, torch.tensor(order, device=images.device))
    logits = forward(full)
    require(len(logits) == len(order), "Historical forward returned an unexpected batch shape")
    return logits.index_select(0, torch.tensor(positions, device=logits.device))


@contextmanager
def historical_batch(harness):
    import torch

    original = harness.model_inference
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)

    def measured(images):
        started.record()
        logits = padded_forward(images, original)
        finished.record()
        return logits

    require(harness.cfg.DATALOADER.TEST.BATCH_SIZE == PILOT["historical_batch_size"], "Historical batch size changed")
    with patch.object(harness, "model_inference", measured):
        yield lambda: started.elapsed_time(finished) / 1000


def export_enhanced(inference, harness, output, corruption, level, rows):
    import torch
    from torch.utils.data import DataLoader
    from enhancement_v1 import enhancement_parameters, enhancer_for

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    dataset = inference.ManifestImages(harness.dataset_key, rows, corruption, level, harness.cfg)
    dataset.transform = EnhancedTransform(dataset.transform, enhancer_for(corruption))
    cfg = harness.cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = str(output)
    cfg.EXPERIMENT.EXPORT_DIR = str(output)
    cfg.DATASET.ROOT = str(dataset.root)
    cfg.freeze()
    config = cfg.dump()
    harness.cfg = cfg
    harness.config_sha256 = sha256(config.encode("utf-8")).hexdigest()
    harness.test_loader = DataLoader(dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False,
                                    num_workers=0, drop_last=False)
    harness.val_loader = None
    with torch.no_grad():
        metadata = inference.export_subset(harness, output)
    for record in dataset.input_records:
        record["corrupted_rgb_uint8_sha256"] = dataset.transform.corrupted_hashes[record["image_id"]]
        record["enhanced_rgb_uint8_sha256"] = record["rgb_uint8_sha256"]
    manifest, _ = inference.sample_manifest(harness.dataset_key, SEED, cfg.DATASET.SUBSAMPLE_CLASSES)
    metadata.update(variant="enhanced", corruption=corruption, level=level, sample_scope="subset",
                    protocol_version=PROTOCOL["corruption_protocol"], enhancement_protocol=PROTOCOL["version"],
                    enhancement_protocol_sha256=digest(PROTOCOL_PATH), parameters=inference.parameters(corruption, level),
                    enhancement=enhancement_parameters(corruption), historical_config_sha256=digest(harness.historical_config),
                    sample_manifest_sha256=digest(manifest), sample_records=dataset.input_records,
                    initialization_seconds=harness.initialization_seconds,
                    initialization_peak_memory_gib=harness.initialization_peak_memory_gib,
                    export_wall_seconds=time.perf_counter() - started)
    (output / "metadata.json").rename(output / "historical_export_metadata.json")
    write_json(output / "metadata.json", metadata)
    with (output / "effective_config.yaml").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(config)
    return metadata


def subset_reference(common, reference, key, rows, classnames):
    import numpy as np

    directory = reference / "predictions" / key
    pinned = INPUTS["references"][key]
    require_digest(directory / "metadata.json", pinned["metadata_sha256"])
    arrays, metadata = common.load_predictions(directory)
    require(metadata["predictions_sha256"] == pinned["predictions_sha256"], f"Formal prediction hash differs: {key}")
    by_id = {image_id: index for index, image_id in enumerate(arrays["image_ids"].tolist())}
    ids = [row["image_id"] for row in rows]
    selected = [by_id[image_id] for image_id in ids]
    result = {name: arrays[name][selected] for name in ("logits", "labels", "image_ids")}
    result["classnames"] = arrays["classnames"]
    require(np.isfinite(result["logits"]).all(), f"Formal logits are nonfinite: {key}")
    require(result["labels"].tolist() == [row["label"] for row in rows], f"Formal labels differ: {key}")
    require(result["classnames"].tolist() == classnames, f"Formal candidate order differs: {key}")
    records = {row["image_id"]: row for row in metadata["sample_records"]}
    require(len(records) == len(arrays["image_ids"]), f"Formal input records differ in size: {key}")
    return result, metadata, {image_id: records[image_id] for image_id in ids}


def check_export(common, directory, expected, formal, records, variant, key):
    import numpy as np

    arrays, current = common.load_predictions(directory)
    dataset, method, seed, group, condition = key.split("/")
    corruption, level = condition.rsplit("_L", 1)
    observed = tuple(current[name] for name in ("dataset", "method", "seed", "candidate_group", "corruption", "level"))
    require(observed == (dataset, method, int(seed[4:]), group, corruption, int(level)), f"Export identity differs: {directory}")
    for name in ("image_ids", "labels", "classnames"):
        require(np.array_equal(arrays[name], expected[name]), f"{variant} {name} differ: {directory}")
    require(arrays["logits"].shape == expected["logits"].shape and np.isfinite(arrays["logits"]).all(),
            f"Invalid logits: {directory}")
    for name in ("checkpoint_sha256", "historical_config_sha256", "sample_manifest_sha256", "data_manifest_sha256", "parameters"):
        require(current[name] == formal[name], f"{variant} {name} differs from the archived control: {directory}")
    require_digest(directory / "effective_config.yaml", current["config_sha256"])
    actual_records = current["sample_records"]
    require([row["image_id"] for row in actual_records] == expected["image_ids"].tolist(),
            f"{variant} input record ordering differs: {directory}")
    for row in actual_records:
        known = records[row["image_id"]]
        for name in ("source_sha256", "noise_seed"):
            require(row[name] == known[name], f"{variant} {name} differs: {row['image_id']}")
        name = "corrupted_rgb_uint8_sha256" if variant == "enhanced" else "rgb_uint8_sha256"
        require(row[name] == known["rgb_uint8_sha256"], f"Corrupted pixels differ for {row['image_id']}: {directory}")
        if variant == "corrupted":
            require(row["tensor_sha256"] == known["tensor_sha256"], f"Control tensor differs: {row['image_id']}")
        else:
            require(row["enhanced_rgb_uint8_sha256"] == row["rgb_uint8_sha256"], "Enhanced hash bookkeeping differs")
    return arrays, current


def remember_inputs(cache, dataset, corruption, level, records):
    for row in records:
        key = dataset, corruption, level, row["image_id"]
        signature = row["rgb_uint8_sha256"], row["tensor_sha256"]
        require(cache.setdefault(key, signature) == signature, f"Enhanced pixels or tensors differ across methods: {key}")


def files_under(run):
    return sorted(path for path in run.rglob("*") if path.is_file() and path != run / "SHA256SUMS.json")


def seal(run):
    files = {path.relative_to(run).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
             for path in files_under(run)}
    write_json(run / "SHA256SUMS.json", {"files": files})


def verify(run, expected_commit, allow_failed=False):
    inventory = read_json(run / "SHA256SUMS.json")["files"]
    observed = {path.relative_to(run).as_posix(): path for path in files_under(run)}
    require(inventory.keys() == observed.keys(), "Transferred run has missing or extra artifacts")
    for name, record in inventory.items():
        path = observed[name]
        require(not path.is_symlink() and path.resolve().is_relative_to(run.resolve()), f"Unsafe transferred path: {path}")
        require(path.stat().st_size == record["bytes"], f"Transferred size differs: {name}")
        require_digest(path, record["sha256"])
    identity = read_json(run / "run_manifest.json")
    status = read_json(run / "completion.json")
    require(identity["commit"] == expected_commit and identity["protocol_sha256"] == digest(PROTOCOL_PATH)
            and identity["inputs_sha256"] == digest(INPUTS_PATH), "Transferred commit, protocol or inputs differ")
    if status["passed"]:
        expected = {f"predictions/{key}/{variant}/predictions.npz" for key in INPUTS["references"]
                    for variant in ("corrupted", "enhanced")}
        actual = {name for name in inventory if name.endswith("/predictions.npz")}
        require(expected == actual and status["exports"] == len(expected), "Incomplete pilot prediction matrix")
        for name in expected:
            folder = name.rsplit("/", 1)[0]
            require({f"{folder}/{artifact}" for artifact in ("metadata.json", "effective_config.yaml",
                     "historical_export_metadata.json")}.issubset(inventory), f"Incomplete prediction artifacts: {folder}")
            if folder.endswith("/corrupted"):
                require(f"{folder}/parity.json" in inventory and read_json(run / folder / "parity.json")["passed"],
                        f"Archived control did not pass: {folder}")
        require({"metrics.csv", "budget.json", "preflight.json"}.issubset(inventory), "Missing pilot summary")
        with (run / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = {f"predictions/{condition_key(row['dataset'], row['method'], row['group'], row['corruption'], row['level'])}"
                f"/{row['variant']}/predictions.npz" for row in rows}
        require(keys == expected and len(rows) == len(expected)
                and all(int(row["seed"]) == SEED and int(row["n"]) == PILOT["images_per_group"]
                        and int(row["batch_slots"]) == PILOT["historical_batch_size"] for row in rows),
                "Transferred metrics do not match the locked pilot matrix")
    else:
        require(allow_failed, "Pilot archive reports a failed run")
    return {"passed": status["passed"], "verified_files": len(inventory), "commit": identity["commit"],
            "exports": status["exports"], "manifest_sha256": digest(run / "SHA256SUMS.json")}


def budget(history, metrics, initializations):
    sizes = {}
    for dataset in PILOT["datasets"]:
        for group in PILOT["groups"]:
            manifest = history / "records" / "manifests" / dataset / f"seed{SEED}" / f"test_{group}.json"
            sizes[dataset, group] = len(read_json(manifest))
    seed_counts = {method: len(seeds) for method, seeds in PROTOCOL["planned_formal_seeds"].items()}
    estimates = {}
    for variant in ("corrupted", "enhanced"):
        forward_seconds = preprocessing_seconds = overhead = volume = 0.0
        for row in metrics:
            if row["variant"] == variant:
                copies = seed_counts[row["method"]]
                count = sizes[row["dataset"], row["group"]]
                scale = count / row["n"]
                forward_seconds += row["forward_seconds"] * math.ceil(count / row["batch_slots"]) * copies
                preprocessing_seconds += max(0, row["evaluation_seconds"] - row["forward_seconds"]) * scale * copies
                overhead += max(0, row["wall_seconds"] - row["evaluation_seconds"]) * copies
                volume += row["export_bytes"] * scale * copies
        estimates[variant] = {"estimated_forward_seconds": forward_seconds,
                              "estimated_preprocessing_and_transfer_seconds": preprocessing_seconds,
                              "estimated_fixed_export_seconds": overhead,
                              "rough_output_bytes": round(volume)}
    return {"pilot_exports": len(metrics), "pilot_output_rows": sum(row["n"] for row in metrics),
            "pilot_forward_slots_including_padding": sum(row["batch_slots"] for row in metrics),
            "peak_allocated_gib": max(row["peak_memory_gib"] for row in metrics),
            "peak_reserved_gib": max(row["peak_reserved_memory_gib"] for row in metrics),
            "peak_initialization_allocated_gib": max(row["peak_allocated_gib"] for row in initializations),
            "formal_estimates": estimates, "estimated_shared_initialization_seconds": sum(
                row["seconds"] * seed_counts[row["method"]] for row in initializations),
            "basis": "Scale padded 100-slot CUDA-event timing by the full batch count and other evaluation time by image count. "
                     "Treat export overhead as fixed and scale compressed output volume by image count. Estimates are approximate."}


def metric_row(key, variant, directory, info, forward_seconds, parity=None):
    dataset, method, _, group, condition = key.split("/")
    corruption, level = condition.split("_L")
    return dict(dataset=dataset, method=method, seed=SEED, group=group, corruption=corruption, level=level,
                variant=variant, n=info["n"], evaluation_seconds=info["evaluation_seconds"],
                batch_slots=PILOT["historical_batch_size"], forward_seconds=forward_seconds,
                wall_seconds=info["export_wall_seconds"], peak_memory_gib=info["peak_memory_gib"],
                peak_reserved_memory_gib=info["peak_reserved_memory_gib"],
                export_bytes=sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
                predictions_sha256=info["predictions_sha256"],
                control_passed=parity["passed"] if parity else None,
                max_abs_logit_error=parity["max_abs_logit_error"] if parity else None,
                top1_agreement=parity["top1_agreement"] if parity else None)


def pilot(history, reference, device, run, metrics, initializations):
    import torch

    common, inference = import_history(history)
    from course_eval.metrics import compare_clean

    thresholds = read_json(history / "course_eval" / "protocol_v1.json")["clean_thresholds"]
    shared = {}
    for dataset, method, group in COMBINATIONS:
        rows = select_samples(history, dataset, group)
        classnames = common.classnames(dataset, group)
        block = run / "predictions" / dataset / method / f"seed{SEED}" / group
        controls = {}
        with inference.restored_model(dataset, method, SEED, group, block, device) as harness:
            initializations.append(dict(dataset=dataset, method=method, group=group, seconds=harness.initialization_seconds,
                                        peak_allocated_gib=harness.initialization_peak_memory_gib))
            for corruption, level in CONDITIONS:
                key = condition_key(dataset, method, group, corruption, level)
                known, former, records = subset_reference(common, reference, key, rows, classnames)
                directory = run / "predictions" / key / "corrupted"
                with historical_batch(harness) as forward_time:
                    inference.export(harness, directory, corruption, level, rows=rows)
                    seconds = forward_time()
                arrays, current = check_export(common, directory, known, former, records, "corrupted", key)
                comparison = compare_clean(arrays, known, thresholds)
                write_json(directory / "parity.json", comparison)
                metrics.append(metric_row(key, "corrupted", directory, current, seconds, comparison))
                require(comparison["passed"], f"Frozen Corrupted logits differ: {key}. See {directory / 'parity.json'}")
                controls[corruption, level] = (known, former, records)
                print(f"{key}/corrupted passed, max |Δlogit| = {comparison['max_abs_logit_error']:.6g}", flush=True)
            for corruption, level in CONDITIONS:
                key = condition_key(dataset, method, group, corruption, level)
                directory = run / "predictions" / key / "enhanced"
                with historical_batch(harness) as forward_time:
                    export_enhanced(inference, harness, directory, corruption, level, rows)
                    seconds = forward_time()
                known, former, records = controls[corruption, level]
                _, current = check_export(common, directory, known, former, records, "enhanced", key)
                remember_inputs(shared, dataset, corruption, level, current["sample_records"])
                metrics.append(metric_row(key, "enhanced", directory, current, seconds))
                print(f"{key}/enhanced passed", flush=True)
        del harness
        gc.collect()
        torch.cuda.empty_cache()
    return len(shared)


def run_pilot(args, history, reference, identity):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id), "Invalid pilot run ID")
    namespace = CHECKOUT / "outputs" / "course_enhancement_v1"
    require(namespace.resolve() == namespace, "Pilot output namespace must stay inside the checkout")
    run = namespace / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "run_manifest.json", dict(identity, scope="pilot", started_at=time_utc(), history=str(history),
                                              pilot_configuration=PILOT,
                                              reference=str(reference), frozen_reference=PROTOCOL["reference_run"],
                                              reference_manifest_sha256=INPUTS["formal_manifest_sha256"]))
    metrics, initializations = [], []
    status = {"passed": False}
    try:
        readiness = check_environment(args.device, history, reference)
        write_json(run / "preflight.json", readiness)
        require(readiness["passed"], f"Preflight failed. See {run / 'preflight.json'}")
        shared_count = pilot(history, reference, args.device, run, metrics, initializations)
        require(len(metrics) == 2 * len(INPUTS["references"]) and shared_count == 192, "Unexpected pilot coverage")
        write_json(run / "budget.json", budget(history, metrics, initializations))
        status = {"passed": True, "unique_enhanced_inputs_across_methods": shared_count}
    except Exception:
        (run / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if metrics:
            with (run / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
        write_json(run / "completion.json", dict(status, exports=len(metrics), finished_at=time_utc()))
        seal(run)
    print(f"Pilot artifacts sealed in {run}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "pilot"):
        command = commands.add_parser(name)
        command.add_argument("--history", type=Path, required=True)
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--commit", required=True)
        command.add_argument("--device", default="cuda:0")
        if name == "pilot":
            command.add_argument("--run-id", required=True)
    command = commands.add_parser("verify")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--commit", required=True)
    command.add_argument("--allow-failed", action="store_true")
    command.add_argument("--manifest-sha256", required=True, help="Digest recorded over SSH on the remote host")
    args = parser.parse_args(argv)
    if args.command == "verify":
        require_digest(args.run_dir / "SHA256SUMS.json", args.manifest_sha256)
        print(json.dumps(verify(args.run_dir, args.commit, args.allow_failed), indent=2))
        return 0
    history, reference = args.history.resolve(), args.reference.resolve()
    identity = source_identity(args.commit)
    if args.command == "preflight":
        report = check_environment(args.device, history, reference)
        print(json.dumps(dict(identity, **report), ensure_ascii=False, indent=2))
        return int(not report["passed"])
    run_pilot(args, history, reference, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
