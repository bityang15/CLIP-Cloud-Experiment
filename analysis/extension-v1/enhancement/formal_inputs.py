"""Frozen matrix, full reference parity, and hardware readiness for the formal run."""

import importlib
from itertools import product
import shutil
import subprocess
import sys

from pilot import remember_inputs
from pilot_inputs import (CHECKOUT, CONDITIONS, HERE, PILOT, PROTOCOL, check_environment, digest,
                          import_history, read_json, require, require_digest, source_identity)

LOCK_PATH = HERE / "formal_inputs_v1.json"
LOCK = read_json(LOCK_PATH)
PILOT_RECEIPT = HERE / "runs" / "enhancement-v1-pilot-20261124-01" / "run_record.json"
PILOT_BUDGET = PILOT_RECEIPT.with_name("RESOURCE_BUDGET.json")
BLOCKS = [(dataset, method, seed, group) for dataset, method, group in product(
    PILOT["datasets"], PILOT["methods"], PILOT["groups"]) for seed in PROTOCOL["planned_formal_seeds"][method]]
GROUPS = list(product(PILOT["datasets"], PILOT["groups"]))
BATCH_SIZE = PILOT["historical_batch_size"]


def key_for(dataset, method, seed, group, corruption, level):
    return f"{dataset}/{method}/seed{seed}/{group}/{corruption}_L{level}"


def split_key(key):
    dataset, method, seed, group, condition = key.split("/")
    corruption, level = condition.rsplit("_L", 1)
    return dataset, method, int(seed[4:]), group, corruption, int(level)


MATRIX = [key_for(*block, *condition) for block, condition in product(BLOCKS, CONDITIONS)]
CLEAN = [key_for(dataset, "clip", 1, group, "clean", 0) for dataset, group in GROUPS]
ROW_COUNTS = {dataset_group: LOCK["references"][key][2] for dataset_group, key in zip(GROUPS, CLEAN)}
QUALITY_ROWS = sum(ROW_COUNTS.values()) * len(CONDITIONS)
# Run directory and transfer archive, plus per-image quality rows and space for logs.
OUTPUT_BYTES = read_json(PILOT_BUDGET)["scenario_totals"]["enhanced_only_with_archived_controls"]["estimated_output_bytes"]
SPACE_BYTES = 2 * (OUTPUT_BYTES + QUALITY_ROWS * 512 + 64 * 1024**2)


def check_matrix():
    require(len(BLOCKS) == 40 and len(MATRIX) == len(set(MATRIX)) == 240 and len(CLEAN) == 4,
            "Formal matrix size differs from the V1 plan")
    require(set(LOCK["references"]) == set(MATRIX) | set(CLEAN), "Historical reference lock differs from the matrix")
    require(sum(LOCK["references"][key][2] for key in MATRIX) == 249300 and QUALITY_ROWS == 24930,
            "Formal sample counts differ from V1")


def formal_identity(commit):
    identity = source_identity(commit)
    require_digest(PILOT_RECEIPT, LOCK["pilot_record_sha256"])
    pilot = read_json(PILOT_RECEIPT)
    require(pilot["passed"] and pilot["runtime_commit"] == LOCK["base_commit"] and
            pilot["source_sha256"] == identity["source_sha256"], "The stage 3 executable baseline has changed")
    files = {}
    for name in ("formal.py", "formal_inputs.py", LOCK_PATH.name, "formal_analysis.py", "enhancement.py",
                 PILOT_RECEIPT, PILOT_BUDGET):
        path = HERE / name
        rel = path.relative_to(CHECKOUT).as_posix()
        expected = subprocess.check_output(["git", "-C", str(CHECKOUT), "show", f"{commit}:{rel}"])
        require(path.read_bytes() == expected, f"Formal working-tree bytes differ: {rel}")
        files[rel] = digest(path)
    return dict(identity, formal_source_sha256=files, formal_inputs_sha256=digest(LOCK_PATH),
                frozen_code_commit=LOCK["base_commit"], stage3_record_commit=LOCK["pilot_record_commit"],
                reference_manifest_sha256=LOCK["formal_manifest_sha256"])


def import_archive(history):
    if str(history) not in sys.path:
        sys.path.insert(0, str(history))
    common = importlib.import_module("course_eval.common")
    require(common.REPO == history, "Imported common.py from the wrong historical tree")
    return common


def indexed_records(metadata, image_ids):
    records = metadata["sample_records"]
    require([row["image_id"] for row in records] == list(image_ids), "Stored input records differ from NPZ image order")
    return {row["image_id"]: row for row in records}


def load_reference(common, reference, key):
    import numpy as np

    directory = reference / "predictions" / key
    meta_hash, pred_hash, n = LOCK["references"][key]
    require_digest(directory / "metadata.json", meta_hash)
    arrays, meta = common.load_predictions(directory)
    observed = tuple(meta[name] for name in ("dataset", "method", "seed", "candidate_group", "corruption", "level"))
    require(observed == split_key(key) and meta["n"] == n and meta["sample_scope"] == "full" and
            meta["predictions_sha256"] == pred_hash and meta["raw_logit_scale"], f"Wrong reference metadata: {key}")
    require(np.isfinite(arrays["logits"]).all(), f"Nonfinite reference logits: {key}")
    records = indexed_records(meta, arrays["image_ids"])
    return arrays, meta, records


def validate_archives(common, reference, inference=None):
    """Validate all paired controls and the four canonical clean cohorts without model loading."""
    require_digest(reference / "run_manifest.json", LOCK["formal_manifest_sha256"])
    cohorts, signatures = {}, {}
    for key in CLEAN + MATRIX:
        dataset, method, seed, group, corruption, level = split_key(key)
        arrays, meta, _ = load_reference(common, reference, key)
        manifest, rows = common.sample_manifest(dataset, seed, group)
        ids, labels, names = (arrays[name].tolist() for name in ("image_ids", "labels", "classnames"))
        require(digest(manifest) == meta["sample_manifest_sha256"] and ids == [row["image_id"] for row in rows]
                and labels == [row["label"] for row in rows] and names == common.classnames(dataset, group),
                f"Archived cohort or class mapping differs: {key}")
        require(all(0 <= label < len(names) and names[label] == row["classname"] for label, row in zip(labels, rows)),
                f"Label mapping differs: {key}")
        cohort = list(zip(ids, labels)), names
        require(cohorts.setdefault((dataset, group), cohort) == cohort, f"Test cohort differs across seeds: {key}")
        require_digest(reference / "predictions" / key / "effective_config.yaml", meta["config_sha256"])
        remember_inputs(signatures, dataset, corruption, level, meta["sample_records"])
        if inference is not None:
            require(meta["parameters"] == inference.parameters(corruption, level), f"Corruption parameters differ: {key}")
            cfg, historical, path = inference.configuration(dataset, method, seed, group, CHECKOUT)
            require(cfg.DATALOADER.TEST.BATCH_SIZE == BATCH_SIZE and cfg.TEST.SPLIT == "test" and
                    digest(path) == meta["historical_config_sha256"] and
                    historical["checkpoint_sha256"] == meta["checkpoint_sha256"], f"Frozen configuration differs: {key}")
    require(len(signatures) == QUALITY_ROWS + sum(ROW_COUNTS.values()), "Independent image coverage differs")
    return {"corrupted": len(MATRIX), "clean": len(CLEAN), "independent_corrupted_images": QUALITY_ROWS,
            "prediction_rows": sum(LOCK["references"][key][2] for key in MATRIX)}


def full_preflight(identity, history, reference, device, output_volume=CHECKOUT):
    check_matrix()
    readiness = check_environment(device, history, reference)
    report = dict(identity, runtime=readiness, expected_exports=len(MATRIX), required_free_bytes=SPACE_BYTES,
                  output_free_bytes=shutil.disk_usage(output_volume).free, errors=list(readiness["errors"]))

    def check(name, operation):
        try:
            report[name] = operation()
        except Exception as error:
            report["errors"].append(f"{name}: {type(error).__name__}: {error}")

    if report["output_free_bytes"] < SPACE_BYTES:
        report["errors"].append("Insufficient space for the run and its transfer archive")

    def image_hashes():
        root = (history / "datasets").resolve()
        manifest = read_json(history / "records" / "data_manifest.json")["datasets"]
        result = {}
        for dataset in PILOT["datasets"]:
            ids = {row["image_id"] for group in PILOT["groups"] for row in read_json(
                history / "records" / "manifests" / dataset / "seed1" / f"test_{group}.json")}
            require(len(ids) == sum(n for (d, _), n in ROW_COUNTS.items() if d == dataset), f"Wrong image set: {dataset}")
            for image_id in sorted(ids):
                path = (root / image_id).resolve(strict=True)
                require(path.is_relative_to(root), f"Image path escapes data root: {image_id}")
                require_digest(path, manifest[dataset]["images"][image_id]["sha256"])
            result[dataset] = len(ids)
        return result

    def checkpoint_hashes():
        expected = {row["path"]: row for row in read_json(history / "records" / "checkpoints.json")
                    if row["path"].startswith("outputs/B/")}
        actual = readiness["prompt_checkpoints"]
        require(len(expected) == 18 and set(actual) == set(expected), "Incomplete formal checkpoint inventory")
        for path, row in expected.items():
            if not actual[path]["sha256_verified"]:
                require_digest(history / path, row["sha256"])
        return {"files_hashed": len(expected), "shared_weight": readiness["shared_weight"]["sha256"]}

    check("all_prompt_checkpoints", checkpoint_hashes)
    check("original_test_images_hashed", image_hashes)
    if readiness["passed"]:
        common, inference = import_history(history)
        check("reference_inventory", lambda: validate_archives(common, reference, inference))
    report["passed"] = not report["errors"]
    return report
