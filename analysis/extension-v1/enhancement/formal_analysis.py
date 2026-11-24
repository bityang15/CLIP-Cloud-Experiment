"""Collect independent CPU image quality, then analyze the backed-up formal logits locally."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from formal import validate_combination, verify_formal
from formal_inputs import (CONDITIONS, GROUPS, LOCK, MATRIX, QUALITY_ROWS, formal_identity, import_archive, import_history,
                           key_for, load_reference, split_key)
from pilot import write_json
from pilot_inputs import digest, read_json, require, require_digest


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fresh_output(path, run, reference, history):
    path = path.resolve()
    protected = ("code", "course_eval", "datasets", "records", "weights", "execution", "outputs/B", "outputs/E",
                 "outputs/course_robustness_v1")
    require(all(not path.is_relative_to(root.resolve()) for root in (run, reference, *(history / item for item in protected))),
            "Output must be separate from sealed or archival data")
    path.mkdir(parents=True, exist_ok=False)
    return path


def difference(after, before):
    return 0.0 if after == before else after - before


def quality_pair(clean, corrupted, enhanced):
    from enhancement import quality_metrics

    before_psnr, before_ssim = quality_metrics(clean, corrupted)
    after_psnr, after_ssim = quality_metrics(clean, enhanced)
    return dict(corrupted_psnr_db=before_psnr, enhanced_psnr_db=after_psnr,
                delta_psnr_db=difference(after_psnr, before_psnr), corrupted_ssim=before_ssim,
                enhanced_ssim=after_ssim, delta_ssim=difference(after_ssim, before_ssim))


def rgb_sha(image):
    return sha256(image.tobytes()).hexdigest()


def quality_rows(history, reference, run, identity):
    """Use one verified set of pixels for each dataset, split, corruption, and level."""
    from PIL import Image
    from enhancement_v1 import enhancer_for

    common, inference = import_history(history)
    from course_eval.corruptions import corrupt
    from course_eval.preprocessing import CourseTransform

    data_root = (history / "datasets").resolve()
    operators = {name: enhancer_for(name) for name, _ in CONDITIONS}
    for dataset, group in GROUPS:
        clean, _, clean_records = load_reference(common, reference, key_for(dataset, "clip", 1, group, "clean", 0))
        cfg, _, _ = inference.configuration(dataset, "clip", 1, group, run)
        spatial = CourseTransform(dataset, cfg=cfg).spatial
        variants = {}
        for corruption, level in CONDITIONS:
            key = key_for(dataset, "clip", 1, group, corruption, level)
            control = load_reference(common, reference, key)
            _, meta = validate_combination(common, run / "predictions" / key / "enhanced", key, identity, reference=control)
            variants[corruption, level] = (control[2], {row["image_id"]: row for row in meta["sample_records"]})
        for image_id in clean["image_ids"].tolist():
            original = clean_records[image_id]
            path = (data_root / image_id).resolve(strict=True)
            require(path.is_relative_to(data_root), f"Image escapes data root: {image_id}")
            require_digest(path, original["source_sha256"])
            with Image.open(path) as image:
                clean_rgb = np.array(spatial(image.convert("RGB")), dtype=np.uint8)
            require(rgb_sha(clean_rgb) == original["rgb_uint8_sha256"], f"Clean crop hash differs: {image_id}")
            for (corruption, level), (controls, enhanced) in variants.items():
                degraded = corrupt(clean_rgb, corruption, level, dataset=dataset, image_id=image_id)
                recovered = operators[corruption](degraded)
                require(rgb_sha(degraded) == controls[image_id]["rgb_uint8_sha256"] and
                        rgb_sha(recovered) == enhanced[image_id]["enhanced_rgb_uint8_sha256"],
                        f"Corrupted or Enhanced pixel hash differs: {image_id}/{corruption}/L{level}")
                yield dict(dataset=dataset, group=group, corruption=corruption, level=level, image_id=image_id,
                           **quality_pair(clean_rgb, degraded, recovered))


def collect_quality(history, reference, run, output, identity, run_hash):
    require_digest(reference / "run_manifest.json", LOCK["formal_manifest_sha256"])
    rows = iter(quality_rows(history, reference, run, identity))
    quality = output / "image_quality.csv"
    first = next(rows)
    count = 1
    with quality.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first), lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        for row in rows:
            writer.writerow(row)
            count += 1
            if count % 3000 == 0:
                print(f"Quality: {count}/{QUALITY_ROWS}", flush=True)
    require(count == QUALITY_ROWS, "Incomplete independent image-quality export")
    write_json(output / "quality_manifest.json", {"commit": identity["commit"], "run_inventory_sha256": run_hash,
                "quality_sha256": digest(quality), "n": count, "reference_manifest_sha256": LOCK["formal_manifest_sha256"]})


def transitions(reference, enhanced):
    labels = reference["labels"]
    base, changed = reference["logits"].argmax(1), enhanced["logits"].argmax(1)
    was_correct, is_correct = base == labels, changed == labels
    sets = {"cc": was_correct & is_correct, "cw": was_correct & ~is_correct,
            "wc": ~was_correct & is_correct, "ww": ~was_correct & ~is_correct}
    counts = {f"{key}_count": int(mask.sum()) for key, mask in sets.items()}
    rates = {f"{key}_pct": float(mask.mean() * 100) for key, mask in sets.items()}
    return dict(n=len(labels), **counts, **rates, correctness_changed_pct=rates["cw_pct"] + rates["wc_pct"],
                top1_changed_pct=float((base != changed).mean() * 100),
                ww_top1_changed_count=int((sets["ww"] & (base != changed)).sum()))


def paired_metrics(base, new):
    a, b = base
    x, y = new

    def h(first, second):
        return 2 * first * second / (first + second) if first + second else 0.0

    values = {"base_accuracy_pct": (a["accuracy"], b["accuracy"]), "new_accuracy_pct": (x["accuracy"], y["accuracy"]),
              "h_pct": (h(a["accuracy"], x["accuracy"]), h(b["accuracy"], y["accuracy"])),
              "new_nll": (x["nll"], y["nll"])}
    for bins in (10, 15, 20):
        values[f"new_ece_{bins}_pct"] = x[f"ece_{bins}"], y[f"ece_{bins}"]
    result = {}
    for name, (before, after) in values.items():
        delta = f"{name.removesuffix('_pct')}_pp" if name.endswith("_pct") else name
        result.update({f"corrupted_{name}": before, f"enhanced_{name}": after, f"delta_{delta}": after - before})
    return result


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for values, items in sorted(groups.items()):
        fields = [key for key in items[0] if key not in (*keys, "seed")]
        seed_count = len({item["seed"] for item in items})
        require(seed_count == len(items) == (1 if items[0]["method"] == "clip" else 3), "Wrong independent seed count")
        numbers = np.array([[item[field] for field in fields] for item in items], dtype=np.float64)
        means = dict(zip((f"{f}_mean" for f in fields), numbers.mean(axis=0).tolist()))
        sd = numbers.std(axis=0, ddof=1).tolist() if seed_count > 1 else [None] * len(fields)
        result.append(dict(zip(keys, values), effective_seed_count=seed_count, **means,
                           **dict(zip((f"{f}_std" for f in fields), sd))))
    return result


def quality_summary(common, csv_path):
    expected = {(dataset, group, corruption, level, row["image_id"])
                for dataset, group in GROUPS for row in common.sample_manifest(dataset, 1, group)[1]
                for corruption, level in CONDITIONS}
    groups = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["dataset"], row["group"], row["corruption"], int(row["level"]), row["image_id"]
            require(key in expected, f"Unexpected or repeated independent quality entry: {key}")
            expected.remove(key)
            numbers = {k: float(row[k]) for k in ("corrupted_psnr_db", "enhanced_psnr_db", "delta_psnr_db",
                                                   "corrupted_ssim", "enhanced_ssim", "delta_ssim")}
            require(not any(math.isnan(value) for value in numbers.values()), "Undefined per-image quality score")
            for field in ("psnr_db", "ssim"):
                require(math.isclose(numbers[f"delta_{field}"], difference(numbers[f"enhanced_{field}"],
                                    numbers[f"corrupted_{field}"]), abs_tol=1e-12), "Quality delta differs")
            groups[key[:4]].append(numbers)
    require(not expected, "Independent image-quality table is incomplete")
    result = []
    for key, values in sorted(groups.items()):
        aggregate = {}
        for field in values[0]:
            column = [row[field] for row in values]
            aggregate[f"{field}_mean"] = None if math.inf in column and -math.inf in column else sum(column) / len(column)
            aggregate[f"{field}_finite_n"] = sum(math.isfinite(value) for value in column)
        result.append(dict(zip(("dataset", "group", "corruption", "level"), key), n=len(values), **aggregate))
    return result


def analyze(history, reference, run, output, identity, quality_csv):
    common = import_archive(history)
    from course_eval.metrics import all_metrics

    require_digest(reference / "run_manifest.json", LOCK["formal_manifest_sha256"])
    scores, changes = defaultdict(dict), []
    for key in MATRIX:
        dataset, method, seed, group, corruption, level = split_key(key)
        control = load_reference(common, reference, key)
        arrays, _ = validate_combination(common, run / "predictions" / key / "enhanced", key, identity, reference=control)
        before = control[0]
        score = all_metrics(before["logits"], before["labels"]), all_metrics(arrays["logits"], arrays["labels"])
        scores[dataset, method, seed, corruption, level][group] = score
        row = transitions(before, arrays)
        require(math.isclose(score[1]["accuracy"] - score[0]["accuracy"], row["wc_pct"] - row["cw_pct"], abs_tol=1e-10),
                f"Transition accuracy identity failed: {key}")
        changes.append(dict(dataset=dataset, method=method, seed=seed, group=group, corruption=corruption, level=level, **row))
    keys = ("dataset", "method", "corruption", "level")
    paired = [dict(zip(("dataset", "method", "seed", "corruption", "level"), key), **paired_metrics(groups["base"], groups["new"]))
              for key, groups in sorted(scores.items())]
    quality = quality_summary(common, quality_csv)
    for name, rows in (("paired_per_seed.csv", paired), ("paired_summary.csv", summarize(paired, keys)),
                       ("transitions_per_seed.csv", changes), ("transitions_summary.csv", summarize(changes, (*keys, "group"))),
                       ("image_quality_summary.csv", quality)):
        write_csv(output / name, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("quality", "analyze"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--history", type=Path, required=True)
        cmd.add_argument("--reference", type=Path, required=True)
        cmd.add_argument("--run-dir", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--commit", required=True)
        cmd.add_argument("--manifest-sha256", required=True)
    analysis = commands.choices["analyze"]
    analysis.add_argument("--quality-dir", type=Path, required=True)
    analysis.add_argument("--quality-manifest-sha256", required=True)
    args = parser.parse_args()
    history, reference, run = args.history.resolve(), args.reference.resolve(), args.run_dir.resolve()
    receipt = verify_formal(run, args.commit, args.manifest_sha256)
    identity = read_json(run / "run_manifest.json")
    if args.command == "quality":
        current = formal_identity(args.commit)
        require(all(identity[key] == value for key, value in current.items()), "Quality runtime differs from the inference commit")
        output = fresh_output(args.out, run, reference, history)
        collect_quality(history, reference, run, output, identity, receipt["manifest_sha256"])
    else:
        manifest = args.quality_dir / "quality_manifest.json"
        require_digest(manifest, args.quality_manifest_sha256)
        lock = read_json(manifest)
        require(lock["commit"] == args.commit and lock["run_inventory_sha256"] == receipt["manifest_sha256"] and
                lock["reference_manifest_sha256"] == LOCK["formal_manifest_sha256"] and lock["n"] == QUALITY_ROWS,
                "Image-quality data do not belong to this formal run")
        quality_csv = args.quality_dir / "image_quality.csv"
        require_digest(quality_csv, lock["quality_sha256"])
        output = fresh_output(args.out, run, reference, history)
        analyze(history, reference, run, output, identity, quality_csv)
    print(f"Wrote {args.command} outputs to {output}")


if __name__ == "__main__":
    main()
