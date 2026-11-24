"""CPU-only tests for the full exporter, resume transaction, and paired analysis."""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from yacs.config import CfgNode

import enhancement_v1
import formal
import formal_analysis as analysis
import formal_inputs as inputs
from pilot_inputs import PROTOCOL_PATH, digest, read_json

REFERENCE = inputs.CHECKOUT / "outputs" / "course_robustness_v1" / "formal-v1-confirmed-01"


def test_frozen_full_matrix():
    inputs.check_matrix()
    assert len(inputs.MATRIX) == 240 and len(inputs.BLOCKS) == 40
    assert {seed for _, method, seed, _ in inputs.BLOCKS if method == "clip"} == {1}
    assert {seed for _, method, seed, _ in inputs.BLOCKS if method == "tcpt"} == {1, 2, 3}
    assert inputs.QUALITY_ROWS == 24930


@pytest.mark.skipif(not REFERENCE.exists(), reason="local V1 archive is required")
def test_all_244_reference_artifacts_match_the_formal_archive():
    common = inputs.import_archive(inputs.CHECKOUT)
    summary = inputs.validate_archives(common, REFERENCE)
    assert summary == {"corrupted": 240, "clean": 4, "prediction_rows": 249300,
                       "independent_corrupted_images": 24930}


def test_only_one_writer_can_hold_the_run_lock(tmp_path):
    with formal.run_lock(tmp_path), pytest.raises(RuntimeError, match="Another writer"):
        with formal.run_lock(tmp_path):
            pytest.fail("second writer entered")
    with formal.run_lock(tmp_path):
        assert (tmp_path / "run.lock").exists()


def test_run_identity_is_immutable_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, "MATRIX", ["a"])
    path = formal.setup_run(tmp_path, "test-run", False)
    first = formal.bind_run(path, {"commit": "x"}, tmp_path, tmp_path, "cuda:0", False)
    assert formal.setup_run(tmp_path, "test-run", True) == path
    assert formal.bind_run(path, {"commit": "x"}, tmp_path, tmp_path, "cuda:0", True) == first
    with pytest.raises(ValueError, match="Resume requires"):
        formal.bind_run(path, {"commit": "y"}, tmp_path, tmp_path, "cuda:0", True)
    with pytest.raises(FileExistsError):
        formal.setup_run(tmp_path, "test-run", False)


def test_archived_full_loader_uses_natural_tail_and_no_padding(tmp_path, monkeypatch):
    root, stage = tmp_path / "data", tmp_path / "output"
    cfg = CfgNode({"DATASET": {"ROOT": str(root)}, "DATALOADER": {"TEST": {"BATCH_SIZE": 100}}})
    identity = {"commit": "example", "protocol_sha256": "protocol", "formal_inputs_sha256": "inputs"}
    row_list = [{"image_id": f"image-{i}.jpg"} for i in range(203)]
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(enhancement_v1, "enhancer_for", lambda *_: lambda image: image.copy())
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    class Dataset:
        def __init__(self, dataset, rows, corruption, level, cfg):
            self.root, self.rows, self.input_records = root, rows, []
            self.transform = SimpleNamespace(rgb=lambda *_: rgb.copy(), corruption=corruption, tensor=None)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            image_id = self.rows[index]["image_id"]
            output = self.transform.rgb(None, image_id)
            self.input_records.append(dict(image_id=image_id, rgb_uint8_sha256=sha256(output.tobytes()).hexdigest()))
            return index

    def frozen_forward(_):
        pytest.fail("This fixture must never call model forward")

    def archived_test(harness, split):
        assert split == "test" and harness.model_inference is frozen_forward
        batches = [batch.tolist() for batch in harness.test_loader]
        assert [len(batch) for batch in batches] == [100, 100, 3]
        assert [item for batch in batches for item in batch] == list(range(203))
        formal.write_json(stage / "metadata.json", {"n": 203})

    inference = SimpleNamespace(ManifestImages=Dataset, parameters=lambda *_: {}, sample_manifest=lambda *_: (manifest, row_list))
    harness = SimpleNamespace(experiment=SimpleNamespace(ExperimentMixin=SimpleNamespace(test=archived_test)),
                              historical_config=manifest, initialization_seconds=0, initialization_peak_memory_gib=0,
                              model_inference=frozen_forward)
    formal.export_full(inference, harness, stage, "dtd/maple/seed3/new/gaussian_blur_L2", cfg, row_list, identity)
    meta = read_json(stage / "metadata.json")
    assert meta["batch_size"] == 100 and meta["sample_scope"] == "full"
    assert len(meta["sample_records"]) == 203
    assert meta["sample_manifest_sha256"] == digest(manifest)
    assert harness.model_inference is frozen_forward


@pytest.fixture
def resume_fixture(tmp_path, monkeypatch):
    block = ("dtd", "maple", 1, "new")
    keys = [inputs.key_for(*block, "gaussian_blur", level) for level in (2, 3)]
    monkeypatch.setattr(formal, "MATRIX", keys)
    monkeypatch.setattr(formal, "BLOCKS", [block])
    monkeypatch.setattr(formal, "CONDITIONS", [("gaussian_blur", 2), ("gaussian_blur", 3)])
    monkeypatch.setattr(formal, "QUALITY_ROWS", 2)
    monkeypatch.setattr(formal, "LOCK", {"references": {key: ["meta", "pred", 1] for key in keys}})
    identity = {"commit": "fixture", "protocol_sha256": digest(PROTOCOL_PATH), "formal_inputs_sha256": digest(inputs.LOCK_PATH)}
    run = formal.setup_run(tmp_path, "resume-fixture", False)
    manifest = formal.bind_run(run, identity, tmp_path, tmp_path, "cuda:0", False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    config = SimpleNamespace(dump=lambda: "expected config\n")
    monkeypatch.setattr(formal, "expected_config", lambda *_: config)
    monkeypatch.setattr(formal, "load_reference", lambda *_: (None, None, None))
    model_calls = []

    @contextmanager
    def model(*args):
        model_calls.append(args)
        yield object()

    common = SimpleNamespace(sample_manifest=lambda *_: (None, [{}]),
                             load_predictions=lambda directory: (None, read_json(directory / "metadata.json")))
    monkeypatch.setattr(formal, "check_export", lambda common, directory, *args: common.load_predictions(directory))
    monkeypatch.setattr(formal, "import_history", lambda _: (common, SimpleNamespace(restored_model=model)))
    attempts = []
    fail = {keys[1]}

    def export(inference, harness, stage, key, cfg, rows, identity):
        attempts.append(key)
        stage.mkdir(parents=True)
        (stage / "predictions.npz").write_bytes(key.encode())
        if key in fail:
            fail.remove(key)
            raise RuntimeError("interrupted fixture")
        config_path = stage / "effective_config.yaml"
        config_path.write_bytes(config.dump().encode())
        metadata = dict.fromkeys(formal.METRIC_FIELDS, 0)
        metadata.update(n=1, batch_size=100, sample_scope="full", variant="enhanced", runtime_commit=identity["commit"],
                        enhancement_protocol_sha256=identity["protocol_sha256"], formal_inputs_sha256=identity["formal_inputs_sha256"],
                        enhancement=enhancement_v1.enhancement_parameters("gaussian_blur"), config_sha256=digest(config_path),
                        sample_records=[dict(image_id="one", rgb_uint8_sha256="rgb", tensor_sha256="tensor")])
        formal.write_json(stage / "metadata.json", metadata)
        formal.write_json(stage / "historical_export_metadata.json", {})

    monkeypatch.setattr(formal, "export_full", export)
    args = run, manifest, tmp_path, tmp_path, "cuda:0"
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        formal.process_run(*args)
    assert len(list(run.glob("failure-*.json"))) == 1
    return SimpleNamespace(run=run, args=args, keys=keys, attempts=attempts, model_calls=model_calls, manifest=manifest)


def test_resume_reuses_verified_output_and_quarantines_interruption(resume_fixture):
    fix = resume_fixture
    first = formal.combo_paths(fix.run, fix.keys[0])[1]
    before = digest(first / "predictions.npz")
    formal.process_run(*fix.args)
    assert fix.attempts == [*fix.keys, fix.keys[1]]
    assert digest(first / "predictions.npz") == before
    assert len(list((fix.run / "failed_staging").rglob("predictions.npz"))) == 1
    result = formal.verify_formal(fix.run, "fixture", digest(fix.run / "SHA256SUMS.json"))
    assert result["exports"] == 2 and result["passed"]
    (first / "predictions.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        formal.verify_formal(fix.run, "fixture")


def test_resume_detects_corrupt_completed_file_before_loading_model(resume_fixture):
    fix = resume_fixture
    directory = formal.combo_paths(fix.run, fix.keys[0])[1]
    (directory / "predictions.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="receipt or file hash differs"):
        formal.process_run(*fix.args)
    assert len(fix.model_calls) == 1


def test_resume_recomputes_effective_config_even_with_a_matching_receipt(resume_fixture):
    fix = resume_fixture
    key = fix.keys[0]
    directory = formal.combo_paths(fix.run, key)[1]
    (directory / "effective_config.yaml").write_bytes(b"tampered config")
    (directory / "receipt.json").unlink()
    formal.write_json(directory / "receipt.json", formal.receipt(directory, key, fix.manifest))
    with pytest.raises(ValueError, match="effective config differs"):
        formal.process_run(*fix.args)
    assert len(fix.model_calls) == 1


def test_paired_metrics_and_seed_statistics():
    def example(accuracy, nll, ece):
        return dict(accuracy=accuracy, nll=nll, **{f"ece_{bins}": ece for bins in (10, 15, 20)})

    paired = analysis.paired_metrics((example(80, 1, 10), example(84, 1, 10)),
                                     (example(40, 2, 20), example(60, 1.5, 17)))
    assert paired["delta_base_accuracy_pp"] == 4 and paired["delta_new_accuracy_pp"] == 20
    assert paired["delta_new_ece_15_pp"] == -3 and paired["delta_new_nll"] == -0.5
    assert paired["delta_h_pp"] == pytest.approx(2 * 84 * 60 / 144 - 2 * 80 * 40 / 120)
    rows = [dict(method="maple", seed=seed, delta=seed + 1) for seed in (1, 2, 3)]
    rows.append(dict(method="clip", seed=1, delta=2))
    clip, maple = analysis.summarize(rows, ("method",))
    assert clip["effective_seed_count"] == 1 and clip["delta_std"] is None
    assert maple["effective_seed_count"] == 3 and maple["delta_mean"] == 3 and maple["delta_std"] == 1


def test_transitions_distinguish_ww_class_change():
    labels = np.array([0, 0, 0, 0])
    scores = np.eye(3)
    a = {"labels": labels, "logits": scores[[0, 0, 1, 1]]}
    b = {"labels": labels, "logits": scores[[0, 1, 0, 2]]}
    result = analysis.transitions(a, b)
    assert all(result[f"{state}_count"] == 1 for state in ("cc", "cw", "wc", "ww"))
    assert result["correctness_changed_pct"] == 50 and result["top1_changed_pct"] == 75
    assert result["ww_top1_changed_count"] == 1


def test_quality_counts_independent_images_once(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "GROUPS", [("dtd", "new")])
    monkeypatch.setattr(analysis, "CONDITIONS", [("gaussian_blur", 2)])
    root = SimpleNamespace(sample_manifest=lambda *_: (None, [{"image_id": "x"}]))
    clean = np.zeros((224, 224, 3), dtype=np.uint8)
    scores = analysis.quality_pair(clean, clean, clean)
    assert scores["delta_psnr_db"] == scores["delta_ssim"] == 0
    record = dict(dataset="dtd", group="new", corruption="gaussian_blur", level=2, image_id="x", **scores)
    one = tmp_path / "one.csv"
    many = tmp_path / "many.csv"
    analysis.write_csv(one, [record])
    analysis.write_csv(many, [record, record])
    assert analysis.quality_summary(root, one)[0]["n"] == 1
    with pytest.raises(ValueError, match="repeated independent quality entry"):
        analysis.quality_summary(root, many)
