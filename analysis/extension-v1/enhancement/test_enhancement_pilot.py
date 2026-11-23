"""CPU-only coverage for the freeze, preprocessing adapter, and transfer gates."""

from hashlib import sha256
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageFilter
import pytest
import torch

import enhancement_v1 as enhancement
import pilot
from pilot_inputs import (CHECKOUT, COMBINATIONS, CONDITIONS, INPUTS, INPUTS_PATH, PILOT, PROTOCOL_PATH,
                          condition_key, digest, read_json, select_samples)


REFERENCE = CHECKOUT / "outputs" / "course_robustness_v1" / "formal-v1-confirmed-01"
ARCHIVE_PRESENT = (CHECKOUT / "course_eval" / "preprocessing.py").is_file() and REFERENCE.is_dir()
NEEDS_ARCHIVE = pytest.mark.skipif(not ARCHIVE_PRESENT, reason="requires the local frozen V1.0 archive")


@pytest.fixture
def rgb():
    return np.random.default_rng(1503).integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.mark.parametrize("profile,radius,percent,threshold", [("original", 1.5, 120, 3), ("candidate", 3.0, 150, 0)])
def test_both_unsharp_profiles_are_exact(rgb, profile, radius, percent, threshold):
    expected = np.array(Image.fromarray(rgb).filter(ImageFilter.UnsharpMask(radius, percent, threshold)))
    np.testing.assert_array_equal(enhancement.unsharp_profile(profile)(rgb), expected)
    if profile == "candidate":
        np.testing.assert_array_equal(enhancement.enhancer_for("gaussian_blur")(rgb), expected)


def test_keep_the_original_bilateral_and_lightness_only_clahe(rgb):
    expected_noise = cv2.bilateralFilter(rgb, 5, 50, 5, borderType=cv2.BORDER_REFLECT_101)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    expected_gamma = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    np.testing.assert_array_equal(enhancement.enhancer_for("gaussian_noise")(rgb), expected_noise)
    np.testing.assert_array_equal(enhancement.enhancer_for("gamma_darkening")(rgb), expected_gamma)


@pytest.mark.parametrize("corruption", enhancement.PROTOCOL["algorithms"])
def test_rgb_operator_is_deterministic_and_pixels_only(rgb, corruption, monkeypatch):
    operator = enhancement.enhancer_for(corruption)
    original = rgb.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("Tried to access an external image or file")

    with monkeypatch.context() as patches:
        patches.setattr(Path, "open", forbidden)
        patches.setattr(Image, "open", forbidden)
        patches.setattr(np, "load", forbidden)
        actual = operator(rgb)
        np.testing.assert_array_equal(operator(rgb), actual)
    assert actual.shape == (224, 224, 3) and actual.dtype == np.uint8
    assert not np.shares_memory(actual, rgb)
    np.testing.assert_array_equal(rgb, original)
    with pytest.raises(TypeError):
        operator(rgb.astype(np.float32))
    with pytest.raises(ValueError):
        operator(rgb[..., 0])


def test_padding_preserves_sample_positions_and_only_uses_group_images():
    images = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    positions = PILOT["sample_positions"]
    seen = []

    def fake_forward(padded):
        seen.append(padded.clone())
        # Depend on the original batch position to detect incorrect slicing.
        return padded + torch.arange(len(padded)).reshape(-1, 1)

    combined = pilot.padded_forward(images, fake_forward)
    assert len(seen) == 1 and tuple(seen[0].shape) == (100, 2)
    assert torch.equal(seen[0][positions], images)
    other = [i for i in range(100) if i not in positions]
    assert torch.equal(seen[0][other], images[0].expand(92, -1))
    assert torch.equal(combined, images + torch.tensor(positions).reshape(-1, 1))
    with pytest.raises(ValueError, match="eight-image"):
        pilot.padded_forward(images[:7], fake_forward)


@NEEDS_ARCHIVE
@pytest.mark.parametrize("corruption,level", CONDITIONS)
def test_adapter_resizes_corrupts_enhances_then_normalizes(rgb, corruption, level):
    from course_eval.preprocessing import CourseTransform

    image = Image.fromarray(np.pad(rgb, ((11, 27), (0, 18), (0, 0)), mode="reflect"))
    image_id = "dtd/images/synthetic.jpg"
    historical = CourseTransform("dtd", corruption, level)
    original = historical.rgb(image, image_id)
    seen = []

    def capture(corrupted_pixels):
        seen.append(corrupted_pixels.copy())
        return enhancement.enhancer_for(corruption)(corrupted_pixels)

    wrapped = pilot.EnhancedTransform(historical, capture)
    output = wrapped.rgb(image, image_id)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], original)
    np.testing.assert_array_equal(historical.rgb(image, image_id), original)
    want = enhancement.enhancer_for(corruption)(original)
    np.testing.assert_array_equal(output, want)
    assert wrapped.corrupted_hashes[image_id] == sha256(original.tobytes()).hexdigest()
    assert torch.equal(wrapped.tensor(Image.fromarray(output)), historical.tensor(Image.fromarray(want)))


def test_adapter_rejects_input_mutation(rgb):
    def mutate(pixels):
        pixels[0, 0, 0] ^= 1
        return pixels

    historical = SimpleNamespace(corruption="gaussian_blur", tensor=None, rgb=lambda *args: rgb.copy())
    with pytest.raises(ValueError, match="modified its input"):
        pilot.EnhancedTransform(historical, mutate).rgb(None, "test.jpg")


def test_enhanced_export_contract_without_model_forward(tmp_path, rgb):
    from yacs.config import CfgNode

    data_root = tmp_path / "datasets"
    ids = [f"dtd/images/fixture-{i}.jpg" for i in range(8)]
    rows = [{"image_id": image_id, "label": i % 2} for i, image_id in enumerate(ids)]
    manifest = tmp_path / "sample.json"
    manifest.write_text("fixture", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("fixture", encoding="utf-8")
    cfg = CfgNode({"DATASET": {"SUBSAMPLE_CLASSES": "base", "ROOT": ""}, "OUTPUT_DIR": "",
                   "DATALOADER": {"TEST": {"BATCH_SIZE": 100}}, "EXPERIMENT": {"EXPORT_DIR": ""}})
    cfg.freeze()
    harness = SimpleNamespace(cfg=cfg, dataset_key="dtd", historical_config=config, initialization_seconds=0,
                              initialization_peak_memory_gib=0, model_inference=None)

    class FixtureImages:
        root = data_root

        def __init__(self, dataset, selected, corruption, level, cfg):
            self.rows = selected
            self.input_records = []
            self.transform = SimpleNamespace(corruption=corruption, rgb=lambda image, key: rgb.copy(),
                                             tensor=lambda image: torch.from_numpy(np.array(image)))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            processed = self.transform.rgb(None, row["image_id"])
            self.input_records.append(dict(image_id=row["image_id"], rgb_uint8_sha256=sha256(processed.tobytes()).hexdigest()))
            return dict(img=self.transform.tensor(Image.fromarray(processed)), label=row["label"],
                        impath=str(data_root / row["image_id"]))

    def export_fixture(harness, folder):
        batch, = list(harness.test_loader)
        seen = [Path(path).relative_to(data_root).as_posix() for path in batch["impath"]]
        logits = np.zeros((len(seen), 2), dtype=np.float32)
        np.savez_compressed(folder / "predictions.npz", logits=logits, labels=batch["label"].numpy(),
                            image_ids=np.asarray(seen), classnames=np.asarray(["a", "b"]))
        metadata = {"n": len(seen), "config_sha256": harness.config_sha256}
        pilot.write_json(folder / "metadata.json", metadata)
        return metadata

    inference = SimpleNamespace(ManifestImages=FixtureImages, export_subset=export_fixture,
                                sample_manifest=lambda *args: (manifest, rows), parameters=lambda *args: {"sigma": 6})
    folder = tmp_path / "enhanced"
    metadata = pilot.export_enhanced(inference, harness, folder, "gaussian_blur", 3, rows)
    assert metadata["enhancement"] == enhancement.enhancement_parameters("gaussian_blur")
    assert metadata["config_sha256"] == digest(folder / "effective_config.yaml")
    assert metadata["sample_manifest_sha256"] == digest(manifest)
    assert read_json(folder / "historical_export_metadata.json")["n"] == 8
    expected = enhancement.enhancer_for("gaussian_blur")(rgb)
    for record in metadata["sample_records"]:
        assert record["corrupted_rgb_uint8_sha256"] == sha256(rgb.tobytes()).hexdigest()
        assert record["enhanced_rgb_uint8_sha256"] == sha256(expected.tobytes()).hexdigest()
    with np.load(folder / "predictions.npz", allow_pickle=False) as prediction:
        assert prediction["image_ids"].tolist() == ids
        assert prediction["labels"].tolist() == [row["label"] for row in rows]


@NEEDS_ARCHIVE
def test_fixed_sample_and_reference_matrix_matches_the_archives():
    from course_eval import common
    from course_eval.metrics import compare_clean

    assert digest(REFERENCE / "run_manifest.json") == INPUTS["formal_manifest_sha256"]
    expected = {condition_key(*group, *condition) for group, condition in product(COMBINATIONS, CONDITIONS)}
    assert set(INPUTS["references"]) == expected and len(expected) == 96
    ids = set()
    for dataset, method, group in COMBINATIONS:
        rows = select_samples(CHECKOUT, dataset, group)
        ids.update((dataset, row["image_id"]) for row in rows)
        assert len(rows) == 8
        for corruption, level in CONDITIONS:
            key = condition_key(dataset, method, group, corruption, level)
            subset, former, inputs = pilot.subset_reference(common, REFERENCE, key, rows, common.classnames(dataset, group))
            assert subset["logits"].shape == (8, former["classes"]) and len(inputs) == 8
            assert subset["image_ids"].tolist() == [row["image_id"] for row in rows]
    assert len(ids) == 32
    thresholds = read_json(CHECKOUT / "course_eval" / "protocol_v1.json")["clean_thresholds"]
    assert compare_clean(subset, subset, thresholds)["passed"]
    drifted = dict(subset, logits=(subset["logits"] + 0.03125))
    assert not compare_clean(drifted, subset, thresholds)["passed"]


@NEEDS_ARCHIVE
def test_windows_archive_content_matches_linux_source_except_line_endings():
    source = read_json(REFERENCE / "run_manifest.json")["sources"]
    for name, want in source.items():
        path = CHECKOUT / name
        assert digest(path) == want or sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == want, name


def test_cross_method_pixel_verification_detects_mismatches():
    record = {"image_id": "test.jpg", "rgb_uint8_sha256": "same", "tensor_sha256": "tensor"}
    cache = {}
    pilot.remember_inputs(cache, "dtd", "gaussian_blur", 2, [record])
    pilot.remember_inputs(cache, "dtd", "gaussian_blur", 2, [record])
    with pytest.raises(ValueError, match="differ across methods"):
        pilot.remember_inputs(cache, "dtd", "gaussian_blur", 2, [dict(record, tensor_sha256="other")])


def test_seal_verify_and_detect_tampering(tmp_path):
    commit = "1" * 40
    pilot.write_json(tmp_path / "run_manifest.json", {"commit": commit, "protocol_sha256": digest(PROTOCOL_PATH),
                                                      "inputs_sha256": digest(INPUTS_PATH)})
    pilot.write_json(tmp_path / "completion.json", {"passed": False, "exports": 2})
    artifact = tmp_path / "failure.log"
    artifact.write_text("failure!", encoding="utf-8")
    pilot.seal(tmp_path)
    with pytest.raises(ValueError, match="failed run"):
        pilot.verify(tmp_path, commit)
    assert pilot.verify(tmp_path, commit, allow_failed=True)["verified_files"] == 3
    with pytest.raises(ValueError, match="commit"):
        pilot.verify(tmp_path, "2" * 40, allow_failed=True)
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        pilot.verify(tmp_path, commit, allow_failed=True)
