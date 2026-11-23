"""Fixed RGB enhancement operators and a CPU-only precheck against V1.0 pixels."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN_NAME = "formal-v1-confirmed-01"
FORMAL = ROOT / "outputs" / "course_robustness_v1" / RUN_NAME
CROP_SOURCE = ROOT.parent / "CLIP_LaTeX_论文修订V4"
DATASETS = {"dtd": "DTD", "oxford_flowers": "Flowers102"}
RGB_SHAPE = (224, 224, 3)
LEVELS = (2, 3)
GAUSSIAN_NOISE = "gaussian_noise"
GAUSSIAN_BLUR = "gaussian_blur"
GAMMA_DARKENING = "gamma_darkening"
METHODS = {
    GAUSSIAN_NOISE: ("Gaussian noise", "Bilateral filter"),
    GAUSSIAN_BLUR: ("Gaussian blur", "Unsharp masking"),
    GAMMA_DARKENING: ("Gamma darkening", "CLAHE (L channel)"),
}


def validate_rgb(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("Expected an RGB NumPy uint8 array")
    if image.shape != RGB_SHAPE:
        raise ValueError(f"Expected RGB shape {RGB_SHAPE}, got {image.shape}")


def enhance(corrupted_rgb, corruption):
    """Return a new RGB uint8 crop using only the supplied corrupted pixels."""
    validate_rgb(corrupted_rgb)
    if corruption == GAUSSIAN_NOISE:
        return cv2.bilateralFilter(corrupted_rgb, d=5, sigmaColor=50, sigmaSpace=5,
                                   borderType=cv2.BORDER_DEFAULT)
    if corruption == GAUSSIAN_BLUR:
        filter_settings = ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)
        sharpened = Image.fromarray(corrupted_rgb).filter(filter_settings)
        return np.array(sharpened)
    if corruption == GAMMA_DARKENING:
        lightness, a, b = cv2.split(cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2LAB))
        equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(cv2.merge((equalized, a, b)), cv2.COLOR_LAB2RGB)
    raise ValueError(f"No enhancement registered for {corruption!r}")


def quality_metrics(clean, compared):
    """Return full RGB PSNR (dB) and the mean of the three channel SSIMs."""
    validate_rgb(clean)
    validate_rgb(compared)
    reference = clean.astype(np.float64)
    image = compared.astype(np.float64)
    mse = np.mean((reference - image) ** 2)
    psnr = float(10 * np.log10(255**2 / mse)) if mse > 0 else float("inf")

    def local_mean(values):
        # 11x11 spatial window (sigma 1.5), with no interpolation across RGB channels.
        return gaussian_filter(values, sigma=(1.5, 1.5, 0), truncate=5 / 1.5)[5:-5, 5:-5]

    mean_ref = local_mean(reference)
    mean_image = local_mean(image)
    var_ref = local_mean(reference**2) - mean_ref**2
    var_image = local_mean(image**2) - mean_image**2
    covariance = local_mean(reference * image) - mean_ref * mean_image
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = ((2 * mean_ref * mean_image + c1) * (2 * covariance + c2)
                / ((mean_ref**2 + mean_image**2 + c1) * (var_ref + var_image + c2)))
    return psnr, float(ssim_map.mean())


@dataclass(frozen=True)
class Crop:
    dataset: str
    number: int
    image_id: str
    pixels: np.ndarray

    @property
    def key(self):
        return f"{self.dataset}_sample{self.number}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixel_hash(rgb):
    return sha256(rgb.tobytes()).hexdigest()


def metadata_path(dataset, corruption, level):
    return FORMAL / "predictions" / dataset / "clip" / "seed1" / "base" / f"{corruption}_L{level}" / "metadata.json"


def formal_rgb_hashes():
    by_condition = {}
    conditions = [("clean", 0), *product(METHODS, LEVELS)]
    for dataset, (corruption, level) in product(DATASETS, conditions):
        records = read_json(metadata_path(dataset, corruption, level))["sample_records"]
        by_condition[dataset, corruption, level] = {row["image_id"]: row["rgb_uint8_sha256"] for row in records}
    return by_condition


def load_crops(expected):
    provenance = read_json(CROP_SOURCE / "records" / "IMAGE_PROVENANCE.json")
    if not provenance["passed"]:
        raise ValueError("Archived crop provenance did not pass")
    sources = {(row["dataset"], row["sample_number"]): row for row in provenance["images"]
               if row["condition"] == "clean_L0" and row["candidate_group"] == "base"}
    for dataset, number in product(DATASETS, (1, 2)):
        row = sources[dataset, number]
        image_id = row["image_id"]
        path = CROP_SOURCE / "assets" / f"{dataset}_sample{number}_clean_L0.png"
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"Expected a lossless RGB PNG: {path}")
            clean = np.array(image)
        validate_rgb(clean)
        digest = pixel_hash(clean)
        if digest != row["rgb_uint8_sha256"] or digest != expected[dataset, "clean", 0][image_id]:
            raise ValueError(f"Clean crop pixel SHA-256 differs from formal metadata: {image_id}")
        yield Crop(dataset, number, image_id, clean)


def render_grid(crop, corruption, level, degraded, enhanced, before, after):
    pad, gap, side, top = 18, 12, RGB_SHAPE[0], 96
    canvas = Image.new("RGB", (pad * 2 + side * 3 + gap * 2, top + side + 55), "#f4f7f9")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.load_default(size=20)
    regular = ImageFont.load_default(size=15)
    name, method = METHODS[corruption]
    draw.text((pad, 12), f"{DATASETS[crop.dataset]} / {Path(crop.image_id).name}", font=heading, fill="#142831")
    draw.text((pad, 40), f"{name} L{level} / {method}", font=regular, fill="#435760")
    panels = [("Clean", crop.pixels, None), ("Corrupted", degraded, before), ("Enhanced", enhanced, after)]
    for index, (label, pixels, scores) in enumerate(panels):
        x = pad + index * (side + gap)
        draw.text((x, top - 25), label, font=regular, fill="#142831")
        canvas.paste(Image.fromarray(pixels), (x, top))
        note = "Reference crop" if scores is None else f"{scores[0]:.2f} dB PSNR   |   {scores[1]:.4f} SSIM"
        draw.text((x, top + side + 12), note, font=regular, fill="#435760")
    return canvas


def run_precheck(output_dir=OUT):
    # The precheck consumes the existing corruption module. The enhancement operator needs no course data.
    from course_eval.corruptions import corrupt

    hashes = formal_rgb_hashes()
    rows = []
    for crop in load_crops(hashes):
        sample_dir = output_dir / "preview" / crop.key
        sample_dir.mkdir(parents=True, exist_ok=True)
        clean_file = sample_dir / "clean.png"
        Image.fromarray(crop.pixels).save(clean_file)
        clean_hash = pixel_hash(crop.pixels)
        for corruption, level in product(METHODS, LEVELS):
            degraded = corrupt(crop.pixels, corruption, level, dataset=crop.dataset, image_id=crop.image_id)
            degraded_hash = pixel_hash(degraded)
            if degraded_hash != hashes[crop.dataset, corruption, level][crop.image_id]:
                raise ValueError(f"Corrupted pixel SHA-256 differs from V1.0: {crop.image_id}, {corruption} L{level}")

            enhanced = enhance(degraded, corruption)
            before = quality_metrics(crop.pixels, degraded)
            after = quality_metrics(crop.pixels, enhanced)
            condition_dir = sample_dir / f"{corruption}_L{level}"
            condition_dir.mkdir(parents=True, exist_ok=True)
            corrupted_file = condition_dir / "corrupted.png"
            enhanced_file = condition_dir / "enhanced.png"
            grid_file = condition_dir / "grid.png"
            Image.fromarray(degraded).save(corrupted_file)
            Image.fromarray(enhanced).save(enhanced_file)
            render_grid(crop, corruption, level, degraded, enhanced, before, after).save(grid_file)
            rows.append(dict(dataset=crop.dataset, sample=crop.key, image_id=crop.image_id,
                             corruption=corruption, level=level, enhancement=METHODS[corruption][1],
                             corrupted_psnr_db=before[0], enhanced_psnr_db=after[0], delta_psnr_db=after[0] - before[0],
                             corrupted_ssim=before[1], enhanced_ssim=after[1], delta_ssim=after[1] - before[1],
                             clean_rgb_sha256=clean_hash, corrupted_rgb_sha256=degraded_hash,
                             enhanced_rgb_sha256=pixel_hash(enhanced),
                             clean_png=clean_file.relative_to(output_dir).as_posix(),
                             corrupted_png=corrupted_file.relative_to(output_dir).as_posix(),
                             enhanced_png=enhanced_file.relative_to(output_dir).as_posix(),
                             grid_png=grid_file.relative_to(output_dir).as_posix()))

    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    row_count = run_precheck()
    print(f"Verified {row_count} V1.0 corrupted inputs and saved all images, comparison grids, and metrics.")
