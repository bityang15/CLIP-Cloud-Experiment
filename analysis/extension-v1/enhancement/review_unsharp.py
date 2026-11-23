"""Compare the two fixed unsharp profiles using changes to blurred pixels."""

import csv
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from enhancement import OUT, ROOT, formal_rgb_hashes, load_crops, pixel_hash
from enhancement_v1 import PROTOCOL, unsharp_profile


def pixel_changes(before, after):
    delta = np.abs(after.astype(np.int16) - before.astype(np.int16))
    newly_saturated = ((after == 0) | (after == 255)) & ((before != 0) & (before != 255))
    return dict(pixels_changed_pct=100 * np.any(delta != 0, axis=2).mean(),
                mean_abs_channel_delta=float(delta.mean()), p95_abs_channel_delta=float(np.percentile(delta, 95)),
                max_abs_channel_delta=int(delta.max()), newly_saturated_channels_pct=100 * newly_saturated.mean())


def comparison_grid(image_id, level, panels):
    image = Image.new("RGB", (720, 302), "#f4f7f9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    draw.text((12, 12), f"{image_id} / Gaussian blur L{level}", font=font, fill="#172630")
    for index, (name, pixels) in enumerate(panels):
        x = 12 + index * 236
        draw.text((x, 40), name, font=font, fill="#172630")
        image.paste(Image.fromarray(pixels), (x, 63))
    return image


def main():
    from course_eval.corruptions import corrupt

    destination = OUT / "unsharp_review_preview"
    destination.mkdir(exist_ok=True)
    rows = []
    expected = formal_rgb_hashes()
    profiles = {name: unsharp_profile(name) for name in PROTOCOL["unsharp_profiles"]}
    for crop in load_crops(expected):
        for level in PROTOCOL["levels"]:
            blurred = corrupt(crop.pixels, "gaussian_blur", level, dataset=crop.dataset, image_id=crop.image_id)
            blurred_hash = pixel_hash(blurred)
            if blurred_hash != expected[crop.dataset, "gaussian_blur", level][crop.image_id]:
                raise ValueError(f"Formal blur pixels differ: {crop.image_id} L{level}")
            panels = [("Corrupted", blurred)]
            for name, operation in profiles.items():
                enhanced = operation(blurred)
                np.testing.assert_array_equal(enhanced, operation(blurred))
                rows.append(dict(dataset=crop.dataset, image_id=crop.image_id, level=level, profile=name,
                                 **PROTOCOL["unsharp_profiles"][name], **pixel_changes(blurred, enhanced),
                                 corrupted_rgb_sha256=blurred_hash, enhanced_rgb_sha256=pixel_hash(enhanced)))
                panels.append((name.capitalize(), enhanced))
            comparison_grid(crop.image_id, level, panels).save(destination / f"{crop.key}_L{level}.png")
    with (OUT / "unsharp_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} reviews and {len(rows)//2} comparison PNGs")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
