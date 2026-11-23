"""Fixed RGB pixel operators for the enhancement V1 protocol."""

from functools import partial
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


PROTOCOL_FILE = Path(__file__).with_name("enhancement_protocol_v1.json")
PROTOCOL = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))


def configured(operation):
    def apply(rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise TypeError("Expected a uint8 NumPy array")
        if rgb.shape != tuple(PROTOCOL["image_shape"]):
            raise ValueError("Expected a 224x224 RGB array")
        return operation(rgb)
    return apply


def unsharp(rgb, *, radius, percent, threshold):
    image_filter = ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    return np.array(Image.fromarray(rgb).filter(image_filter))


def clahe(rgb, *, clip_limit, tiles):
    lightness, a, b = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tiles).apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2RGB)


def unsharp_profile(profile):
    """Bind one of the two CPU review profiles to a pixels-only operation."""
    return configured(partial(unsharp, **PROTOCOL["unsharp_profiles"][profile]))


def enhancement_parameters(corruption):
    spec = PROTOCOL["algorithms"][corruption].copy()
    if spec["name"] == "unsharp_mask":
        spec.update(PROTOCOL["unsharp_profiles"][spec["profile"]])
    return spec


def enhancer_for(corruption):
    """Resolve the fixed treatment before reading any image, then accept pixels only."""
    spec = enhancement_parameters(corruption)
    if spec["name"] == "bilateral":
        return configured(partial(cv2.bilateralFilter, d=spec["d"], sigmaColor=spec["sigmaColor"],
                                  sigmaSpace=spec["sigmaSpace"], borderType=getattr(cv2, spec["borderType"])))
    if spec["name"] == "unsharp_mask":
        return unsharp_profile(spec["profile"])
    if spec["name"] == "clahe":
        return configured(partial(clahe, clip_limit=spec["clipLimit"], tiles=tuple(spec["tileGridSize"])))
    raise ValueError(f"Unrecognized enhancement {spec['name']}")
