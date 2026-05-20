"""
transforms.py — Cross-domain degradation transforms for robustness analysis.

Design contract
───────────────
Every transform is a pure function:

    apply_<name>(image: PIL.Image, intensity: float) -> PIL.Image

- Input/output are always RGB PIL Images (mode "RGB").
- intensity=0 (or the natural baseline value) must be an identity transform —
  i.e., the image is returned unchanged.  This anchors the x-axis of the
  robustness curves at "no degradation".
- The transform must be deterministic given the same intensity value.

Why PIL and not numpy/torch?
  PIL is the interchange format used by both the dataset loaders and the
  Gradio demo. Keeping everything in PIL avoids double conversions.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gaussian Blur
#    intensity = sigma (std-dev of the Gaussian kernel in pixels)
#    sigma=0 → identity
# ─────────────────────────────────────────────────────────────────────────────

def apply_blur(image: Image.Image, intensity: float) -> Image.Image:
    """
    Gaussian blur with radius = intensity (sigma).

    sigma=0   : no blurring  (identity)
    sigma=1.0 : mild sensor defocus
    sigma=3.5 : heavy motion blur / low-resolution upsampled face
    sigma=5.0 : nearly unrecognisable

    PIL's ImageFilter.GaussianBlur radius parameter IS sigma, not kernel size.
    """
    if intensity <= 0.0:
        return image.copy()
    return image.filter(ImageFilter.GaussianBlur(radius=intensity))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Low-light (gamma correction)
#    intensity = gamma exponent
#    gamma=1.0 → identity
#    gamma>1   → image gets darker  (gamma < 1 would brighten)
# ─────────────────────────────────────────────────────────────────────────────

def apply_low_light(image: Image.Image, intensity: float) -> Image.Image:
    """
    Gamma correction: pixel_out = pixel_in ^ gamma

    gamma=1.0 : no change (identity)
    gamma=2.0 : moderate darkening — typical indoor low-light
    gamma=4.0 : severe darkening — near night-time conditions
    gamma=5.0 : very dark; face details barely visible

    We use a LUT (lookup table) for speed — computing x^gamma for all 256
    values once is far faster than applying per-pixel operations in Python.
    """
    if abs(intensity - 1.0) < 1e-6:
        return image.copy()

    # Build LUT: for each input level [0..255], compute output level
    lut = np.array(
        [int((i / 255.0) ** intensity * 255 + 0.5) for i in range(256)],
        dtype=np.uint8,
    )
    # PIL point() applies the LUT channel-wise; multiply by 3 for RGB
    return image.point(list(lut) * 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sketch-style transform
#    Models the cross-domain heterogeneous gap (photo ↔ forensic sketch)
#    intensity in [0, 1]: 0 = original photo, 1 = full sketch rendering
# ─────────────────────────────────────────────────────────────────────────────

def apply_sketch(image: Image.Image, intensity: float) -> Image.Image:
    """
    Approximate sketch rendering via edge-enhancement + desaturation blend.

    Algorithm:
      1. Convert to grayscale (removes colour cue — sketches are achromatic)
      2. Detect edges via Laplacian sharpening (amplifies fine structure)
      3. Invert edges to get pencil-stroke look on white background
      4. Blend original ↔ sketch by intensity:
           result = (1 - intensity) * original + intensity * sketch

    intensity=0.0 : original colour photo
    intensity=0.5 : partial sketch (texture+colour both visible)
    intensity=1.0 : full grayscale sketch rendering

    This is NOT a GAN-based style transfer — it's a deterministic signal
    processing approximation that is fast, reproducible, and captures the
    key challenge: the system must match a high-frequency edge map against
    a full-colour photo.
    """
    if intensity <= 0.0:
        return image.copy()

    img_np = np.array(image, dtype=np.float32)  # H×W×3, values in [0,255]

    # Step 1: Grayscale
    gray = np.dot(img_np, [0.299, 0.587, 0.114])  # H×W

    # Step 2: Laplacian edge detection (accentuates boundaries)
    gray_pil = Image.fromarray(gray.astype(np.uint8))
    edges_pil = gray_pil.filter(ImageFilter.SMOOTH_MORE)
    # Invert-divide trick: sketch = gray / (blurred_gray + eps)
    blur_np = np.array(edges_pil, dtype=np.float32)
    sketch_gray = np.clip(gray / (blur_np + 10.0) * 200.0, 0, 255)

    # Step 3: Expand sketch back to 3 channels
    sketch_rgb = np.stack([sketch_gray] * 3, axis=-1)  # H×W×3

    # Step 4: Alpha-blend
    blended = (1.0 - intensity) * img_np + intensity * sketch_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps config names to functions
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM_REGISTRY: dict[str, Callable[[Image.Image, float], Image.Image]] = {
    "blur":       apply_blur,
    "low_light":  apply_low_light,
    "sketch":     apply_sketch,
}


def get_transform(name: str) -> Callable[[Image.Image, float], Image.Image]:
    """Look up a transform by its config name."""
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown degradation '{name}'. "
            f"Available: {list(TRANSFORM_REGISTRY.keys())}"
        )
    return TRANSFORM_REGISTRY[name]


def apply_degradation(
    image: Image.Image,
    degradation_type: str,
    intensity: float,
) -> Image.Image:
    """
    Convenience wrapper: apply a named degradation at a given intensity.

    Args:
        image:            RGB PIL Image
        degradation_type: one of "blur", "low_light", "sketch"
        intensity:        scalar intensity (type-specific units)

    Returns:
        Degraded RGB PIL Image
    """
    fn = get_transform(degradation_type)
    result = fn(image, intensity)
    # Guarantee output is RGB (some PIL ops return L or RGBA)
    if result.mode != "RGB":
        result = result.convert("RGB")
    return result
