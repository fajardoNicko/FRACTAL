"""Extract a coastline from a raster map image (e.g. a JPG).

The coastline is the *boundary* between sea and land. Two extraction methods are
provided, because real maps differ:

* ``method="otsu"`` — split the image into two regions by **brightness** (Otsu's
  method). Works for clean "filled land vs sea" images where one side is light
  and the other dark.

* ``method="sea_color"`` — detect the **blue sea by colour** (in HSV space).
  Needed for full topographic maps (e.g. NAMRIA sheets) where land contains
  white/green/text that overlaps the sea in brightness, so no single brightness
  cutoff separates them — but the sea is a distinct blue.

In both cases the pipeline is: build a sea/land mask -> (optionally) fill holes
and keep the largest region to drop labels/contour lines/specks -> take the
mask's morphological boundary. That boundary is the coastline, returned as a
binary image ready for the box-counting engine.

Only Pillow and scipy.ndimage are required (no OpenCV / scikit-image).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def load_grayscale(path: str) -> np.ndarray:
    """Load an image file as a 2-D float grayscale array in [0, 255]."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=float)


def load_hsv(path: str) -> np.ndarray:
    """Load an image as an (H, W, 3) HSV array; each channel in [0, 255].

    PIL maps the full hue circle to 0–255 (so blue ~= 170, cyan ~= 127,
    green ~= 85), saturation and value also 0–255.
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("HSV"), dtype=float)


def otsu_threshold(gray: np.ndarray) -> float:
    """Compute Otsu's threshold — the gray level best separating two classes."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 127.5

    levels = np.arange(256)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_mean = np.cumsum(hist * levels)
    mean_bg = np.divide(cum_mean, weight_bg, out=np.zeros(256), where=weight_bg > 0)
    total_mean = cum_mean[-1]
    mean_fg = np.divide(total_mean - cum_mean, weight_fg,
                        out=np.zeros(256), where=weight_fg > 0)
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(np.argmax(between_var))


def _crop_border(arr: np.ndarray, frac: float) -> np.ndarray:
    """Crop ``frac`` of each side off a 2-D or 3-D array."""
    if frac <= 0:
        return arr
    h, w = arr.shape[:2]
    dh, dw = int(h * frac), int(w * frac)
    return arr[dh:h - dh, dw:w - dw]


def _sea_mask_otsu(gray: np.ndarray) -> np.ndarray:
    """Binary mask from a brightness threshold (polarity-independent boundary)."""
    return gray > otsu_threshold(gray)


def _sea_mask_color(hsv: np.ndarray, hue_lo: float, hue_hi: float,
                    sat_min: float, val_min: float) -> np.ndarray:
    """Binary mask of blue-ish sea pixels in HSV space.

    Selects pixels whose hue falls in the cyan–blue band and that are colourful
    and bright enough — which rejects white/gray paper (low saturation) and
    green land (different hue).
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= hue_lo) & (h <= hue_hi) & (s >= sat_min) & (v >= val_min)


def extract_coastline(
    path: str,
    method: str = "otsu",
    threshold: float | str = "otsu",
    keep_largest: bool = True,
    fill_holes: bool = True,
    crop_border_frac: float = 0.0,
    hue_lo: float = 120.0,
    hue_hi: float = 185.0,
    sat_min: float = 25.0,
    val_min: float = 40.0,
    debug_path: str | None = None,
) -> np.ndarray:
    """Turn a map image into a binary coastline image.

    Args:
        path:             Image file (JPG/PNG/...).
        method:           ``"otsu"`` (brightness) or ``"sea_color"`` (blue sea).
        threshold:        For ``otsu``: ``"otsu"`` or a fixed gray level 0–255.
        keep_largest:     Keep only the largest connected region before taking
                          the boundary (drops text/legend/contour specks).
        fill_holes:       Fill interior holes (labels/contours inside the sea).
        crop_border_frac: Crop this fraction off each side first (map frames).
        hue_lo, hue_hi:   For ``sea_color``: hue band selecting the sea (PIL hue
                          scale 0–255; blue ~= 170, cyan ~= 127).
        sat_min, val_min: For ``sea_color``: minimum saturation/brightness.
        debug_path:       If set, save a colour overlay (coastline drawn in red
                          over the original) here for visual inspection.

    Returns:
        Boolean array; True on coastline (sea/land boundary) pixels.
    """
    if method == "sea_color":
        hsv = _crop_border(load_hsv(path), crop_border_frac)
        mask = _sea_mask_color(hsv, hue_lo, hue_hi, sat_min, val_min)
    elif method == "otsu":
        gray = _crop_border(load_grayscale(path), crop_border_frac)
        thresh = otsu_threshold(gray) if threshold == "otsu" else float(threshold)
        mask = gray > thresh
    else:
        raise ValueError(f"unknown method {method!r}; use 'otsu' or 'sea_color'")

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if keep_largest:
        mask = _keep_largest_region(mask)

    # Morphological boundary: mask pixels bordering a non-mask pixel. border_value=1
    # so a region touching the image edge does not create a boundary along the frame.
    eroded = ndimage.binary_erosion(mask, border_value=1)
    boundary = mask & ~eroded

    if debug_path is not None:
        _save_overlay(path, boundary, crop_border_frac, debug_path)
    return boundary


def _keep_largest_region(mask: np.ndarray) -> np.ndarray:
    """Return a mask with only its single largest connected component kept.

    Considers both the mask and its complement and keeps whichever connected
    component is largest overall — robust to whether sea or land is foreground.
    """
    best, best_size = mask, int(mask.sum())
    for candidate in (mask, ~mask):
        labels, n = ndimage.label(candidate)
        if n == 0:
            continue
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        region = labels == largest
        if region.sum() > best_size:
            best, best_size = region, int(region.sum())
    return best


def _save_overlay(path: str, boundary: np.ndarray, crop_border_frac: float,
                  out_path: str) -> None:
    """Save the original image with the detected coastline drawn in red."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB")).copy()
    rgb = _crop_border(rgb, crop_border_frac)
    # Dilate slightly so the thin coastline is visible in the preview.
    thick = ndimage.binary_dilation(boundary, iterations=2)
    rgb[thick] = [255, 0, 0]
    Image.fromarray(rgb).save(out_path)
