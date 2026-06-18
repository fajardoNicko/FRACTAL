"""Validation test for the JPG -> coastline extractor.

Builds a synthetic "filled land vs sea" image whose land/sea boundary is a Koch
curve (known dimension D = log4/log3 ~= 1.2619), saves it as a JPG, then runs the
extractor + box-counting and checks the recovered dimension. This proves the
raster-image path matches the validated engine.

Run with:
    pytest tests/test_image_coastline.py
    python  tests/test_image_coastline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fractal.box_counting import box_counting_dimension  # noqa: E402
from src.fractal.fractals import KOCH_DIMENSION, koch_curve, straight_line  # noqa: E402
from src.preprocessing.image_coastline import extract_coastline, otsu_threshold  # noqa: E402
from src.preprocessing.raster_converter import rasterize_polyline  # noqa: E402

DIMENSION_TOLERANCE = 0.06
RASTER_SIZE = 2048


def _make_landsea_jpg(coords: np.ndarray, path: str, size: int = RASTER_SIZE) -> None:
    """Render a filled land/sea JPG whose land/sea boundary follows ``coords``.

    The curve is rasterized into a connected barrier spanning the image; the
    region below it (touching the bottom edge) is filled as "land", the rest as
    "sea". The shared boundary is therefore exactly the input curve.
    """
    curve = rasterize_polyline(coords, size=size, padding=0.0)
    barrier = ndimage.binary_dilation(curve, iterations=1)
    labels, _ = ndimage.label(~barrier)
    bottom_components = set(labels[-1, :]) - {0}
    land = np.isin(labels, list(bottom_components))
    Image.fromarray(np.where(land, 205, 35).astype("uint8")).save(path, quality=92)


def _extract_dimension(coords: np.ndarray) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "coast.jpg")
        _make_landsea_jpg(coords, path)
        coastline = extract_coastline(path)
    return box_counting_dimension(coastline).dimension


def test_otsu_splits_bimodal_image():
    # Two clear gray levels (35 and 205) -> threshold should land between them.
    gray = np.array([[35, 205], [35, 205]], dtype=float)
    t = otsu_threshold(gray)
    assert 35 < t < 205


def test_extracted_koch_dimension():
    d = _extract_dimension(koch_curve(order=7))
    assert abs(d - KOCH_DIMENSION) < DIMENSION_TOLERANCE


def test_extracted_straight_coast_is_one():
    d = _extract_dimension(straight_line())
    assert abs(d - 1.0) < DIMENSION_TOLERANCE


def _main() -> int:
    cases = [
        ("Straight coast (JPG)", straight_line(), 1.0),
        ("Koch coast (JPG, order 7)", koch_curve(order=7), KOCH_DIMENSION),
    ]
    print(f"{'Case':<30}{'Theoretical':>12}{'Extracted':>12}{'Error':>9}")
    print("-" * 63)
    ok = True
    for name, coords, theoretical in cases:
        d = _extract_dimension(coords)
        err = abs(d - theoretical)
        ok &= err < DIMENSION_TOLERANCE
        print(f"{name:<30}{theoretical:>12.4f}{d:>12.4f}{err:>9.4f}"
              f"  {'OK' if err < DIMENSION_TOLERANCE else 'FAIL'}")
    print("-" * 63)
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
