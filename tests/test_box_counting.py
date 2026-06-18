"""Validation tests for the box-counting engine.

The whole study rests on the dimension estimator being correct, so we test it
against curves whose fractal dimension is known *exactly*:

    * straight line     D = 1.0
    * Koch curve        D = log4/log3 ~= 1.2619
    * Minkowski sausage D = log8/log4  = 1.5

If the estimator recovers these within a small tolerance, we can trust the
values it produces for real coastline segments.

Run with either:
    pytest tests/test_box_counting.py
    python  tests/test_box_counting.py     (no pytest needed — prints a report)
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make ``src`` importable whether run via pytest or directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fractal.box_counting import box_counting_dimension  # noqa: E402
from src.fractal.fractals import (  # noqa: E402
    KOCH_DIMENSION,
    MINKOWSKI_DIMENSION,
    koch_curve,
    minkowski_sausage,
    straight_line,
)
from src.preprocessing.raster_converter import rasterize_polyline  # noqa: E402

# Box-counting on a finite raster carries a small, well-documented bias, so we
# allow a modest tolerance. The fractal orders are chosen high enough that
# self-similar detail reaches down to the smallest box in the scaling region.
DIMENSION_TOLERANCE = 0.06
MIN_R_SQUARED = 0.99
RASTER_SIZE = 4096  # matches the outline §III-D rasterization resolution


def _estimate(coords: np.ndarray):
    image = rasterize_polyline(coords, size=RASTER_SIZE)
    return box_counting_dimension(image)  # uses the validated 4..256 px default


def test_straight_line_dimension_is_one():
    result = _estimate(straight_line())
    assert abs(result.dimension - 1.0) < DIMENSION_TOLERANCE
    assert result.r_squared > MIN_R_SQUARED


def test_koch_curve_dimension():
    result = _estimate(koch_curve(order=7))
    assert abs(result.dimension - KOCH_DIMENSION) < DIMENSION_TOLERANCE
    assert result.r_squared > MIN_R_SQUARED


def test_minkowski_dimension():
    result = _estimate(minkowski_sausage(order=5))
    assert abs(result.dimension - MINKOWSKI_DIMENSION) < DIMENSION_TOLERANCE
    assert result.r_squared > MIN_R_SQUARED


def _main() -> int:
    """Pretty-print a validation report when run as a plain script."""
    cases = [
        ("Straight line", straight_line(), 1.0),
        ("Koch curve (order 7)", koch_curve(order=7), KOCH_DIMENSION),
        ("Minkowski sausage (order 5)", minkowski_sausage(order=5), MINKOWSKI_DIMENSION),
    ]
    print(f"{'Curve':<30}{'Theoretical':>12}{'Estimated':>12}{'Error':>9}{'R^2':>8}")
    print("-" * 71)
    all_ok = True
    for name, coords, theoretical in cases:
        r = _estimate(coords)
        error = abs(r.dimension - theoretical)
        ok = error < DIMENSION_TOLERANCE and r.r_squared > MIN_R_SQUARED
        all_ok &= ok
        flag = "OK" if ok else "FAIL"
        print(
            f"{name:<30}{theoretical:>12.4f}{r.dimension:>12.4f}"
            f"{error:>9.4f}{r.r_squared:>8.4f}  {flag}"
        )
    print("-" * 71)
    print("ALL PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
