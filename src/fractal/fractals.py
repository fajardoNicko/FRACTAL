"""Reference fractal curves with known theoretical dimensions.

These generators exist to *validate* the box-counting engine: each curve has a
mathematically exact fractal dimension, so we can confirm the estimator recovers
the right value before trusting it on real coastline data.

Every generator returns an ``(N, 2)`` array of (x, y) vertices describing a
polyline, suitable for :func:`src.preprocessing.raster_converter.rasterize_polyline`.
"""

from __future__ import annotations

import math

import numpy as np

# Exact theoretical box-counting dimensions, for reference and test assertions.
KOCH_DIMENSION = math.log(4) / math.log(3)        # ~= 1.2619
MINKOWSKI_DIMENSION = math.log(8) / math.log(4)   # ~= 1.5000 (quadratic Koch)
LINE_DIMENSION = 1.0


def straight_line(num_points: int = 2) -> np.ndarray:
    """A straight segment from (0, 0) to (1, 0). Theoretical D = 1.0."""
    xs = np.linspace(0.0, 1.0, num_points)
    ys = np.zeros_like(xs)
    return np.column_stack([xs, ys])


def koch_curve(order: int = 5) -> np.ndarray:
    """Koch curve of the given recursion order. Theoretical D = log4/log3.

    Each iteration replaces every segment with four segments (the middle third
    is pushed out into an equilateral "bump"), multiplying the vertex count by 4
    while shrinking the ruler by 3 — the source of the log4/log3 dimension.

    Args:
        order: Number of recursive subdivisions (>= 0). Order 5 gives ~1025
               vertices, enough detail for a stable estimate.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    for _ in range(order):
        new_points = []
        for start, end in zip(points[:-1], points[1:]):
            delta = end - start
            # The four new vertices along this segment.
            a = start
            b = start + delta / 3.0                       # 1/3 point
            d = start + 2.0 * delta / 3.0                 # 2/3 point
            # Apex of the equilateral bump: midpoint of b..d pushed out along
            # the segment normal by sqrt(3)/6 of the segment length.
            normal = np.array([-delta[1], delta[0]])
            c = start + delta / 2.0 + normal * (math.sqrt(3) / 6.0)
            new_points.extend([a, b, c, d])
        new_points.append(points[-1])
        points = np.array(new_points)
    return points


def minkowski_sausage(order: int = 4) -> np.ndarray:
    """Minkowski / quadratic Koch curve. Theoretical D = log8/log4 = 1.5.

    Each segment is replaced by eight segments of one-quarter length arranged in
    a square-wave bump, giving the exact dimension 1.5 — a useful second check
    at a higher dimension than the Koch curve.

    Args:
        order: Number of recursive subdivisions (>= 0).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    for _ in range(order):
        new_points = []
        for start, end in zip(points[:-1], points[1:]):
            delta = end - start
            unit = delta / 4.0                            # quarter-segment vector
            normal = np.array([-delta[1], delta[0]]) / 4.0  # perpendicular quarter
            # Eight-segment square-wave generator (9 vertices, last shared with
            # the next segment's start).
            offsets = [
                np.zeros(2),
                unit,
                unit + normal,
                2 * unit + normal,
                2 * unit,
                2 * unit - normal,
                3 * unit - normal,
                3 * unit,
            ]
            for off in offsets:
                new_points.append(start + off)
        new_points.append(points[-1])
        points = np.array(new_points)
    return points
