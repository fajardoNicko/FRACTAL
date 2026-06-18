"""Vector-to-raster conversion.

The box-counting engine operates on a *binary image*: a 2-D grid where a pixel
is "on" if the coastline passes through it. This module turns coordinate data
into such a grid.

:func:`rasterize_polyline` is a dependency-light (NumPy-only) rasterizer used
both for validating the engine against synthetic fractals and as a fallback for
real data. When the full geospatial stack is installed, prefer rasterizing
shapely geometries via rasterio for projection-aware results (left as a TODO
until real NAMRIA shapefiles are wired in).
"""

from __future__ import annotations

import numpy as np

# Outline §III-D specifies a 4096 x 4096 rasterization resolution.
DEFAULT_RASTER_SIZE = 4096


def rasterize_polyline(
    coords: np.ndarray,
    size: int = DEFAULT_RASTER_SIZE,
    padding: float = 0.05,
) -> np.ndarray:
    """Rasterize an (N, 2) polyline into a square binary image.

    The polyline is scaled to fit the image while preserving aspect ratio, with
    a small uniform margin so no detail touches the border. Consecutive vertices
    are connected by densely sampling the segment, which guarantees an unbroken
    pixel chain (important — gaps would inflate the box count at small scales).

    Args:
        coords:  (N, 2) array of (x, y) vertices.
        size:    Output image side length, in pixels.
        padding: Fractional margin on each side, in [0, 0.5).

    Returns:
        A ``(size, size)`` boolean array; True where the polyline passes.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")
    if not 0.0 <= padding < 0.5:
        raise ValueError(f"padding must be in [0, 0.5), got {padding}")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = (maxs - mins).max()
    if span == 0:
        raise ValueError("coordinates have zero extent; nothing to rasterize")

    # Scale uniformly into the padded drawing area, then offset by the margin.
    usable = (1.0 - 2.0 * padding) * (size - 1)
    scaled = (coords - mins) / span * usable + padding * (size - 1)

    image = np.zeros((size, size), dtype=bool)
    for start, end in zip(scaled[:-1], scaled[1:]):
        # Oversample by 2x the pixel length so adjacent samples never skip a
        # pixel, regardless of segment angle.
        pixel_length = np.hypot(*(end - start))
        num_samples = max(2, int(np.ceil(pixel_length * 2)) + 1)
        xs = np.linspace(start[0], end[0], num_samples)
        ys = np.linspace(start[1], end[1], num_samples)
        # Row index = y, column index = x.
        image[ys.astype(int), xs.astype(int)] = True

    return image
