"""Coastline geometry cleaning.

Raw coastline vertices (from a shapefile or a synthetic generator) often carry
artefacts that distort box-counting: duplicate consecutive points, NaN/inf
coordinates, or degenerate zero-length runs. :func:`clean_coords` normalizes an
``(N, 2)`` polyline into a well-formed one before rasterization.
"""

from __future__ import annotations

import numpy as np


def clean_coords(coords: np.ndarray) -> np.ndarray:
    """Return a cleaned copy of an (N, 2) coastline polyline.

    Steps:
        1. Drop any vertex containing NaN or infinity.
        2. Drop consecutive duplicate vertices (zero-length steps), which add no
           geometric information but can bias fine-scale box counts.

    Args:
        coords: (N, 2) array of (x, y) vertices.

    Returns:
        A cleaned (M, 2) array with M <= N.

    Raises:
        ValueError: If the input is not (N, 2) or fewer than two vertices
            survive cleaning (too little to define a line).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")

    # 1. Keep only finite vertices.
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]

    # 2. Remove consecutive duplicates (keep the first vertex, then any vertex
    #    that differs from its predecessor).
    if len(coords) > 1:
        differs = np.any(np.diff(coords, axis=0) != 0, axis=1)
        keep = np.concatenate([[True], differs])
        coords = coords[keep]

    if len(coords) < 2:
        raise ValueError("fewer than two distinct vertices after cleaning")

    return coords
