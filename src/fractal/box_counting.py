"""Box-counting fractal dimension.

Implements the box-counting (Minkowski-Bouligand) estimate of fractal dimension
described in the research outline (§VII-B):

    A grid of boxes of side ``epsilon`` is laid over a binary image of the
    coastline. ``N(epsilon)`` is the number of boxes that contain at least one
    coastline pixel. As the box size shrinks, the count grows as a power law:

        N(epsilon) ~ epsilon^(-D)        <=>        N(epsilon) ~ (1/epsilon)^D

    Taking logs gives a straight line whose slope is the dimension:

        log N(epsilon) = D * log(1/epsilon) + C

    so D is recovered as the slope of a linear regression of ``log N`` against
    ``log(1/epsilon)``. For a smooth curve D -> 1; for a space-filling curve
    D -> 2. Natural coastlines fall in roughly 1.05 <= D <= 1.52.

The implementation depends only on NumPy so it can be validated against fractals
of known dimension before any geospatial data is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default box sizes (in pixels). The research outline §III-D lists
# [2, 4, 8, 16, 32, 64, 128, 256]; we drop the 2 px box from the default fit.
#
# Rationale (validated in tests/test_box_counting.py against curves of known
# dimension): at ~1 px line thickness, a 2 px box measures the line *thickness*
# rather than the fractal's self-similar structure, which biases the slope down
# toward the smooth-curve value D = 1. Restricting the regression to the
# scaling region (4..256 px) recovers the Koch (1.262) and Minkowski (1.500)
# dimensions to within 0.05. Powers of two also tile power-of-two images evenly,
# avoiding partial-box edge effects. The full list can still be passed explicitly.
DEFAULT_BOX_SIZES: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class BoxCountResult:
    """Outcome of a box-counting dimension estimate.

    Attributes:
        dimension:   Estimated fractal dimension D (slope of the log-log fit).
        r_squared:   Goodness-of-fit of the regression. The outline flags any
                     segment with ``r_squared < 0.97`` for re-examination.
        intercept:   Intercept C of the log-log line (``log N = D*log(1/eps)+C``).
        box_sizes:   The box sizes (epsilon, in pixels) actually used.
        counts:      ``N(epsilon)`` for each box size, in the same order.
    """

    dimension: float
    r_squared: float
    intercept: float
    box_sizes: tuple[int, ...]
    counts: tuple[int, ...]

    @property
    def flagged(self) -> bool:
        """True if the fit is too poor to trust (outline threshold R^2 < 0.97)."""
        return self.r_squared < 0.97


def count_boxes(image: np.ndarray, box_size: int) -> int:
    """Count boxes of side ``box_size`` that contain at least one set pixel.

    The image is partitioned into a grid of ``box_size`` x ``box_size`` blocks
    (zero-padded so the blocks tile evenly) and we count how many blocks contain
    any non-zero pixel. This is a fully vectorized reshape-and-reduce, so it is
    far faster than looping over individual boxes.

    Args:
        image:    2-D array; any non-zero pixel is treated as "coastline".
        box_size: Side length of each box, in pixels. Must be >= 1.

    Returns:
        The number of occupied boxes, ``N(epsilon)``.
    """
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1, got {box_size}")

    occupied = image != 0  # normalize to boolean; treats binary/uint8/float alike

    height, width = occupied.shape
    # Pad with empty pixels so both dimensions are exact multiples of box_size.
    pad_h = (-height) % box_size
    pad_w = (-width) % box_size
    if pad_h or pad_w:
        occupied = np.pad(occupied, ((0, pad_h), (0, pad_w)), mode="constant")

    padded_h, padded_w = occupied.shape
    # Reshape into (rows, box_size, cols, box_size) and ask whether any pixel in
    # each (box_size x box_size) block is set.
    blocks = occupied.reshape(
        padded_h // box_size, box_size, padded_w // box_size, box_size
    )
    occupied_blocks = blocks.any(axis=(1, 3))
    return int(occupied_blocks.sum())


def box_counting_dimension(
    image: np.ndarray,
    box_sizes: tuple[int, ...] | list[int] = DEFAULT_BOX_SIZES,
) -> BoxCountResult:
    """Estimate the box-counting fractal dimension of a binary image.

    Args:
        image:     2-D array where non-zero pixels represent the coastline.
        box_sizes: Box side lengths (epsilon) to sample, in pixels. At least two
                   distinct sizes are required to fit a line.

    Returns:
        A :class:`BoxCountResult` with the dimension estimate and fit quality.

    Raises:
        ValueError: If the image is not 2-D, fewer than two box sizes are given,
                    or the image contains no coastline pixels.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")

    sizes = tuple(int(s) for s in box_sizes)
    if len(set(sizes)) < 2:
        raise ValueError("at least two distinct box sizes are required")

    counts = tuple(count_boxes(image, s) for s in sizes)
    if all(c == 0 for c in counts):
        raise ValueError("image contains no coastline pixels (all counts are 0)")

    # Regress log N(eps) against log(1/eps). Per the power law N ~ (1/eps)^D,
    # the slope of this line IS the dimension D (no sign flip needed).
    log_inv_eps = np.log(1.0 / np.asarray(sizes, dtype=float))
    log_n = np.log(np.asarray(counts, dtype=float))

    slope, intercept = np.polyfit(log_inv_eps, log_n, 1)

    # Coefficient of determination (R^2) of the fit.
    predicted = slope * log_inv_eps + intercept
    ss_residual = float(np.sum((log_n - predicted) ** 2))
    ss_total = float(np.sum((log_n - np.mean(log_n)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 1.0

    return BoxCountResult(
        dimension=float(slope),
        r_squared=float(r_squared),
        intercept=float(intercept),
        box_sizes=sizes,
        counts=counts,
    )
