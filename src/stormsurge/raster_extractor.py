"""Storm-surge height per segment (the study's dependent variable).

Real mode (TODO): for each segment, take the maximum recorded PAGASA surge
height within the segment's bounding box across typhoon events 2010–2024,
falling back to NDRRMC situation reports / literature where PAGASA data is
missing (outline §III-G).

Synthetic mode: surge heights are *generated* with a deliberate, noisy positive
dependence on fractal dimension —

    surge = surge_base + surge_slope * (D − 1) + noise   (clipped at 0)

— so the statistics stage has a real (but imperfect) correlation to detect.
This lets us verify the analysis end to end; it is NOT a physical surge model
and must be replaced with PAGASA data for any real finding.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.segment_generator import Segment


def get_surge_heights(
    segments: list[Segment],
    dimensions: np.ndarray,
    config,
) -> np.ndarray:
    """Return the maximum surge height (metres) for each segment.

    Args:
        segments:   The analysed segments, in order.
        dimensions: Fractal dimension D for each segment, same order.
        config:     A :class:`src.utils.config.Config`.

    Returns:
        1-D array of surge heights (metres), one per segment.
    """
    if config.mode == "real":
        # Surge heights come from the segment metadata CSV. For each segment use
        # an explicit surge_height_m if given, else derive one from its PAGASA
        # SSA level (ssa_level column). Validate that every segment has one.
        from src.stormsurge.vulnerability_scorer import ssa_to_height

        surge = np.array(
            [s.surge_height_m if not np.isnan(s.surge_height_m)
             else ssa_to_height(s.ssa_level) for s in segments],
            dtype=float,
        )
        missing = [s.id for s, v in zip(segments, surge) if np.isnan(v)]
        if missing:
            raise ValueError(
                "real mode: no surge value for segment(s) "
                f"{', '.join(missing)}. In data/raw/segments.csv, give each "
                "either a 'surge_height_m' (metres) or an 'ssa_level' "
                "(none/1/2/3/4 from HazardHunterPH)."
            )
        return surge

    syn = config.synthetic
    dimensions = np.asarray(dimensions, dtype=float)

    # Reproducible noise, independent of the segment-generation stream.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x5126]))
    noise = rng.normal(0.0, syn["surge_noise"], size=dimensions.size)

    surge = syn["surge_base"] + syn["surge_slope"] * (dimensions - 1.0) + noise
    return np.clip(surge, 0.0, None)
