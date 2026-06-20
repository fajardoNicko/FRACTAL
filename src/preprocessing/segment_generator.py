"""Coastline segment generation.

In ``synthetic`` mode (the default until real NAMRIA shapefiles are supplied)
this module fabricates coastline segments whose geometric roughness — and hence
fractal dimension — is controlled, so the whole pipeline can be exercised and
validated end to end.

Roughness is produced by **midpoint displacement** (a 1-D fractional-Brownian
profile). The Hurst exponent ``H`` tunes it: a fractional-Brownian trace has
box-counting dimension ``D ≈ 2 − H``, so low ``H`` yields a rugged, high-D
coastline and high ``H`` a smooth, low-D one. Each synthetic coastline is given
a plausible Philippine location and centroid coordinate so downstream maps and
tables look realistic.

For ``real`` mode, :func:`generate_segments` is where NAMRIA shapefile loading
and 50 km clipping (outline §III-D) would plug in — left as a TODO.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Plausible Philippine coastal locations (name, province, region, lat, lon).
# Coordinates are approximate coastal centroids, used so synthetic maps render
# over the real archipelago. The first entries echo the outline's high-surge
# examples (Eastern Samar, Leyte Gulf, Albay).
_PH_LOCATIONS: list[tuple[str, str, str, float, float]] = [
    ("Eastern Samar coast", "Eastern Samar", "Region VIII", 11.50, 125.60),
    ("Leyte Gulf shore", "Leyte", "Region VIII", 11.00, 125.00),
    ("Albay Pacific coast", "Albay", "Region V", 13.20, 123.80),
    ("Tacloban Bay", "Leyte", "Region VIII", 11.24, 125.00),
    ("Sorsogon coast", "Sorsogon", "Region V", 12.97, 124.00),
    ("Catanduanes east", "Catanduanes", "Region V", 13.70, 124.30),
    ("Northern Samar coast", "Northern Samar", "Region VIII", 12.50, 124.80),
    ("Aurora Pacific coast", "Aurora", "Region III", 15.80, 121.60),
    ("Cagayan coast", "Cagayan", "Region II", 18.30, 122.00),
    ("Isabela coast", "Isabela", "Region II", 17.00, 122.20),
    ("Surigao del Norte coast", "Surigao del Norte", "Region XIII", 9.80, 125.50),
    ("Surigao del Sur coast", "Surigao del Sur", "Region XIII", 8.80, 126.20),
    ("Davao Oriental coast", "Davao Oriental", "Region XI", 7.30, 126.50),
    ("Camarines Sur coast", "Camarines Sur", "Region V", 13.60, 123.30),
    ("Quezon Pacific coast", "Quezon", "Region IV-A", 14.00, 122.30),
    ("Batanes shore", "Batanes", "Region II", 20.40, 121.97),
    ("Ilocos Norte coast", "Ilocos Norte", "Region I", 18.20, 120.60),
    ("La Union coast", "La Union", "Region I", 16.60, 120.30),
    ("Zambales coast", "Zambales", "Region III", 15.30, 119.95),
    ("Palawan west coast", "Palawan", "Region IV-B", 9.80, 118.70),
    ("Occidental Mindoro coast", "Occidental Mindoro", "Region IV-B", 12.80, 120.90),
    ("Iloilo coast", "Iloilo", "Region VI", 10.70, 122.50),
    ("Antique coast", "Antique", "Region VI", 11.00, 121.90),
    ("Cebu east coast", "Cebu", "Region VII", 10.30, 124.00),
    ("Bohol coast", "Bohol", "Region VII", 9.70, 124.30),
    ("Zamboanga Peninsula", "Zamboanga del Sur", "Region IX", 7.80, 123.00),
    ("Tawi-Tawi (Sulu Sea)", "Tawi-Tawi", "BARMM", 5.10, 119.90),
    ("Manila Bay interior", "Bataan", "Region III", 14.60, 120.50),
    ("Lingayen Gulf", "Pangasinan", "Region I", 16.20, 120.20),
    ("Sarangani Bay", "Sarangani", "Region XII", 6.00, 125.20),
]


@dataclass
class Segment:
    """One coastline segment and its metadata.

    A segment carries its coastline as *either* vector vertices (``coords``,
    synthetic mode) *or* a path to a raster map image (``image_path``, real
    mode). Downstream code branches on which is set.
    """

    id: str
    location: str
    province: str = ""
    region: str = ""
    lat: float = float("nan")
    lon: float = float("nan")
    coords: np.ndarray | None = None     # (N, 2) polyline vertices (synthetic)
    image_path: str | None = None        # coastline map image (real mode)
    hurst: float = float("nan")          # roughness control (synthetic only)
    surge_height_m: float = float("nan")  # known surge from metadata (real mode)
    ssa_level: str = ""                  # PAGASA SSA level fallback (real mode)


def _midpoint_displacement(depth: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a 1-D fractional-Brownian profile via midpoint displacement.

    Starts from a flat segment and repeatedly fills in midpoints with the
    average of their neighbours plus random displacement. The displacement
    magnitude shrinks by ``0.5**hurst`` each level, so a smaller Hurst exponent
    leaves more fine-scale roughness (higher fractal dimension).

    Args:
        depth: Recursion depth; produces ``2**depth + 1`` samples.
        hurst: Hurst exponent in (0, 1). Lower => rougher => higher dimension.
        rng:   Seeded NumPy generator for reproducibility.

    Returns:
        1-D array of ``2**depth + 1`` height values.
    """
    n = 2**depth + 1
    y = np.zeros(n)
    step = n - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        mids = np.arange(half, n - 1, step)        # midpoints to fill this level
        left, right = mids - half, mids + half
        y[mids] = 0.5 * (y[left] + y[right]) + rng.normal(0.0, scale, size=mids.size)
        scale *= 0.5**hurst                          # diminish displacement per level
        step = half
    return y


def _synthetic_coastline(depth: int, amplitude: float, hurst: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Build an (N, 2) synthetic coastline polyline with controlled roughness.

    The profile is laid along x in [0, 1] and its relief rescaled to span
    ``amplitude`` units, so every segment has comparable extent and only the
    texture (set by ``hurst``) drives differences in fractal dimension.
    """
    y = _midpoint_displacement(depth, hurst, rng)
    span = y.max() - y.min()
    if span > 0:
        y = (y - y.min()) / span * amplitude       # rescale relief to `amplitude`
    x = np.linspace(0.0, 1.0, y.size)
    return np.column_stack([x, y])


def generate_segments(config) -> list[Segment]:
    """Produce the list of coastline segments to analyse.

    In synthetic mode, returns ``config.n_segments`` fractal coastlines with
    Hurst exponents spread across ``[hurst_min, hurst_max]`` (a small random
    jitter per segment keeps them from being perfectly ordered). Locations cycle
    through a built-in list of Philippine coastal sites.

    Args:
        config: A :class:`src.utils.config.Config`.

    Returns:
        A list of :class:`Segment`.
    """
    if config.mode == "real":
        return _load_image_segments(config)
    if config.mode != "synthetic":
        raise ValueError(f"unknown mode {config.mode!r}; use 'synthetic' or 'real'")

    syn = config.synthetic
    n = config.n_segments
    # Independent, reproducible RNG stream per segment.
    seeds = np.random.SeedSequence(config.seed).spawn(n)

    # Hurst spread from smoothest to roughest, with deterministic jitter.
    hurst_grid = np.linspace(syn["hurst_max"], syn["hurst_min"], n)

    segments: list[Segment] = []
    for i, seed_seq in enumerate(seeds):
        rng = np.random.default_rng(seed_seq)
        jitter = rng.uniform(-0.03, 0.03)
        hurst = float(np.clip(hurst_grid[i] + jitter, 0.05, 0.99))
        coords = _synthetic_coastline(syn["depth"], syn["amplitude"], hurst, rng)

        location, province, region, lat, lon = _PH_LOCATIONS[i % len(_PH_LOCATIONS)]
        segments.append(
            Segment(
                id=f"{i + 1:02d}",
                location=location,
                province=province,
                region=region,
                lat=lat,
                lon=lon,
                coords=coords,
                hurst=hurst,
            )
        )
    return segments


_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")


def _load_image_segments(config) -> list[Segment]:
    """Load one coastline image per segment for real (image) mode.

    Reads every image in ``paths.images`` (sorted by filename) and enriches each
    with metadata from ``paths.segments_csv`` if that file exists. The CSV is
    matched on a ``filename`` column and may provide ``location``, ``province``,
    ``region``, ``lat``, ``lon`` and ``surge_height_m``. Missing metadata is
    left as blank / NaN; surge is validated later, only if needed.

    Raises:
        FileNotFoundError: If the images directory has no usable images.
    """
    images_dir = config.path("images")
    paths: list[str] = []
    for pattern in _IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(images_dir, pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"real mode: no coastline images found in {images_dir!r}. "
            f"Place one JPG per segment there (extensions: "
            f"{', '.join(_IMAGE_EXTENSIONS)})."
        )

    # Optional metadata table keyed by filename.
    meta: dict[str, dict] = {}
    csv_path = config.path("segments_csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            raise ValueError(f"{csv_path}: metadata CSV must have a 'filename' column")
        meta = {str(row["filename"]): row.to_dict() for _, row in df.iterrows()}

    segments: list[Segment] = []
    for i, path in enumerate(paths):
        fname = os.path.basename(path)
        row = meta.get(fname, {})

        def _get(key, default):
            value = row.get(key, default)
            return default if value is None or (isinstance(value, float) and np.isnan(value)) else value

        segments.append(
            Segment(
                id=str(_get("id", f"{i + 1:02d}")),
                location=str(_get("location", os.path.splitext(fname)[0])),
                province=str(_get("province", "")),
                region=str(_get("region", "")),
                lat=float(_get("lat", float("nan"))),
                lon=float(_get("lon", float("nan"))),
                image_path=path,
                surge_height_m=float(_get("surge_height_m", float("nan"))),
                ssa_level=str(_get("ssa_level", "")),
            )
        )
    return segments
