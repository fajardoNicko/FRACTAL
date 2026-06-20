"""Configuration loading.

The defaults below are the single source of truth, so the pipeline runs even if
``config.yaml`` (or PyYAML) is missing. When ``config.yaml`` is present and
PyYAML is installed, its values are deep-merged over the defaults, letting the
researcher tweak parameters without touching code.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

# Project root = two levels up from this file (src/utils/config.py -> FRACTAL/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Baked-in defaults (mirror config.yaml) --------------------------------- #
DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "mode": "synthetic",
    "n_segments": 30,
    "raster": {"size": 2048, "padding": 0.05},
    "box_counting": {
        "box_sizes": [4, 8, 16, 32, 64, 128, 256],
        "min_r_squared": 0.97,
    },
    "tiers": {"high_min": 2.5, "moderate_min": 1.0},
    # Coastline extraction from raster map images (used in "real" mode).
    "image": {
        "method": "otsu",          # "otsu" (brightness) or "sea_color" (blue sea)
        "threshold": "otsu",       # otsu method: "otsu" or a fixed gray level 0-255
        "keep_largest": True,      # keep largest region (drop text/legend specks)
        "fill_holes": True,        # fill interior holes (labels/contours in sea)
        "crop_border_frac": 0.0,   # crop this fraction off each side (map frames)
        "hue_lo": 120.0,           # sea_color: hue band (PIL scale; blue ~= 170)
        "hue_hi": 185.0,
        "sat_min": 25.0,           # sea_color: min saturation (rejects white paper)
        "val_min": 40.0,           # sea_color: min brightness
        "open_iterations": 0,      # sea_color: opening to sever rivers (0 = off)
        "min_feature_area": 2000.0,  # sea_color: fill sea holes smaller than this
                                     # (drops contour/text); keep larger (islands)
        "save_previews": False,    # save coastline overlay per segment for review
    },
    "synthetic": {
        "depth": 9,
        "amplitude": 0.6,
        "hurst_min": 0.45,
        "hurst_max": 0.95,
        "surge_slope": 9.0,
        "surge_base": 0.3,
        "surge_noise": 0.45,
    },
    "paths": {
        "data_raw": "data/raw",
        "images": "data/raw/images",
        "segments_csv": "data/raw/segments.csv",
        "data_processed": "data/processed",
        "data_exports": "data/exports",
        "figures": "output/figures",
        "previews": "output/previews",
        "reports": "output/reports",
        "tables": "output/tables",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class Config:
    """Resolved configuration with convenient typed accessors.

    The raw merged dictionary is kept in :attr:`data`; the properties expose the
    values the pipeline reads most often and resolve paths to absolute form.
    """

    data: dict[str, Any]
    root: str = PROJECT_ROOT

    # --- scalars --- #
    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def mode(self) -> str:
        return str(self.data["mode"])

    @property
    def n_segments(self) -> int:
        return int(self.data["n_segments"])

    # --- nested sections --- #
    @property
    def raster_size(self) -> int:
        return int(self.data["raster"]["size"])

    @property
    def raster_padding(self) -> float:
        return float(self.data["raster"]["padding"])

    @property
    def box_sizes(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.data["box_counting"]["box_sizes"])

    @property
    def min_r_squared(self) -> float:
        return float(self.data["box_counting"]["min_r_squared"])

    @property
    def tier_high_min(self) -> float:
        return float(self.data["tiers"]["high_min"])

    @property
    def tier_moderate_min(self) -> float:
        return float(self.data["tiers"]["moderate_min"])

    @property
    def synthetic(self) -> dict[str, Any]:
        return self.data["synthetic"]

    @property
    def image(self) -> dict[str, Any]:
        """Keyword arguments for image_coastline.extract_coastline."""
        return self.data["image"]

    def path(self, key: str) -> str:
        """Return an absolute path for one of the configured output locations."""
        return os.path.join(self.root, self.data["paths"][key])


def load_config(config_path: str | None = None) -> Config:
    """Load configuration, merging ``config.yaml`` over the baked-in defaults.

    Args:
        config_path: Path to a YAML config. Defaults to ``config.yaml`` at the
            project root. Missing file or missing PyYAML => defaults are used.

    Returns:
        A resolved :class:`Config`.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")

    merged = DEFAULTS
    if os.path.exists(config_path):
        try:
            import yaml  # optional dependency

            with open(config_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            merged = _deep_merge(DEFAULTS, loaded)
        except ImportError:
            # PyYAML not installed — fall back to defaults silently. The caller's
            # logger will note the mode/params actually in effect.
            pass

    return Config(data=merged)
