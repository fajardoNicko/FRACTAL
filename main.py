"""FRACTAL — end-to-end pipeline orchestrator.

Runs the full study workflow:

    1. Load configuration.
    2. Generate / load coastline segments.
    3. For each segment: clean -> rasterize -> box-counting -> fractal dimension D.
    4. Obtain a maximum storm-surge height per segment (dependent variable).
    5. Classify each segment into a vulnerability tier.
    6. Assemble the results table (outline §V-A) and export it.
    7. Run the statistical analysis (outline §III-H) and write the report.
    8. Render the interactive map and analytical charts.

Run with:
    python main.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.fractal.box_counting import box_counting_dimension
from src.maps.folium_maps import build_dimension_map
from src.maps.plotly_maps import boxplot_dimension_by_tier, scatter_dimension_vs_surge
from src.preprocessing.coastline_cleaner import clean_coords
from src.preprocessing.image_coastline import coastline_diagnostics, extract_coastline
from src.preprocessing.raster_converter import rasterize_polyline
from src.preprocessing.segment_generator import generate_segments
from src.stormsurge import statistics
from src.stormsurge.raster_extractor import get_surge_heights
from src.stormsurge.vulnerability_scorer import classify_tiers
from src.utils.config import load_config
from src.utils.file_helpers import ensure_dir, save_dataframe, save_json, save_text
from src.utils.logger import get_logger

log = get_logger("fractal")


# A segment's box-counting D is trustworthy only if the sheet holds enough
# coastline. The hard auto-exclusion is a geometrically IMPOSSIBLE dimension
# (D < ~1.02) — which is what sheets dominated by open sea with a few tiny
# scattered islands produce. Sheets with little land but a plausible D (island
# archipelagos like El Nido) are KEPT but get a low-land advisory, so the
# researcher can decide whether to exclude them too.
MIN_PLAUSIBLE_D = 1.02      # D below this is geometrically implausible
LOW_LAND_ADVISORY = 0.035   # below this, flag as archipelago/open-sea for review


def compute_dimensions(segments, config):
    """Clean/extract and box-count every segment, with reliability QA.

    Returns:
        dict of aligned arrays: dimension, r_squared, reliable, sea_fraction,
        largest_land_fraction.
    """
    dims, r2, reliable, sea_frac, land_frac = [], [], [], [], []
    # Separate the preview toggle from the extractor kwargs.
    img_kwargs = dict(config.image)
    save_previews = img_kwargs.pop("save_previews", False)
    previews_dir = config.path("previews")

    for seg in segments:
        sea_f, land_f = float("nan"), float("nan")
        if seg.image_path is not None:
            # Real mode: extract the coastline (land/sea boundary) from the map image.
            debug_path = (os.path.join(previews_dir, f"coastline_{seg.id}.png")
                          if save_previews else None)
            if debug_path:
                ensure_dir(previews_dir)
            image = extract_coastline(seg.image_path, debug_path=debug_path, **img_kwargs)
            diag = coastline_diagnostics(
                seg.image_path, method=img_kwargs.get("method", "otsu"),
                crop_border_frac=img_kwargs.get("crop_border_frac", 0.0),
                hue_lo=img_kwargs.get("hue_lo", 120.0), hue_hi=img_kwargs.get("hue_hi", 185.0),
                sat_min=img_kwargs.get("sat_min", 25.0), val_min=img_kwargs.get("val_min", 40.0))
            sea_f, land_f = diag["sea_fraction"], diag["largest_land_fraction"]
        else:
            # Synthetic mode: rasterize the vector coastline to a binary image.
            coords = clean_coords(seg.coords)
            image = rasterize_polyline(coords, size=config.raster_size,
                                       padding=config.raster_padding)
        result = box_counting_dimension(image, config.box_sizes)

        # Reliability: synthetic data is always reliable; real sheets are
        # auto-excluded only for a geometrically impossible dimension. A low land
        # fraction with a plausible D is kept but noted for the researcher.
        ok, note = True, ""
        if seg.image_path is not None:
            if result.dimension < MIN_PLAUSIBLE_D:
                ok = False
                note = f"  [EXCLUDED: implausible D={result.dimension:.3f} " \
                       f"(open sea, land {land_f*100:.1f}%)]"
            elif land_f < LOW_LAND_ADVISORY:
                note = f"  [review: archipelago, land {land_f*100:.1f}%]"

        dims.append(result.dimension); r2.append(result.r_squared)
        reliable.append(ok); sea_frac.append(sea_f); land_frac.append(land_f)
        log.info("  #%s %-26s D = %.4f  (R^2 = %.4f)%s",
                 seg.id, seg.location, result.dimension, result.r_squared, note)
    return {
        "dimension": np.array(dims), "r_squared": np.array(r2),
        "reliable": np.array(reliable), "sea_fraction": np.array(sea_frac),
        "largest_land_fraction": np.array(land_frac),
    }


def build_results_table(segments, dim, surge, tiers) -> pd.DataFrame:
    """Assemble the §V-A results DataFrame (dim is the compute_dimensions dict)."""
    return pd.DataFrame({
        "id": [s.id for s in segments],
        "location": [s.location for s in segments],
        "province": [s.province for s in segments],
        "region": [s.region for s in segments],
        "lat": [s.lat for s in segments],
        "lon": [s.lon for s in segments],
        "fractal_dimension": np.round(dim["dimension"], 4),
        "r_squared": np.round(dim["r_squared"], 4),
        "largest_land_fraction": np.round(dim["largest_land_fraction"], 4),
        "reliable": dim["reliable"],
        "surge_height_m": np.round(surge, 2),
        "tier": tiers,
    })


def main() -> None:
    config = load_config()
    np.random.seed(config.seed)  # belt-and-suspenders global determinism
    log.info("FRACTAL pipeline starting (mode=%s, n=%d, raster=%dpx, seed=%d)",
             config.mode, config.n_segments, config.raster_size, config.seed)

    # 2-3. Segments -> fractal dimensions.
    log.info("Generating %d coastline segments and computing fractal dimensions...",
             config.n_segments)
    segments = generate_segments(config)
    dim = compute_dimensions(segments, config)
    dimensions = dim["dimension"]

    # 4-5. Surge heights and vulnerability tiers.
    surge = get_surge_heights(segments, dimensions, config)
    tiers = classify_tiers(surge, config)

    # 6. Results table -> CSV.
    df = build_results_table(segments, dim, surge, tiers)
    table_path = os.path.join(config.path("tables"), "results.csv")
    save_dataframe(df, table_path)
    log.info("Results table written to %s", table_path)

    # 7. Statistics -> JSON + text report. Exclude QA-flagged segments so an
    #    undefined coastline dimension never enters the correlation.
    reliable = df["reliable"].to_numpy()
    n_excluded = int((~reliable).sum())
    if n_excluded:
        log.warning("Excluding %d segment(s) with unreliable coastline D from "
                    "statistics: %s", n_excluded,
                    ", ".join(df.loc[~reliable, "id"].tolist()))
    log.info("Running statistical analysis on %d reliable segment(s)...",
             int(reliable.sum()))
    results = statistics.analyze(dimensions[reliable], surge[reliable],
                                 [t for t, k in zip(tiers, reliable) if k])
    results["n_excluded"] = n_excluded
    report = statistics.format_report(results)
    save_json(results, os.path.join(config.path("reports"), "statistics.json"))
    save_text(report, os.path.join(config.path("reports"), "statistical_report.txt"))
    print("\n" + report + "\n")

    # 8. Visualizations. The map shows all segments; the analytical charts use
    #    only the reliable subset (matching the statistics).
    log.info("Rendering map and charts...")
    figures = config.path("figures")
    df_ok = df[df["reliable"]]
    build_dimension_map(df, os.path.join(figures, "dimension_map.html"))
    scatter_dimension_vs_surge(
        df_ok, results["regression"], results["pearson"]["r"],
        os.path.join(figures, "scatter_dimension_vs_surge.html"),
    )
    boxplot_dimension_by_tier(df_ok, os.path.join(figures, "boxplot_by_tier.html"))

    # Tier counts summary.
    counts = df["tier"].value_counts().to_dict()
    log.info("Tier distribution: %s", counts)
    log.info("Fractal dimension range: %.4f - %.4f",
             df["fractal_dimension"].min(), df["fractal_dimension"].max())
    log.info("Done. Outputs in output/tables, output/reports, output/figures.")


if __name__ == "__main__":
    main()
