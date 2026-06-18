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
from src.preprocessing.image_coastline import extract_coastline
from src.preprocessing.raster_converter import rasterize_polyline
from src.preprocessing.segment_generator import generate_segments
from src.stormsurge import statistics
from src.stormsurge.raster_extractor import get_surge_heights
from src.stormsurge.vulnerability_scorer import classify_tiers
from src.utils.config import load_config
from src.utils.file_helpers import ensure_dir, save_dataframe, save_json, save_text
from src.utils.logger import get_logger

log = get_logger("fractal")


def compute_dimensions(segments, config) -> tuple[np.ndarray, np.ndarray]:
    """Clean, rasterize and box-count every segment.

    Returns:
        (dimensions, r_squared) arrays aligned with ``segments``.
    """
    dimensions, r_squared = [], []
    # Separate the preview toggle from the extractor kwargs.
    img_kwargs = dict(config.image)
    save_previews = img_kwargs.pop("save_previews", False)
    previews_dir = config.path("previews")

    for seg in segments:
        if seg.image_path is not None:
            # Real mode: extract the coastline (land/sea boundary) from the map image.
            debug_path = (os.path.join(previews_dir, f"coastline_{seg.id}.png")
                          if save_previews else None)
            if debug_path:
                ensure_dir(previews_dir)
            image = extract_coastline(seg.image_path, debug_path=debug_path, **img_kwargs)
        else:
            # Synthetic mode: rasterize the vector coastline to a binary image.
            coords = clean_coords(seg.coords)
            image = rasterize_polyline(coords, size=config.raster_size,
                                       padding=config.raster_padding)
        result = box_counting_dimension(image, config.box_sizes)
        dimensions.append(result.dimension)
        r_squared.append(result.r_squared)
        flag = "  [FLAGGED: R^2 < %.2f]" % config.min_r_squared if result.flagged else ""
        log.info("  #%s %-26s D = %.4f  (R^2 = %.4f)%s",
                 seg.id, seg.location, result.dimension, result.r_squared, flag)
    return np.array(dimensions), np.array(r_squared)


def build_results_table(segments, dimensions, r_squared, surge, tiers) -> pd.DataFrame:
    """Assemble the §V-A results DataFrame."""
    return pd.DataFrame({
        "id": [s.id for s in segments],
        "location": [s.location for s in segments],
        "province": [s.province for s in segments],
        "region": [s.region for s in segments],
        "lat": [s.lat for s in segments],
        "lon": [s.lon for s in segments],
        "fractal_dimension": np.round(dimensions, 4),
        "r_squared": np.round(r_squared, 4),
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
    dimensions, r_squared = compute_dimensions(segments, config)

    # 4-5. Surge heights and vulnerability tiers.
    surge = get_surge_heights(segments, dimensions, config)
    tiers = classify_tiers(surge, config)

    # 6. Results table -> CSV.
    df = build_results_table(segments, dimensions, r_squared, surge, tiers)
    table_path = os.path.join(config.path("tables"), "results.csv")
    save_dataframe(df, table_path)
    log.info("Results table written to %s", table_path)

    # 7. Statistics -> JSON + text report.
    log.info("Running statistical analysis...")
    results = statistics.analyze(dimensions, surge, tiers)
    report = statistics.format_report(results)
    save_json(results, os.path.join(config.path("reports"), "statistics.json"))
    save_text(report, os.path.join(config.path("reports"), "statistical_report.txt"))
    print("\n" + report + "\n")

    # 8. Visualizations.
    log.info("Rendering map and charts...")
    figures = config.path("figures")
    build_dimension_map(df, os.path.join(figures, "dimension_map.html"))
    scatter_dimension_vs_surge(
        df, results["regression"], results["pearson"]["r"],
        os.path.join(figures, "scatter_dimension_vs_surge.html"),
    )
    boxplot_dimension_by_tier(df, os.path.join(figures, "boxplot_by_tier.html"))

    # Tier counts summary.
    counts = df["tier"].value_counts().to_dict()
    log.info("Tier distribution: %s", counts)
    log.info("Fractal dimension range: %.4f - %.4f",
             df["fractal_dimension"].min(), df["fractal_dimension"].max())
    log.info("Done. Outputs in output/tables, output/reports, output/figures.")


if __name__ == "__main__":
    main()
