"""Interactive Philippine map of fractal dimension per coastline segment.

Produces a Folium (Leaflet) map with one circle marker per segment, coloured on
a continuous scale by fractal dimension D — the choropleth-style deliverable in
outline §V-B. Saved as a standalone HTML file.
"""

from __future__ import annotations

import folium
import pandas as pd
from branca.colormap import LinearColormap

# Approximate geographic centre of the Philippines, for the initial map view.
_PH_CENTER = (12.5, 122.5)


def build_dimension_map(df: pd.DataFrame, out_path: str) -> str:
    """Render segments as colour-coded markers on a Philippine map.

    Args:
        df:       Results table with columns ``lat``, ``lon``,
                  ``fractal_dimension``, ``surge_height_m``, ``tier``,
                  ``location``, ``id``.
        out_path: Destination ``.html`` path.

    Returns:
        ``out_path``.
    """
    d_min = float(df["fractal_dimension"].min())
    d_max = float(df["fractal_dimension"].max())
    # Guard against a degenerate range (all-equal dimensions).
    if d_max <= d_min:
        d_max = d_min + 1e-6

    colormap = LinearColormap(
        colors=["#2c7bb6", "#ffffbf", "#d7191c"],  # blue (low D) -> red (high D)
        vmin=d_min,
        vmax=d_max,
        caption="Box-counting fractal dimension (D)",
    )

    fmap = folium.Map(location=_PH_CENTER, zoom_start=6, tiles="cartodbpositron")

    for _, row in df.iterrows():
        d = float(row["fractal_dimension"])
        popup = folium.Popup(
            html=(
                f"<b>{row['location']}</b> (#{row['id']})<br>"
                f"D = {d:.4f}<br>"
                f"Surge = {row['surge_height_m']:.2f} m<br>"
                f"Tier = {row['tier']}"
            ),
            max_width=250,
        )
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=8,
            color=colormap(d),
            fill=True,
            fill_color=colormap(d),
            fill_opacity=0.85,
            weight=1,
            popup=popup,
            tooltip=f"{row['location']}: D={d:.3f}",
        ).add_to(fmap)

    colormap.add_to(fmap)
    fmap.save(out_path)
    return out_path
