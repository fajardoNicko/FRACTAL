"""Interactive analytical charts (Plotly), saved as standalone HTML.

Covers two of the outline §V-B visualizations:
    * scatter of fractal dimension D vs maximum surge height, with the fitted
      regression line and Pearson r annotated;
    * box plots of D across the three vulnerability tiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.stormsurge.vulnerability_scorer import TIER_ORDER


def scatter_dimension_vs_surge(
    df: pd.DataFrame,
    regression: dict,
    pearson_r: float,
    out_path: str,
) -> str:
    """Scatter of D vs surge height with the regression line overlaid.

    Args:
        df:         Results table with ``fractal_dimension``, ``surge_height_m``,
                    ``location``, ``tier``.
        regression: Dict with ``slope`` and ``intercept`` (from statistics).
        pearson_r:  Pearson correlation coefficient, for the annotation.
        out_path:   Destination ``.html`` path.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["fractal_dimension"],
            y=df["surge_height_m"],
            mode="markers",
            marker=dict(size=10, color=df["surge_height_m"],
                        colorscale="Turbo", showscale=True,
                        colorbar=dict(title="Surge (m)")),
            text=df["location"] + " (" + df["tier"] + ")",
            hovertemplate="%{text}<br>D=%{x:.4f}<br>Surge=%{y:.2f} m<extra></extra>",
            name="segments",
        )
    )

    # Regression line across the observed D range.
    x_line = np.linspace(df["fractal_dimension"].min(), df["fractal_dimension"].max(), 50)
    y_line = regression["slope"] * x_line + regression["intercept"]
    fig.add_trace(
        go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="black", dash="dash"),
            name=f"fit (r = {pearson_r:.3f})",
        )
    )

    fig.update_layout(
        title="Fractal Dimension vs Maximum Storm Surge Height",
        xaxis_title="Box-counting fractal dimension (D)",
        yaxis_title="Maximum surge height (m)",
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def boxplot_dimension_by_tier(df: pd.DataFrame, out_path: str) -> str:
    """Box plots of fractal dimension within each vulnerability tier."""
    fig = go.Figure()
    for tier in TIER_ORDER:
        subset = df[df["tier"] == tier]
        if subset.empty:
            continue
        fig.add_trace(
            go.Box(
                y=subset["fractal_dimension"],
                name=tier,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                text=subset["location"],
            )
        )
    fig.update_layout(
        title="Fractal Dimension by Surge Vulnerability Tier",
        yaxis_title="Box-counting fractal dimension (D)",
        xaxis_title="Vulnerability tier",
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
