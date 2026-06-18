"""Statistical analysis of fractal dimension vs storm-surge height.

Implements the tests in outline §III-H using ``scipy.stats``:

    * Pearson's r           — linear correlation (assumes normality)
    * Spearman's rho        — rank correlation (non-parametric fallback)
    * Shapiro–Wilk          — normality of each variable
    * Simple linear regression with a 95% CI on the slope
    * One-way ANOVA across the three vulnerability tiers (optional)

:func:`analyze` returns a structured dict; :func:`format_report` renders it as a
human-readable report for the output/reports deliverable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

ALPHA = 0.05  # significance threshold (outline §III-H)


def _linear_regression_with_ci(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares fit y ~ x with a 95% confidence interval on the slope."""
    result = stats.linregress(x, y)
    n = len(x)
    # 95% CI on the slope via the t-distribution (n - 2 degrees of freedom).
    t_crit = stats.t.ppf(1 - ALPHA / 2, df=n - 2)
    margin = t_crit * result.stderr
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue**2),
        "p_value": float(result.pvalue),
        "std_error": float(result.stderr),
        "slope_ci_low": float(result.slope - margin),
        "slope_ci_high": float(result.slope + margin),
    }


def _correlation_ci(r: float, n: int) -> tuple[float, float]:
    """95% CI for a correlation coefficient via the Fisher z-transform."""
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def analyze(
    dimensions: np.ndarray,
    surge_heights: np.ndarray,
    tiers: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full statistical battery on paired (D, surge) samples.

    Args:
        dimensions:    Fractal dimension D per segment.
        surge_heights: Maximum surge height (m) per segment, same order.
        tiers:         Optional tier label per segment, for the ANOVA.

    Returns:
        A nested dict of results (correlations, normality, regression, ANOVA).
    """
    x = np.asarray(dimensions, dtype=float)
    y = np.asarray(surge_heights, dtype=float)
    n = len(x)
    if n != len(y):
        raise ValueError("dimensions and surge_heights must have equal length")
    if n < 4:
        raise ValueError(f"need at least 4 samples for these tests, got {n}")

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    shapiro_d = stats.shapiro(x)
    shapiro_s = stats.shapiro(y)

    results: dict[str, Any] = {
        "n": n,
        "alpha": ALPHA,
        "pearson": {
            "r": float(pearson_r),
            "p_value": float(pearson_p),
            "ci_low": _correlation_ci(float(pearson_r), n)[0],
            "ci_high": _correlation_ci(float(pearson_r), n)[1],
            "significant": bool(pearson_p < ALPHA),
        },
        "spearman": {
            "rho": float(spearman_r),
            "p_value": float(spearman_p),
            "significant": bool(spearman_p < ALPHA),
        },
        "normality": {
            "dimension": {
                "shapiro_W": float(shapiro_d.statistic),
                "p_value": float(shapiro_d.pvalue),
                "normal": bool(shapiro_d.pvalue > ALPHA),
            },
            "surge": {
                "shapiro_W": float(shapiro_s.statistic),
                "p_value": float(shapiro_s.pvalue),
                "normal": bool(shapiro_s.pvalue > ALPHA),
            },
        },
        "regression": _linear_regression_with_ci(x, y),
    }

    # One-way ANOVA of fractal dimension across tiers (optional, needs >=2 tiers
    # each with >=2 members).
    if tiers is not None:
        groups = [x[np.array(tiers) == t] for t in sorted(set(tiers))]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            f_stat, f_p = stats.f_oneway(*groups)
            results["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(f_p),
                "significant": bool(f_p < ALPHA),
                "n_groups": len(groups),
            }

    return results


def format_report(results: dict[str, Any]) -> str:
    """Render :func:`analyze` output as a plain-text report."""
    p = results["pearson"]
    s = results["spearman"]
    nd = results["normality"]["dimension"]
    ns = results["normality"]["surge"]
    reg = results["regression"]
    alpha = results["alpha"]

    # Pick the appropriate correlation test based on normality.
    both_normal = nd["normal"] and ns["normal"]
    primary = "Pearson (both variables normal)" if both_normal else \
        "Spearman (non-normal data — use rank correlation)"

    lines = [
        "=" * 70,
        "FRACTAL DIMENSION vs STORM SURGE HEIGHT - STATISTICAL REPORT",
        "=" * 70,
        f"Sample size (segments):      n = {results['n']}",
        f"Significance threshold:      alpha = {alpha}",
        "",
        "-- Normality (Shapiro-Wilk) ----------------------------------------",
        f"  Fractal dimension D:  W = {nd['shapiro_W']:.4f}, "
        f"p = {nd['p_value']:.4f}  -> {'normal' if nd['normal'] else 'NOT normal'}",
        f"  Surge height (m):     W = {ns['shapiro_W']:.4f}, "
        f"p = {ns['p_value']:.4f}  -> {'normal' if ns['normal'] else 'NOT normal'}",
        f"  Recommended primary test: {primary}",
        "",
        "-- Correlation -----------------------------------------------------",
        f"  Pearson  r   = {p['r']:.4f}  (95% CI [{p['ci_low']:.4f}, "
        f"{p['ci_high']:.4f}]), p = {p['p_value']:.4g}"
        f"  -> {'SIGNIFICANT' if p['significant'] else 'not significant'}",
        f"  Spearman rho = {s['rho']:.4f}, p = {s['p_value']:.4g}"
        f"  -> {'SIGNIFICANT' if s['significant'] else 'not significant'}",
        "",
        "-- Linear regression (surge ~ D) -----------------------------------",
        f"  surge = {reg['slope']:.4f} * D + ({reg['intercept']:.4f})",
        f"  slope 95% CI = [{reg['slope_ci_low']:.4f}, {reg['slope_ci_high']:.4f}]",
        f"  R^2 = {reg['r_squared']:.4f},  p = {reg['p_value']:.4g}",
    ]

    if "anova" in results:
        a = results["anova"]
        lines += [
            "",
            "-- One-way ANOVA (D across tiers) ----------------------------------",
            f"  F = {a['f_statistic']:.4f}, p = {a['p_value']:.4g} "
            f"({a['n_groups']} tiers) -> "
            f"{'SIGNIFICANT' if a['significant'] else 'not significant'}",
        ]

    # Hypothesis verdict (outline hypothesis: higher D -> higher surge).
    primary_sig = p["significant"] if both_normal else s["significant"]
    primary_dir = (p["r"] if both_normal else s["rho"]) > 0
    if primary_sig and primary_dir:
        verdict = "SUPPORTED: significant positive correlation between D and surge."
    elif primary_sig and not primary_dir:
        verdict = "REJECTED: significant correlation, but in the NEGATIVE direction."
    else:
        verdict = "NOT SUPPORTED: no statistically significant correlation found."

    lines += ["", "-- Hypothesis verdict ----------------------------------------------",
              f"  {verdict}", "=" * 70]
    return "\n".join(lines)
