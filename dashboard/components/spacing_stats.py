"""Pure, testable statistics for the Spacing-vs-Production scatter.

Extracted from ``dashboard/pages/05_explore.py`` so the analytical logic — the
OLS trendline and the binned medians — can be unit-tested independently of the
Plotly figure assembly (which is brittle to test and changes often).

These functions carry the analytical risk of the chart: a wrong slope or R²
would silently mislead, since the dashboard never raises on a bad number. Keep
them pure (no Plotly, no Dash, no I/O) so the tests stay fast and meaningful.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def ols_fit(x, y) -> Optional[Tuple[float, float, float, int]]:
    """Degree-1 least-squares fit over finite ``(x, y)`` pairs.

    Returns ``(slope, intercept, r2, n)`` where ``n`` is the number of finite
    pairs used, or ``None`` when the fit is not well-defined:

    - fewer than 3 finite pairs, or
    - ``x`` has zero range (a vertical line has no OLS slope).

    ``r2`` is defined as 0.0 when ``y`` is constant (total sum of squares is 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xf, yf = x[mask], y[mask]
    if (xf.max() - xf.min()) == 0:
        return None
    slope, intercept = np.polyfit(xf, yf, 1)
    yhat = slope * xf + intercept
    ss_res = float(np.sum((yf - yhat) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), r2, int(mask.sum())


def binned_median(
    x, y, *, max_bins: int = 10, min_points: int = 6
) -> Optional[Tuple[list, list]]:
    """Median of ``y`` within equal-width bins of ``x``.

    Returns ``(centers, medians)`` (bin midpoints and per-bin median ``y``), or
    ``None`` when there are fewer than ``min_points`` finite pairs or ``x`` has
    zero range. The bin count scales with sample size (``len // 10``, floored at
    3) and is capped at ``max_bins``.
    """
    df = pd.DataFrame(
        {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}
    ).dropna()
    if len(df) < min_points:
        return None
    xv = df["x"].to_numpy()
    if (xv.max() - xv.min()) == 0:
        return None
    n_bins = min(max_bins, max(3, len(df) // 10))
    bins = pd.cut(df["x"], bins=n_bins)
    med = df.groupby(bins, observed=True)["y"].median()
    centers = [interval.mid for interval in med.index]
    return centers, list(med.values)
