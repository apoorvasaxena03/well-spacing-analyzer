"""Unit tests for dashboard/components/spacing_stats.py.

Covers the analytical logic behind the Spacing-vs-Production scatter — the OLS
trendline and the binned medians. These carry real risk: a wrong slope/R2 would
silently mislead, since the dashboard never raises on a bad number.

No Plotly, no Dash, no I/O — pure numeric checks.
"""

import numpy as np
import pytest

from dashboard.components.spacing_stats import ols_fit, binned_median

pytestmark = pytest.mark.unit


class TestOlsFit:
    def test_recovers_known_line_exactly(self):
        x = np.arange(50, dtype=float)
        y = 2.5 * x + 7.0  # perfect line
        slope, intercept, r2, n = ols_fit(x, y)
        assert slope == pytest.approx(2.5, abs=1e-9)
        assert intercept == pytest.approx(7.0, abs=1e-9)
        assert r2 == pytest.approx(1.0, abs=1e-9)
        assert n == 50

    def test_positive_slope_for_interference_signal(self):
        # farther from parent (larger x) -> higher per-ft production
        rng = np.random.RandomState(0)
        x = np.linspace(200, 1800, 200)
        y = 0.02 * x + rng.normal(0, 1, 200)
        slope, intercept, r2, n = ols_fit(x, y)
        assert slope > 0
        assert 0.0 <= r2 <= 1.0
        assert n == 200

    def test_ignores_non_finite_pairs(self):
        # the NaN-x row (with an outlier y) must be dropped, not skew the fit
        x = np.array([1, 2, 3, 4, np.nan, 6], dtype=float)
        y = np.array([2, 4, 6, 8, 999, 12], dtype=float)
        slope, intercept, r2, n = ols_fit(x, y)
        assert n == 5
        assert slope == pytest.approx(2.0, abs=1e-9)
        assert intercept == pytest.approx(0.0, abs=1e-9)

    def test_none_when_fewer_than_three_points(self):
        assert ols_fit([1.0, 2.0], [1.0, 2.0]) is None

    def test_none_when_x_has_zero_range(self):
        # vertical cloud -> no OLS slope; must not divide by zero
        assert ols_fit([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]) is None

    def test_r2_is_zero_when_y_constant(self):
        slope, intercept, r2, n = ols_fit([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0])
        assert r2 == 0.0
        assert slope == pytest.approx(0.0, abs=1e-9)
        assert intercept == pytest.approx(5.0, abs=1e-9)


class TestBinnedMedian:
    def test_centers_and_medians_align_and_are_capped(self):
        x = np.linspace(0, 100, 100)
        y = x.copy()
        out = binned_median(x, y, max_bins=5)
        assert out is not None
        centers, medians = out
        assert len(centers) == len(medians)
        assert len(centers) <= 5
        # monotonic y -> bin medians increase with x
        assert medians == sorted(medians)

    def test_none_when_too_few_points(self):
        assert binned_median([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None

    def test_none_when_x_has_zero_range(self):
        assert binned_median([4.0] * 10, list(range(10))) is None

    def test_drops_non_finite_rows(self):
        x = list(np.linspace(0, 100, 40)) + [np.nan, np.nan]
        y = list(np.linspace(0, 100, 40)) + [0.0, 0.0]
        out = binned_median(x, y, max_bins=4)
        assert out is not None
        centers, medians = out
        assert all(np.isfinite(m) for m in medians)
