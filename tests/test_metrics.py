import numpy as np
import pytest

from signal_research.metrics import spearman_ic, summarize_daily_ic


def test_handwritten():
    yhat = [0.2, -0.1, 0.5, 0.0, 0.3]
    y = [0.4, -0.2, 25.0, -0.1, 0.1]
    assert spearman_ic(yhat, y) == pytest.approx(0.9)


def test_matches_scipy_including_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 20, size=500).astype(float)
    b = a + rng.normal(size=500) * 5
    assert spearman_ic(a, b) == pytest.approx(scipy_stats.spearmanr(a, b)[0])


def test_monotone_transform_and_sign():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    b = a + rng.normal(size=300)
    assert spearman_ic(np.exp(a), b) == pytest.approx(spearman_ic(a, b))   # ranks only
    assert spearman_ic(-a, b) == pytest.approx(-spearman_ic(a, b))
    assert spearman_ic(a, a ** 3) == pytest.approx(1.0)


def test_huge_outlier():
    rng = np.random.default_rng(2)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    b[0], a[0] = 1e6, 1e6              # one gigantic point shared by both
    assert abs(np.corrcoef(a, b)[0, 1]) > 0.9
    assert abs(spearman_ic(a, b)) < 0.1


def test_degenerate_cases():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    assert spearman_ic(a, b) == pytest.approx(1.0)
    assert np.isnan(spearman_ic([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]))
    assert np.isnan(spearman_ic([1.0, 2.0], [1.0, 2.0]))


def test_summary_t_stat():
    ics = np.array([0.02, 0.06, 0.04, 0.08, 0.00])
    s = summarize_daily_ic(ics)
    assert s["mean_ic"] == pytest.approx(0.04)
    assert s["t_stat"] == pytest.approx(0.04 / (ics.std(ddof=1) / np.sqrt(5)))
    assert s["pos_days"] == pytest.approx(0.8)
    assert s["n_days"] == 5


def test_summary_behaviour():
    s = summarize_daily_ic([0.1, np.nan, 0.3])
    assert s["n_days"] == 2 and s["mean_ic"] == pytest.approx(0.2)
    assert summarize_daily_ic([np.nan])["n_days"] == 0