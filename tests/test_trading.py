import numpy as np
import pytest

from signal_research.trading import (
    decile_means, gross_bps, pick_trades, roundtrip_spread_bps, trailing_threshold,
)


def test_roundtrip_spread_is_two_half_spreads_and_nan_at_the_end():
    bid, ask = np.full(100, 100.0), np.full(100, 100.2)
    s = roundtrip_spread_bps(bid, ask, latency_s=1, horizon_s=10)
    full_spread_bps = 0.2 / 100.1 * 1e4                         # ask - bid relative to mid
    assert s[:89] == pytest.approx(np.full(89, full_spread_bps))  # half in + half out
    assert np.isnan(s[89:]).all() and len(s) == 100              # last L + h = 11 rows


def test_roundtrip_spread_uses_the_rows_the_trade_touches():
    bid = np.full(20, 100.0)
    ask = np.full(20, 100.1)
    ask[4] = 101.0                                              # wide spread only at row 4
    s = roundtrip_spread_bps(bid, ask, latency_s=1, horizon_s=3)
    # s[i] = half[i + 1] (entry) + half[i + 4] (exit): row 4 is touched by i = 3 (entry) and i = 0 (exit)
    assert s[0] > s[1] and s[3] > s[1]
    assert s[1] == pytest.approx(s[2])                          # rows 1 and 2 touch only narrow rows


def test_pick_trades_holds_one_position_at_a_time():
    score = np.zeros(20)
    score[[0, 1, 2, 5, 6, 12]] = [1, -1, 1, 1, -1, 1]
    assert list(pick_trades(score, thr=0.5, hold=5)) == [0, 5, 12]
    assert list(pick_trades(score, thr=0.5, hold=1)) == [0, 1, 2, 5, 6, 12]
    assert len(pick_trades(np.zeros(5), 0.5, 3)) == 0
    assert len(pick_trades(np.full(5, np.nan), 0.5, 3)) == 0   # NaN scores are never traded


def test_gross_bps_goes_long_on_positive_and_short_on_negative_scores():
    score = np.array([2.0, -2.0, 2.0, -2.0])
    label = np.array([1.0, 1.0, -1.0, -1.0])
    assert list(gross_bps(score, label, np.arange(4))) == [1.0, -1.0, -1.0, 1.0]


def test_trailing_threshold_uses_only_earlier_days():
    days = [np.full(1000, 1.0), np.full(1000, 2.0), np.full(1000, 100.0)]
    assert np.isnan(trailing_threshold(days, 0, 0.1, window=5))
    assert trailing_threshold(days, 1, 0.1, window=5) == pytest.approx(1.0)      # day 0 only
    assert trailing_threshold(days, 2, 0.1, window=1) == pytest.approx(2.0)      # day 1 only, not day 2
    assert trailing_threshold(days, 2, 0.1, window=5) == pytest.approx(2.0)      # days 0-1 pooled, q-quantile


def test_decile_means_are_increasing_for_a_good_predictor():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=20000)
    label = pred + rng.normal(size=20000)
    d = decile_means(pred, label)
    assert len(d) == 10 and np.all(np.diff(d) > 0)
    assert d[-1] > 1.0 and d[0] < -1.0


def test_decile_means_handle_nans_and_tiny_input():
    pred = np.arange(100.0)
    label = np.arange(100.0)
    label[::7] = np.nan
    assert np.all(np.isfinite(decile_means(pred, label)))
    assert np.isnan(decile_means(np.arange(5.0), np.arange(5.0))).all()