import numpy as np
import pytest

from signal_research.labels import forward_mid_return_bps, make_labels


def test_label_uses_entry_at_i_plus_L_and_exit_at_i_plus_L_plus_h():
    mid = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    L, h = 1, 2
    y = forward_mid_return_bps(mid, h, L)
    for i in range(len(mid) - L - h):
        assert y[i] == pytest.approx((mid[i + L + h] / mid[i + L] - 1) * 1e4)
    # the last L + h rows have no exit price yet
    assert np.isnan(y[-(L + h):]).all()
    assert not np.isnan(y[: -(L + h)]).any()


def test_zero_latency_enters_at_the_row_state():
    mid = np.array([100.0, 110.0, 121.0])
    y = forward_mid_return_bps(mid, 1, 0)
    assert y[0] == pytest.approx(1000.0)          # +10 % = 1000 bps
    assert y[1] == pytest.approx(1000.0)
    assert np.isnan(y[2])


def test_label_ignores_everything_before_entry():
    mid = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    a = forward_mid_return_bps(mid, 1, 2)
    mid2 = mid.copy()
    mid2[:2] = [50.0, 500.0]      # prices before t + L must not matter for row 0
    b = forward_mid_return_bps(mid2, 1, 2)
    assert a[0] == b[0]


def test_short_series_is_all_nan_and_bad_args_raise():
    assert np.isnan(forward_mid_return_bps(np.array([1.0, 2.0]), 5, 1)).all()
    with pytest.raises(ValueError):
        forward_mid_return_bps(np.array([1.0, 2.0]), 0, 1)
    with pytest.raises(ValueError):
        forward_mid_return_bps(np.array([1.0, 2.0]), 1, -1)


def test_make_labels_returns_each_horizon():
    out = make_labels(np.linspace(100, 110, 100), horizons_s=(1, 5), latency_s=1)
    assert set(out) == {1, 5} and all(len(v) == 100 for v in out.values())
