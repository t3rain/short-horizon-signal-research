import numpy as np
import pytest

from signal_research.metrics import spearman_ic
from signal_research.placebo import placebo_daily_ic, welch_t


def _days(n_days=10, n=4000, seed=0, link=0.5):
    """Signal and label that really are related within each day."""
    rng = np.random.default_rng(seed)
    sig = [rng.normal(size=n) for _ in range(n_days)]
    lab = [link * s + rng.normal(size=n) for s in sig]
    return sig, lab


def test_no_shift_is_the_ordinary_daily_ic():
    sig, lab = _days()
    ic = placebo_daily_ic(sig, lab)
    assert len(ic) == 10
    assert ic == pytest.approx([spearman_ic(s, y) for s, y in zip(sig, lab)])
    assert ic.mean() > 0.3


def test_other_day_and_rolled_placebos_have_no_skill():
    sig, lab = _days(n_days=12, seed=1)
    for ic in (placebo_daily_ic(sig, lab, day_shift=1),
               placebo_daily_ic(sig, lab, day_shift=-2),
               placebo_daily_ic(sig, lab, row_shift=1000)):
        assert abs(ic.mean()) < 0.03


def test_day_shift_direction_and_count():
    rng = np.random.default_rng(2)
    sig = [rng.normal(size=500) for _ in range(6)]
    lab = [np.zeros(500)] + [s.copy() for s in sig[:-1]]         # label of day i+1 equals signal of day i
    fwd = placebo_daily_ic(sig, lab, day_shift=1)
    assert len(fwd) == 5 and fwd == pytest.approx(np.ones(5))    # day i signal <-> day i+1 label
    assert len(placebo_daily_ic(sig, lab, day_shift=-2)) == 4    # 6 - |shift|
    assert len(placebo_daily_ic(sig, lab, day_shift=9)) == 0      # no partners at all


def test_row_shift_direction():
    rng = np.random.default_rng(3)
    s = rng.normal(size=1000)
    lab = np.roll(s, -50)                                         # lab[i] = s[i + 50]
    assert placebo_daily_ic([s], [lab], row_shift=50)[0] == pytest.approx(1.0)
    assert placebo_daily_ic([s], [lab], row_shift=0)[0] < 0.2


def test_nan_labels_are_ignored_after_a_roll():
    sig, lab = _days(n_days=3, seed=4)
    for y in lab:
        y[-30:] = np.nan                                          # like the end-of-day rows
    ic = placebo_daily_ic(sig, lab, row_shift=500)
    assert np.all(np.isfinite(ic))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        placebo_daily_ic([np.zeros(5)], [np.zeros(5), np.zeros(5)])


def test_welch_t_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.2, 1.0, 18), rng.normal(0.0, 2.0, 42)
    diff, t = welch_t(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert diff == pytest.approx(a.mean() - b.mean())
    assert t == pytest.approx(ref.statistic)


def test_welch_t_small_groups_give_nan():
    diff, t = welch_t(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(diff) and np.isnan(t)