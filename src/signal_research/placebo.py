"""Placebo tests: pair a signal with labels it cannot possibly predict, and check IC is about 0.

If a signal has real predictive power, scoring it against the *right* labels gives a clearly
positive IC. If we then break the link between signal and label (but keep everything else the
same: same signal, same label distribution, same days), the IC must collapse to about 0.
If it does not, something other than prediction is producing the number: a pattern shared by
signal and label (typically time of day), or an evaluation that does not behave like the null
we assume when we quote a t-statistic.

What a placebo can NOT see: look-ahead inside a feature, or test labels leaking into
training. That leak is specific to the day's own labels, so it disappears when the labels are
swapped for another day's, and the placebo IC stays about 0 even though the real IC is
inflated. Those are covered by the perturbation tests (tests/test_features.py,
tests/test_walkforward.py) and by asking whether the real IC is plausible.

Two ways of breaking the link are provided by one function:

    day_shift = s   signal of day i is scored against the labels of day i + s
                    (same time of day, different day: the two never saw the same market)
    row_shift = m   labels of the same day are rotated by m rows (seconds)
                    (same day and regime, but the wrong moment in time)

day_shift = 0 and row_shift = 0 is the real, aligned IC.
"""
from __future__ import annotations

import numpy as np

from .metrics import spearman_ic


def placebo_daily_ic(signal_days: list[np.ndarray], label_days: list[np.ndarray],
                     day_shift: int = 0, row_shift: int = 0) -> np.ndarray:
    """One Spearman IC per day i that has a partner day i + day_shift.

    signal_days[i] and label_days[i] are 1-D arrays for day i (one row per second).
    Days whose partner falls outside the list are skipped, so the result has
    len(signal_days) - |day_shift| entries.
    """
    if len(signal_days) != len(label_days):
        raise ValueError("need one label array per signal array")
    out = []
    for i, sig in enumerate(signal_days):
        j = i + day_shift
        if not 0 <= j < len(label_days):
            continue
        lab = label_days[j]
        n = min(len(sig), len(lab))             # days are normally equal length; be safe
        sig, lab = sig[:n], lab[:n]
        if row_shift:
            lab = np.roll(lab, row_shift)       # rows pushed off the end wrap round to the start
        out.append(spearman_ic(sig, lab))
    return np.array(out)


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean(a) - mean(b), Welch t-statistic) for two groups that may differ in size and spread.

        t = (mean_a - mean_b) / sqrt(s_a^2 / n_a + s_b^2 / n_b)        s^2 with ddof = 1

    NaNs are dropped. Returns (nan, nan) when either group has fewer than 2 values.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    diff = float(a.mean() - b.mean())
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return diff, float(diff / se) if se > 0 else float("nan")