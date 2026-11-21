"""Helpers for the PnL.

Conventions (same as labels.py): a signal on row i is acted on at t_i + L, the position is held
for h seconds, and its gross PnL is  sign(signal) * label[i] in bps per notional.
"""
import numpy as np
from scipy.stats import rankdata

# Spread paid by a taker who enters at row i+L and exits at row i+L+h, in bps of mid.
# Crossing the book costs half the spread on the way in and half on the way out. This is extra cost
# on top of the mid-to-mid returns.
def roundtrip_spread_bps(bid: np.ndarray, ask: np.ndarray, latency_s: int,
                         horizon_s: int) -> np.ndarray:
    bid = np.asarray(bid, dtype=np.float64)
    ask = np.asarray(ask, dtype=np.float64)
    half = (ask - bid) / 2.0 / ((ask + bid) / 2.0) * 1e4       
    n = len(half)
    out = np.full(n, np.nan)
    m = n - latency_s - horizon_s
    if m > 0:
        out[:m] = half[latency_s: latency_s + m] + half[latency_s + horizon_s: latency_s + horizon_s + m]
    return out


def trailing_threshold(abs_preds_by_day: list[np.ndarray], k: int, q: float, window: int) -> float:
    """The |prediction| level that the top fraction q of rows exceeded over the `window` days
    BEFORE day k. Known in real time (it uses no row of day k). NaN for the first day."""
    if k == 0:
        return float("nan")
    pool = np.concatenate([a[np.isfinite(a)] for a in abs_preds_by_day[max(0, k - window):k]])
    return float(np.quantile(pool, 1.0 - q)) if len(pool) else float("nan")


def pick_trades(score: np.ndarray, thr: float, hold: int) -> np.ndarray:
    """Row indices of the trades of a strategy that holds at most one position at a time.

    Walk forward through the rows where |score| > thr. Take the first; ignore every signal in the
    next `hold` rows (the position is still open); then take the next one, and so on. Latency
    delays entry and exit equally, so the position is busy for `hold` = h rows, not L + h.
    """
    cand = np.flatnonzero(np.abs(score) > thr)                    # NaN > thr is False: skipped
    taken, free_from = [], 0
    for i in cand:
        if i >= free_from:
            taken.append(i)
            free_from = i + hold
    return np.array(taken, dtype=int)

# Gross PnL (bps of notional) of trades at rows idx
# long if score > 0
# short if score < 0
def gross_bps(score: np.ndarray, label: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gross PnL (bps of notional) of the trades at rows idx: long if score > 0, short if < 0."""
    return np.sign(score[idx]) * label[idx]


def decile_means(pred: np.ndarray, label: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Mean label in each prediction bin (bin 0 = lowest predictions) for one day.

    Bins hold equal numbers of rows by rank; equal predictions share a bin. NaN for empty bins.
    """
    ok = np.isfinite(pred) & np.isfinite(label)
    out = np.full(n_bins, np.nan)
    if ok.sum() < n_bins:
        return out
    r = rankdata(pred[ok])                                        # 1..n, ties share an average rank
    b = np.minimum(((r - 0.5) / ok.sum() * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins)
    tot = np.bincount(b, weights=label[ok], minlength=n_bins)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)