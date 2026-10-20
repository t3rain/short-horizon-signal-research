"""Forward-return labels.

For the row labelled t (row index i), with latency L and horizon h, both in whole seconds:

    entry  = mid at t + L          -> mid[i + L]
    exit   = mid at t + L + h      -> mid[i + L + h]
    label  = (exit / entry - 1) in basis points

mid[i] is the book state just before row i's end time, so mid[i + L] is the book as it
stands when an order sent at t arrives. L = 0 is the optimistic zero-latency case.
Rows whose exit lies beyond the end of the day get NaN.
"""

import numpy as np
from . import config

def forward_mid_return_bps(mid: np.ndarray, horizon_s: int, latency_s: int) -> np.ndarray:
    if horizon_s < 1 or latency_s < 0:
        raise ValueError("need horizon_s >= 1 and latency_s >= 0")
    mid = np.asarray(mid, dtype=np.float64)
    n = len(mid)
    out = np.full(n, np.nan)
    m = n - latency_s - horizon_s
    if m > 0:
        out[:m] = (mid[latency_s + horizon_s:] / mid[latency_s: n - horizon_s] - 1.0) * 1e4
    return out


def make_labels(mid: np.ndarray,
                horizons_s: tuple[int, ...] = config.HORIZONS_S,
                latency_s: int = config.LATENCY_S) -> dict[int, np.ndarray]:
    return {h: forward_mid_return_bps(mid, h, latency_s) for h in horizons_s}
