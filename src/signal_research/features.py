"""Per-second feature table from raw Binance bookTicker + aggTrades events.

Conventions:

  * One row per second of a UTC day. Row i is labelled with its END time
    t_i = day_start + (i + 1) seconds, so the row covers the window [t_i - 1 s, t_i).
  * STATE features (prices, sizes, mid, microprice, imbalance, ...) describe the last
    book update strictly BEFORE t_i, forward-filled when nothing arrived in the window.
  * FLOW features (OFI, signed trade volume, update count) are SUMS over the events in
    [t_i - 1 s, t_i), plus trailing sums over whole seconds that end at t_i.
  * Nothing at or after t_i is ever used for row i. tests/test_features.py checks this
    with a perturbation test.
  * Prices are compared as integer ticks, never as floats.

All functions here are pure numpy so they can be tested without any data files.
"""

import numpy as np
from . import config

SECONDS_PER_DAY = 86_400

# price -> integer ticks
def to_ticks(price: np.ndarray, tick_size: float) -> np.ndarray:
    return np.rint(np.asarray(price, dtype=np.float64) / tick_size).astype(np.int64)

# end times of each 1-s window, absolute timestamps
def grid_end_times(day_start_ms: int, n_rows: int = SECONDS_PER_DAY) -> np.ndarray:
    return day_start_ms + 1000 * np.arange(1, n_rows + 1, dtype=np.int64)

# ts_ms -> all events in the day; end_ms -> end times computed above
# gives index of the last event strictly before each end times
# NOTE: if no event happens before, index is -1, wrapping to the other end
# this case has to be handled seperately
def last_event_before(ts_ms: np.ndarray, end_ms: np.ndarray) -> np.ndarray:
    return np.searchsorted(ts_ms, end_ms, side="left") - 1  # left -> events strictly less than end_ms_i

# sum all the events in values in each 1-s window
def sum_by_second(ts_ms: np.ndarray, values: np.ndarray,
                  day_start_ms: int, n_rows: int) -> np.ndarray:
    b = (ts_ms - day_start_ms) // 1000
    keep = (b >= 0) & (b < n_rows)  # keep is a T/F array
    # indexing an array with T/F array keeps T positions
    # bucket into n_rows with buckets from b[keep] and weights from values[keep]
    out = np.bincount(b[keep], weights=values[keep], minlength=n_rows)  
    return out.astype(np.float64)       # empty input would otherwise give an int array

# out[i] = x[i-window+1] + ... + x[i], NaN while fewer than window rows exist
def trailing_sum(x: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(np.asarray(x, dtype=np.float64))
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    out[: window - 1] = np.nan
    return out

# past return over horizon k
def lagged_return_bps(mid: np.ndarray, k: int) -> np.ndarray:
    out = np.full(mid.shape, np.nan)
    if k < len(mid):
        out[k:] = (mid[k:] / mid[:-k] - 1.0) * 1e4
    return out

# OFI
def ofi_per_event(bid_ticks: np.ndarray, bid_qty: np.ndarray,
                  ask_ticks: np.ndarray, ask_qty: np.ndarray) -> np.ndarray:
    """Order-flow-imbalance contribution of every best-quote update (Cont, Kukanov, Stoikov 2014).

    e_n = 1{Pb_n >= Pb_n-1} qb_n - 1{Pb_n <= Pb_n-1} qb_n-1
          - 1{Pa_n <= Pa_n-1} qa_n + 1{Pa_n >= Pa_n-1} qa_n-1

    Positive means net buying pressure (bid got stronger or ask got weaker).
    The first event has no predecessor, so its contribution is 0.
    """
    n = len(bid_ticks)
    e = np.zeros(n)
    if n < 2:
        return e
    pb_n, qb_n = bid_ticks[1:], bid_qty[1:]
    pb_n_1, qb_n_1 = bid_ticks[:-1], bid_qty[:-1]
    pa_n, qa_n = ask_ticks[1:], ask_qty[1:]
    pa_n_1, qa_n_1 = ask_ticks[:-1], ask_qty[:-1]
    bid_part = np.where(pb_n >= pb_n_1, qb_n, 0.0) - np.where(pb_n <= pb_n_1, qb_n_1, 0.0)
    ask_part = -np.where(pa_n <= pa_n_1, qa_n, 0.0) + np.where(pa_n >= pa_n_1, qa_n_1, 0.0)
    e[1:] = bid_part + ask_part
    return e


# main builder
def build_features(
    book_ts_ms: np.ndarray, bid_px: np.ndarray, bid_qty: np.ndarray,
    ask_px: np.ndarray, ask_qty: np.ndarray,
    trade_ts_ms: np.ndarray, trade_qty: np.ndarray, trade_is_buyer_maker: np.ndarray,
    day_start_ms: int,
    tick_size: float = config.TICK_SIZE,
    n_rows: int = SECONDS_PER_DAY,
    flow_windows_s: tuple[int, ...] = config.FLOW_WINDOWS_S,
    ret_windows_s: tuple[int, ...] = config.RET_WINDOWS_S,
) -> dict[str, np.ndarray]:
    """Return {column name: array of length n_rows} following the conventions above."""
    book_ts_ms = np.asarray(book_ts_ms, dtype=np.int64)
    trade_ts_ms = np.asarray(trade_ts_ms, dtype=np.int64)
    if len(book_ts_ms) == 0:
        raise ValueError("no book updates")
    if np.any(np.diff(book_ts_ms) < 0) or np.any(np.diff(trade_ts_ms) < 0):
        raise ValueError("timestamps must be non-decreasing (sort by update_id first)")

    bid_px = np.asarray(bid_px, dtype=np.float64)
    ask_px = np.asarray(ask_px, dtype=np.float64)
    bid_qty = np.asarray(bid_qty, dtype=np.float64)
    ask_qty = np.asarray(ask_qty, dtype=np.float64)

    end_ms = grid_end_times(day_start_ms, n_rows)
    cols: dict[str, np.ndarray] = {"t_end_ms": end_ms}

    # get index before each end timestamps
    idx = last_event_before(book_ts_ms, end_ms)
    ok = idx >= 0
    j = np.where(ok, idx, 0)

    def pick(a: np.ndarray) -> np.ndarray:
        return np.where(ok, a[j], np.nan)

    # find values at each last_event_before indices
    b, a = pick(bid_px), pick(ask_px)
    bq, aq = pick(bid_qty), pick(ask_qty)
    mid = (b + a) / 2.0
    micro = (b * aq + a * bq) / (bq + aq)

    # only concern at this point is the values at last_event_before indices
    cols["bid_px"] = b
    cols["ask_px"] = a
    cols["mid"] = mid
    cols["spread_ticks"] = np.rint((a - b) / tick_size)
    cols["imbalance"] = (bq - aq) / (bq + aq)
    cols["depth"] = bq + aq
    cols["micro_dev_bps"] = (micro - mid) / mid * 1e4
    for k in ret_windows_s:
        cols[f"ret_{k}s"] = lagged_return_bps(mid, k)

    # flows from the raw book stream
    e = ofi_per_event(to_ticks(bid_px, tick_size), bid_qty,
                      to_ticks(ask_px, tick_size), ask_qty)
    ofi_1s = sum_by_second(book_ts_ms, e, day_start_ms, n_rows)
    cols["n_updates_1s"] = sum_by_second(book_ts_ms, np.ones(len(book_ts_ms)),
                                         day_start_ms, n_rows)
    for w in flow_windows_s:
        cols[f"ofi_{w}s"] = ofi_1s if w == 1 else trailing_sum(ofi_1s, w)

    # flows from trades (+1 = buyer was the aggressor, -1 = seller was)
    sign = np.where(np.asarray(trade_is_buyer_maker, dtype=bool), -1.0, 1.0)
    qty = np.asarray(trade_qty, dtype=np.float64)
    tflow_1s = sum_by_second(trade_ts_ms, sign * qty, day_start_ms, n_rows)
    cols["tvol_1s"] = sum_by_second(trade_ts_ms, qty, day_start_ms, n_rows)
    for w in flow_windows_s:
        cols[f"tflow_{w}s"] = tflow_1s if w == 1 else trailing_sum(tflow_1s, w)

    return cols
