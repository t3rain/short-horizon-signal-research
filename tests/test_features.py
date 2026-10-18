"""Tests for features.py.

1. Hand-made cases with answers worked out on paper.
2. A naive slow loop (written differently on purpose) must match the vectorised code.
3. The perturbation test: changing events at or after t must not change rows up to t.
"""
import numpy as np
import pytest

from signal_research import features as F

TICK = 1.0
N_ROWS = 20


# ------------------------------------------------------------------ small helpers
def ofi(prev, cur):
    """OFI of one update. prev/cur = (bid_px, bid_qty, ask_px, ask_qty)."""
    pb, pbq, pa, paq = prev
    cb, cbq, ca, caq = cur
    return F.ofi_per_event(
        F.to_ticks(np.array([pb, cb]), TICK), np.array([pbq, cbq], dtype=float),
        F.to_ticks(np.array([pa, ca]), TICK), np.array([paq, caq], dtype=float),
    )[1]


def random_times(rng, n, n_rows):
    """Sorted ms timestamps; ~25 % sit exactly on a second boundary to exercise tie handling."""
    t = rng.integers(0, n_rows * 1000, size=n)
    snap = rng.random(n) < 0.25
    t[snap] = (t[snap] // 1000) * 1000
    return np.sort(t).astype(np.int64)


def make_stream(rng, n_events=300, n_rows=N_ROWS):
    ts = random_times(rng, n_events, n_rows)
    bid = 100 + np.cumsum(rng.integers(-1, 2, size=n_events))
    ask = bid + rng.integers(1, 4, size=n_events)
    bq = rng.integers(1, 6, size=n_events).astype(float)
    aq = rng.integers(1, 6, size=n_events).astype(float)
    n_tr = 120
    tts = random_times(rng, n_tr, n_rows)
    tq = rng.integers(1, 5, size=n_tr).astype(float)
    tm = rng.random(n_tr) < 0.5
    return (ts, bid.astype(float), bq, ask.astype(float), aq), (tts, tq, tm)


def run(book, trades, n_rows=N_ROWS):
    ts, bp, bq, ap, aq = book
    tts, tq, tm = trades
    return F.build_features(ts, bp, bq, ap, aq, tts, tq, tm, day_start_ms=0,
                            tick_size=TICK, n_rows=n_rows,
                            flow_windows_s=(1, 5), ret_windows_s=(1, 5))


# ----------------------------------------------------------------- 1. hand-made cases
def test_ofi_each_case():
    # bid up to a new level with size 3 (ask unchanged): +3
    assert ofi((100, 5, 102, 4), (101, 3, 102, 4)) == 3
    # bid same price, size 5 -> 8: +3
    assert ofi((100, 5, 102, 4), (100, 8, 102, 4)) == 3
    # bid down: the old bid size 5 disappeared: -5
    assert ofi((100, 5, 102, 4), (99, 7, 102, 4)) == -5
    # ask improves (down) with size 2: -2
    assert ofi((100, 5, 102, 4), (100, 5, 101, 2)) == -2
    # ask same price, size 4 -> 1 (sellers pulled back): +3
    assert ofi((100, 5, 102, 4), (100, 5, 102, 1)) == 3
    # ask up: the old ask size 4 disappeared: +4
    assert ofi((100, 5, 102, 4), (100, 5, 103, 9)) == 4


def test_ofi_first_event_is_zero():
    e = F.ofi_per_event(np.array([100]), np.array([1.0]), np.array([101]), np.array([1.0]))
    assert e[0] == 0.0


def test_to_ticks_ignores_float_noise():
    t = F.to_ticks(np.array([0.1 + 0.2, 0.3]), 0.1)
    assert t[0] == t[1] == 3


def test_trailing_sum():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = F.trailing_sum(x, 3)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [6.0, 9.0, 12.0])
    np.testing.assert_allclose(F.trailing_sum(x, 1), x)


def test_state_forward_fill_and_boundaries():
    # events at 500 ms and 1500 ms; nothing afterwards
    book = (np.array([500, 1500]), np.array([100.0, 102.0]), np.array([1.0, 1.0]),
            np.array([101.0, 103.0]), np.array([1.0, 1.0]))
    trades = (np.array([], dtype=np.int64), np.array([]), np.array([], dtype=bool))
    f = run(book, trades, n_rows=4)
    # row 0 ends at 1000 ms -> uses the event at 500
    assert f["bid_px"][0] == 100.0
    # row 1 ends at 2000 ms -> uses the event at 1500
    assert f["bid_px"][1] == 102.0
    # rows 2, 3 have no new events: forward fill
    assert f["bid_px"][2] == f["bid_px"][3] == 102.0


def test_event_exactly_on_boundary_belongs_to_the_next_row():
    book = (np.array([500, 1000]), np.array([100.0, 105.0]), np.array([1.0, 1.0]),
            np.array([101.0, 106.0]), np.array([1.0, 1.0]))
    trades = (np.array([], dtype=np.int64), np.array([]), np.array([], dtype=bool))
    f = run(book, trades, n_rows=3)
    assert f["bid_px"][0] == 100.0          # ts == 1000 is NOT before t_0 = 1000
    assert f["bid_px"][1] == 105.0
    assert f["n_updates_1s"][0] == 1 and f["n_updates_1s"][1] == 1


def test_no_state_before_first_event_is_nan():
    book = (np.array([1500]), np.array([100.0]), np.array([1.0]),
            np.array([101.0]), np.array([1.0]))
    trades = (np.array([], dtype=np.int64), np.array([]), np.array([], dtype=bool))
    f = run(book, trades, n_rows=3)
    assert np.isnan(f["mid"][0]) and not np.isnan(f["mid"][1])


def test_microprice_and_imbalance():
    # bid 100 (size 1), ask 101 (size 3): thin bid, heavy ask -> sell pressure
    book = (np.array([100]), np.array([100.0]), np.array([1.0]),
            np.array([101.0]), np.array([3.0]))
    trades = (np.array([], dtype=np.int64), np.array([]), np.array([], dtype=bool))
    f = run(book, trades, n_rows=2)
    assert f["mid"][0] == 100.5
    assert f["imbalance"][0] == pytest.approx(-0.5)
    micro = (100.0 * 3 + 101.0 * 1) / 4          # 100.25, below the mid
    assert f["micro_dev_bps"][0] == pytest.approx((micro - 100.5) / 100.5 * 1e4)
    assert f["spread_ticks"][0] == 1.0


def test_trade_sign_convention():
    book = (np.array([0]), np.array([100.0]), np.array([1.0]),
            np.array([101.0]), np.array([1.0]))
    # buyer_maker=False -> buyer was the aggressor (+); True -> seller aggressor (-)
    trades = (np.array([100, 200, 1100]), np.array([2.0, 3.0, 4.0]),
              np.array([False, True, True]))
    f = run(book, trades, n_rows=3)
    assert f["tflow_1s"][0] == 2.0 - 3.0
    assert f["tflow_1s"][1] == -4.0
    assert f["tvol_1s"][0] == 5.0


# ----------------------------------------------------------- 2. naive reference check
def naive_event(prev, cur):
    pb, qb, pa, qa = prev
    cb, cqb, ca, cqa = cur
    e = 0.0
    if cb > pb:
        e += cqb
    elif cb == pb:
        e += cqb - qb
    else:
        e -= qb
    if ca < pa:
        e -= cqa
    elif ca == pa:
        e -= cqa - qa
    else:
        e += qa
    return e


def naive(book, trades, n_rows):
    ts, bp, bq, ap, aq = book
    tts, tq, tm = trades
    ofi_s, upd, tflow, tvol = (np.zeros(n_rows) for _ in range(4))
    for n in range(len(ts)):
        b = ts[n] // 1000
        upd[b] += 1
        if n > 0:
            ofi_s[b] += naive_event((bp[n - 1], bq[n - 1], ap[n - 1], aq[n - 1]),
                                    (bp[n], bq[n], ap[n], aq[n]))
    for k in range(len(tts)):
        b = tts[k] // 1000
        tflow[b] += (-1.0 if tm[k] else 1.0) * tq[k]
        tvol[b] += tq[k]
    out = {k: np.full(n_rows, np.nan) for k in
           ("bid_px", "ask_px", "imbalance", "micro_dev_bps", "spread_ticks")}
    for i in range(n_rows):
        last = None
        for n in range(len(ts)):
            if ts[n] < (i + 1) * 1000:
                last = n
        if last is None:
            continue
        b_, a_, bq_, aq_ = bp[last], ap[last], bq[last], aq[last]
        mid = (b_ + a_) / 2
        out["bid_px"][i], out["ask_px"][i] = b_, a_
        out["imbalance"][i] = (bq_ - aq_) / (bq_ + aq_)
        out["micro_dev_bps"][i] = ((b_ * aq_ + a_ * bq_) / (bq_ + aq_) - mid) / mid * 1e4
        out["spread_ticks"][i] = a_ - b_
    out["ofi_1s"], out["n_updates_1s"] = ofi_s, upd
    out["tflow_1s"], out["tvol_1s"] = tflow, tvol
    out["ofi_5s"] = np.array([ofi_s[i - 4: i + 1].sum() if i >= 4 else np.nan
                              for i in range(n_rows)])
    out["tflow_5s"] = np.array([tflow[i - 4: i + 1].sum() if i >= 4 else np.nan
                                for i in range(n_rows)])
    return out


@pytest.mark.parametrize("seed", range(10))
def test_vectorised_matches_naive_loops(seed):
    rng = np.random.default_rng(seed)
    book, trades = make_stream(rng)
    got, want = run(book, trades), naive(book, trades, N_ROWS)
    for name, expected in want.items():
        np.testing.assert_allclose(got[name], expected, rtol=1e-9, atol=1e-9,
                                   equal_nan=True, err_msg=name)


# ------------------------------------------------------- 3. perturbation (no look-ahead)
def _scramble_from(arrays, ts, cutoff, rng):
    """Replace every value of events with ts >= cutoff by random junk (timestamps kept)."""
    late = ts >= cutoff
    out = []
    for a in arrays:
        a = a.copy()
        if a.dtype == bool:
            a[late] = rng.random(late.sum()) < 0.5
        else:
            a[late] = rng.integers(1, 9, size=late.sum()).astype(a.dtype) * 3 + 50
        out.append(a)
    return out


@pytest.mark.parametrize("seed", range(10))
def test_events_at_or_after_t_never_change_rows_up_to_t(seed):
    rng = np.random.default_rng(100 + seed)
    book, trades = make_stream(rng)
    base = run(book, trades)
    i0 = 10
    cutoff = int(F.grid_end_times(0, N_ROWS)[i0])      # end time of row i0

    ts, bp, bq, ap, aq = book
    tts, tq, tm = trades
    bp2, bq2, ap2, aq2 = _scramble_from([bp, bq, ap, aq], ts, cutoff, rng)
    tq2, tm2 = _scramble_from([tq, tm], tts, cutoff, rng)
    changed = run((ts, bp2, bq2, ap2, aq2), (tts, tq2, tm2))

    for name in base:
        np.testing.assert_allclose(changed[name][: i0 + 1], base[name][: i0 + 1],
                                   equal_nan=True, err_msg=f"look-ahead in {name}")


def test_perturbation_test_has_teeth():
    """Positive control: changing events BEFORE t must show up, else the test above proves nothing."""
    rng = np.random.default_rng(7)
    book, trades = make_stream(rng, n_events=600)
    base = run(book, trades)
    ts, bp, bq, ap, aq = book
    i0 = 10
    lo, hi = (i0) * 1000, (i0 + 1) * 1000
    window = (ts >= lo) & (ts < hi)
    assert window.any()
    bq2 = bq.copy()
    bq2[window] += 7
    changed = run((ts, bp, bq2, ap, aq), trades)
    assert not np.allclose(changed["imbalance"][i0], base["imbalance"][i0])
