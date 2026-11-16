"""Does the signal pay for its trading costs? Deciles, edge by threshold, break-even cost, net PnL.

    python scripts/04_costs_and_pnl.py                          # ridge predictions, 10 s horizon
    python scripts/04_costs_and_pnl.py --taker-fee-bps 4.5      # e.g. with a BNB fee discount

Reads   <wf-dir>/preds_ridge.npz     out-of-sample predictions (from 03_run_walkforward.py)
        data/features                mid, bid_px, ask_px of the test days and of the day before
Writes  <wf-dir>/pnl_ridge_h<h>.csv   and   deciles_ridge_h<h>.csv

What it does (one horizon h, latency L as in the labels; every trade is 1 unit of notional):
 1. DECILES   Rank each day's rows by prediction, average the realised return in each tenth.
              A real signal rises from bin 1 to bin 10. Shows the size of the edge in bps.
 2. EDGE      Act only when |prediction| is in the top fraction q of recent predictions. The
              level is the q-quantile of |prediction| over the previous few days, so it is known
              in real time. Mean of sign(prediction) x return over those rows (overlaps allowed).
 3. TRADES    Same rule, but one position at a time (no new trade until the last one is closed).
              Gross PnL per trade, then net of costs:
                 taker  = 2 x taker fee + the spread crossed on entry and exit
                 maker  = 2 x maker fee, assuming EVERY order fills at the touch with no adverse
                          selection. An upper bound that this 1-second data cannot test.
              The gross edge per trade is also the break-even round-trip cost.
 4. REGIMES   The same at the primary q, split by the previous day's volatility (known in advance).
 5. SELECTION q is chosen on the first half of the test days and judged on the second half.
Fee defaults are Binance USD-M VIP 0 as quoted on its help page (maker 0.02%, taker 0.05%).
That page calls them hypothetical examples, so check the fee table for the tier you would use.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from signal_research import config
from signal_research.dataset import find_days, load_columns
from signal_research.labels import make_labels
from signal_research.metrics import spearman_ic, summarize_daily_ic
from signal_research.trading import (
    decile_means, gross_bps, pick_trades, roundtrip_spread_bps, trailing_threshold,
)

DAYS_PER_YEAR = 365                      # crypto trades every day
MODEL = "ridge"                          # the model whose predictions are costed


def t_stat(x: np.ndarray) -> float:
    return summarize_daily_ic(x)["t_stat"]


def trades_for_day(pred, label, spread_rt, thr, hold):
    """(gross bps, spread cost bps) of each trade on one day; empty if there is no threshold."""
    if not np.isfinite(thr):
        return np.array([]), np.array([])
    idx = pick_trades(pred, thr, hold)
    idx = idx[np.isfinite(label[idx]) & np.isfinite(spread_rt[idx])]
    return gross_bps(pred, label, idx), spread_rt[idx]


def daily_pnl(trades: list[tuple[np.ndarray, np.ndarray]], scenario: str, taker: float, maker: float):
    """Sum of net bps per day for one cost scenario (days without trades count as 0)."""
    out = []
    for gross, sp in trades:
        cost = {"zero": 0.0, "maker": 2 * maker, "taker": 2 * taker + sp}[scenario]
        out.append(float((gross - cost).sum()))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--wf-dir", default=None, help="default <root>/walkforward")
    ap.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON_S)
    ap.add_argument("--latency", type=int, default=config.LATENCY_S,
                    help="must equal the --latency of the run that made the predictions")
    ap.add_argument("--taker-fee-bps", type=float, default=5.0, help="per side")
    ap.add_argument("--maker-fee-bps", type=float, default=2.0, help="per side")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001])
    ap.add_argument("--window", type=int, default=5, help="previous days used for the threshold")
    ap.add_argument("--q-primary", type=float, default=0.01)
    args = ap.parse_args()

    root = Path(args.root)
    wf_dir = Path(args.wf_dir) if args.wf_dir else root / "walkforward"
    z = np.load(wf_dir / f"preds_{MODEL}.npz")
    test_dates = [str(d) for d in z["days"]]
    horizons = [int(h) for h in z["horizons"]]
    if args.horizon not in horizons:
        raise SystemExit(f"horizon {args.horizon} s is not in {horizons}")
    hi, h, L = horizons.index(args.horizon), args.horizon, args.latency
    D = len(test_dates)

    # features of the test days plus the day before (for the trailing volatility)
    all_days = find_days(root, args.symbol)
    all_dates = [d for d, _ in all_days]
    first = all_dates.index(test_dates[0])
    need = all_days[max(0, first - 1): first + D]
    cols = load_columns(need, ["mid", "bid_px", "ask_px"])
    off = len(need) - D                                          # 1 if a day before exists, else 0
    pred, label, spread_rt, prev_vol = [], [], [], []
    for k in range(D):
        c = cols[off + k]
        pred.append(z["preds"][k][:, hi].astype(np.float64))
        label.append(make_labels(c["mid"], (h,), L)[h])
        spread_rt.append(roundtrip_spread_bps(c["bid_px"], c["ask_px"], L, h))
        if off:
            r = np.diff(np.log(cols[off + k - 1]["mid"]))
            prev_vol.append(float(np.std(r[np.isfinite(r)]) * 1e4))
        else:
            prev_vol.append(np.nan)
    prev_vol = np.array(prev_vol)
    abs_pred = [np.abs(p) for p in pred]
    taker, maker = args.taker_fee_bps, args.maker_fee_bps
    print(f"{MODEL}: {D} test days {test_dates[0]} .. {test_dates[-1]}, horizon {h} s, latency {L} s")
    print(f"costs per side: taker {taker} bps + half the spread, maker {maker} bps "
          f"(round trip taker about {2 * taker:.0f} bps)")
    pd.set_option("display.width", 200)

    # 1. deciles -------------------------------------------------------------------------
    dec = np.array([decile_means(pred[k], label[k]) for k in range(D)])      # (D, 10)
    dec_mean = np.nanmean(dec, axis=0)
    spread_days = dec[:, -1] - dec[:, 0]
    dec_df = pd.DataFrame({"decile": np.arange(1, 11), "mean_return_bps": dec_mean})
    dec_df.to_csv(wf_dir / f"deciles_{MODEL}_h{h}.csv", index=False)
    print(f"\n1. DECILES: mean realised {h}-s return (bps) by prediction tenth, averaged over days")
    print(dec_df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"top minus bottom tenth: {np.nanmean(spread_days):+.3f} bps  (t across days = {t_stat(spread_days):+.1f})")

    # 2. edge per signal --------------------------------------------------------------------
    thr = {q: [trailing_threshold(abs_pred, k, q, args.window) for k in range(D)] for q in args.quantiles}
    rows, share_by_q = [], {}
    for q in args.quantiles:
        per_day, counts = [], []
        for k in range(D):
            sel = abs_pred[k] > thr[q][k]                                   # NaN threshold -> nothing
            ok = sel & np.isfinite(label[k])
            counts.append(int(ok.sum()))
            per_day.append(float(np.mean(np.sign(pred[k][ok]) * label[k][ok])) if ok.any() else np.nan)
        per_day = np.array(per_day)
        share_by_q[q] = float(np.mean(counts[1:])) / len(label[0])       # share of rows that cleared it
        rows.append({"top_fraction": q, "signal_share": share_by_q[q], "signals_per_day": np.mean(counts[1:]),
                     "edge_bps_per_signal": np.nanmean(per_day), "t_across_days": t_stat(per_day)})
    print(f"\n2. EDGE per signal (rows where |prediction| clears the threshold; overlapping allowed)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    print("(top_fraction is the nominal level used to set the threshold from the previous days. signal_share is the\n"
          " share of rows that actually cleared it today; it differs when the market regime, and so the scale of\n"
          " the predictions, changes.)")

    # 3. non-overlapping trades and costs ----------------------------------------------------
    table, per_q = [], {}
    for q in args.quantiles:
        tr = [trades_for_day(pred[k], label[k], spread_rt[k], thr[q][k], hold=h) for k in range(D)]
        per_q[q] = {s: daily_pnl(tr, s, taker, maker) for s in ("zero", "maker", "taker")}
        n = np.array([len(g) for g, _ in tr[1:]])
        allg = np.concatenate([g for g, _ in tr[1:]]) if n.sum() else np.array([])
        alls = np.concatenate([s for _, s in tr[1:]]) if n.sum() else np.array([])
        gross = float(allg.mean()) if len(allg) else np.nan
        table.append({"top_fraction": q, "signal_share": share_by_q[q], "trades_per_day": n.mean(),
                      "gross_bps_per_trade": gross,
                      "maker_net_per_trade": gross - 2 * maker,
                      "taker_net_per_trade": gross - 2 * taker - (float(alls.mean()) if len(alls) else np.nan),
                      "daily_gross_pnl_bps": per_q[q]["zero"][1:].mean(),
                      "daily_maker_pnl_bps": per_q[q]["maker"][1:].mean(),
                      "daily_taker_pnl_bps": per_q[q]["taker"][1:].mean(),
                      "t_gross": t_stat(per_q[q]["zero"][1:])})
    tab = pd.DataFrame(table)
    tab.to_csv(wf_dir / f"pnl_{MODEL}_h{h}.csv", index=False)
    print("\n3. TRADES, one position at a time. 'gross per trade' = break-even round-trip cost (bps)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))
    best = tab.loc[tab.gross_bps_per_trade.idxmax()]
    print(f"\nbest gross edge: {best.gross_bps_per_trade:.2f} bps per trade at top fraction "
          f"{best.top_fraction:g}; a taker pays about {2 * taker:.0f} bps plus spread, "
          f"a maker (if always filled, which is optimistic) {2 * maker:.0f} bps")

    # 4. regimes -------------------------------------------------------------------------------
    q = args.q_primary
    if q not in thr:
        thr[q] = [trailing_threshold(abs_pred, k, q, args.window) for k in range(D)]
    ks = np.arange(1, D)
    ks = ks[np.isfinite(prev_vol[ks])]
    if len(ks) >= 6:
        edges = np.quantile(prev_vol[ks], [1 / 3, 2 / 3])
        bucket = np.digitize(prev_vol[ks], edges)
        reg = []
        for b, name in enumerate(("low", "middle", "high")):
            kb = ks[bucket == b]
            tr = [trades_for_day(pred[k], label[k], spread_rt[k], thr[q][k], h) for k in kb]
            g = np.concatenate([x for x, _ in tr]) if tr else np.array([])
            reg.append({"prev_day_vol": name, "days": len(kb),
                        "vol_bps": prev_vol[kb].mean(),
                        "ic": np.nanmean([spearman_ic(pred[k], label[k]) for k in kb]),
                        "label_std_bps": np.nanmean([np.nanstd(label[k]) for k in kb]),
                        "trades_per_day": len(g) / max(len(kb), 1),
                        "gross_bps_per_trade": float(g.mean()) if len(g) else np.nan})
        print(f"\n4. REGIMES at top fraction {q:g}, split by the previous day's volatility")
        print(pd.DataFrame(reg).to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    # 5. choose q on the first half, judge on the second ----------------------------------------
    half = (D - 1) // 2
    first_half, second_half = np.arange(1, 1 + half), np.arange(1 + half, D)
    print(f"\n5. SELECTION: top fraction chosen on the first {half} test days by mean daily PnL, "
          f"reported on the last {len(second_half)}")
    for scen in ("zero", "maker", "taker"):
        score = {q_: per_q[q_][scen][first_half].mean() for q_ in args.quantiles}
        qb = max(score, key=score.get)
        s2 = per_q[qb][scen][second_half]
        print(f"   {scen:<6} costs: chose {qb:g}; second-half daily PnL {s2.mean():+9.3f} bps "
              f"(t = {t_stat(s2):+.2f}, {len(s2)} days)")


if __name__ == "__main__":
    main()