"""Walk-forward evaluation of ridge against single-feature baselines.

    python scripts/03_run_walkforward.py                       # every features file found
    python scripts/03_run_walkforward.py --start 2024-01-01 --end 2024-03-30
    python scripts/03_run_walkforward.py --max-days 40 --min-train-days 25   # quick smoke run

Reads   data/features/BTCUSDT-features-DATE.parquet            (from 02_build_features.py)
Writes  data/walkforward/daily_ic.csv      one IC per (signal, horizon, test day)
        data/walkforward/summary.csv       mean IC, t-stat, share of positive days
        data/walkforward/rho.csv           ridge strength chosen in each fold
        data/walkforward/preds_ridge.npz   out-of-sample predictions (for the PnL step)

How it works (see signal_research/walkforward.py): for each test day t, ridge is fitted on
days before t only (rho chosen on the last few of those days by validation squared error),
then scored on day t with IC. The
result is one IC per test day; we report the mean and its t-statistic across days.

BASELINE is declared here, before any results were looked at, so that "does ridge beat it?"
is a fair question. Single features are scored raw (rank-based, so no fitting at all).
"""
import argparse
import datetime as dt
from pathlib import Path
import json

import numpy as np
import pandas as pd
import polars as pl

from signal_research import config
from signal_research.dataset import find_days, load_columns
from signal_research.labels import make_labels
from signal_research.metrics import summarize_daily_ic
from signal_research.walkforward import WFConfig, walk_forward

BASELINE = "imbalance"                                     # declared in advance
TOP_FEATURES_SHOWN = 5


def find_days(root: Path, symbol: str, start: str | None, end: str | None) -> list[tuple[str, Path]]:
    prefix = f"{symbol}-features-"
    found = []
    for p in sorted((root / "features").glob(f"{prefix}*.parquet")):
        date = p.stem[len(prefix):]
        if start and date < start:
            continue
        if end and date > end:
            continue
        found.append((date, p))
    return found


def load_days(days: list[tuple[str, Path]]) -> tuple[list[str], dict[str, np.ndarray]]:
    out = []
    for i, (date, path) in enumerate(days, 1):
        print(f"\r  loading [{i}/{len(days)}] {date}", end="", flush=True)
        df = pl.read_parquet(path)
        out.append({c: df[c].to_numpy() for c in df.columns})
    print()
    return [d for d, _ in days], out


def warn_if_gaps(dates: list[str]) -> None:
    ds = [dt.date.fromisoformat(d) for d in dates]
    gaps = [(a, b) for a, b in zip(ds, ds[1:]) if (b - a).days != 1]
    if gaps:
        print("WARNING: days are not consecutive; walk-forward still runs in time order, "
              "but check if these gaps were intended:")
        for a, b in gaps:
            print(f"   {a} -> {b}")


# walk-forward harness: labels, walk-froward, tables, saving, summary display
def analyse(dates: list[str], cols_by_day: list[dict[str, np.ndarray]], cfg: WFConfig,
            horizons: tuple[int, ...], latency_s: int, out_dir: Path | None) -> pd.DataFrame:
    feature_cols = [c for c in cols_by_day[0] if c not in NOT_FEATURES]
    print(f"{len(dates)} days, {len(feature_cols)} features, horizons {horizons} s, "
          f"latency {latency_s} s")

    X_days, Y_days = [], []
    for cols in cols_by_day:
        X_days.append(np.column_stack([cols[c] for c in feature_cols]).astype(np.float32))
        labels = make_labels(cols["mid"], horizons, latency_s)
        Y_days.append(np.column_stack([labels[h] for h in horizons]).astype(np.float64))

    n_folds = len(dates) - cfg.min_train_days
    print(f"walk-forward: {n_folds} test days "
          f"({dates[cfg.min_train_days]} .. {dates[-1]}), "
          f"first fold trains on {cfg.min_train_days} days "
          f"({cfg.min_train_days - cfg.val_days} fit + {cfg.val_days} validation), "
          f"purge {cfg.purge_rows} s")
    res = walk_forward(X_days, Y_days, cfg)

    test_dates = [dates[i] for i in res.test_days]
    rows = []
    for k, day in enumerate(test_dates):
        for hi, h in enumerate(horizons):
            rows.append((day, h, "ridge", res.ic_model[k, hi])) # ridge IC
            for j, name in enumerate(feature_cols):
                rows.append((day, h, name, res.ic_feature[k, j, hi]))   # IC of all features
    daily = pd.DataFrame(rows, columns=["day", "horizon_s", "signal", "ic"])

    summ = []
    # group by horizon and signal - 4*16 groups
    for (h, sig), g in daily.groupby(["horizon_s", "signal"]):
        # ** unpacks a dict into the dict being built
        summ.append({"horizon_s": h, "signal": sig, **summarize_daily_ic(g["ic"].to_numpy())}) 
    summary = pd.DataFrame(summ)

    pd.set_option("display.width", 200)
    for h in horizons:
        s = summary[summary.horizon_s == h].sort_values("mean_ic", ascending=False)
        keep = pd.concat([s[s.signal == "ridge"],
                          s[s.signal != "ridge"].head(TOP_FEATURES_SHOWN)])
        if BASELINE in set(s.signal) and BASELINE not in set(keep.signal):
            keep = pd.concat([keep, s[s.signal == BASELINE]])
        print(f"\n=== horizon {h} s{'  (PRIMARY)' if h == config.PRIMARY_HORIZON_S else ''}: "
              f"daily IC over {n_folds} out-of-sample days ===")
        print(keep[["signal", "mean_ic", "std_ic", "t_stat", "pos_days", "n_days"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        d_r = daily[(daily.horizon_s == h) & (daily.signal == "ridge")].set_index("day")["ic"]
        d_b = daily[(daily.horizon_s == h) & (daily.signal == BASELINE)].set_index("day")["ic"]
        diff = summarize_daily_ic((d_r - d_b).to_numpy())
        print(f"ridge minus {BASELINE} (paired by day): mean {diff['mean_ic']:+.4f}, "
              f"t = {diff['t_stat']:+.2f}")

    print("\nCaution: the table shows the top single features out of "
          f"{len(feature_cols)} x {len(horizons)} horizons tried. The best of many looks good by "
          f"chance; the pre-declared baseline ({BASELINE}) and ridge are the fair comparison.")

    rho_df = pd.DataFrame(res.rho, columns=[f"rho_h{h}" for h in horizons])
    rho_df.insert(0, "day", test_dates)
    print("\nchosen rho per fold, counts:")
    for hi, h in enumerate(horizons):
        print(f"  h={h:>2}s  " + "  ".join(f"{r:g}: {int((res.rho[:, hi] == r).sum())}"
                                         for r in cfg.rhos))

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out_dir / "daily_ic.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        rho_df.to_csv(out_dir / "rho.csv", index=False)
        np.savez_compressed(out_dir / "preds_ridge.npz",
                            days=np.array(test_dates), horizons=np.array(horizons),
                            preds=np.stack(res.preds))
        print(f"\nsaved results to {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-days", type=int, help="use only the first N days (smoke runs)")
    ap.add_argument("--min-train-days", type=int, default=30)
    ap.add_argument("--val-days", type=int, default=5)
    ap.add_argument("--max-train-days", type=int, help="rolling window; default expanding")
    ap.add_argument("--purge-s", type=int, default=60)
    ap.add_argument("--select-by", choices=("mse", "ic"), default="ic",
                    help="criterion for choosing ridge strength on validation days")
    ap.add_argument("--clip-q", type=float, default=0.001,
                    help="winsorise features at [q, 1-q]; 0 disables")
    ap.add_argument("--latency", type=int, default=config.LATENCY_S)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(config.HORIZONS_S))
    ap.add_argument("--out", default=None, help="default <root>/walkforward")
    args = ap.parse_args()

    root = Path(args.root)
    days = find_days(root, args.symbol, args.start, args.end)
    if args.max_days:
        days = days[:args.max_days]
    if len(days) <= args.min_train_days:
        raise SystemExit(f"found {len(days)} feature days; need more than "
                         f"--min-train-days={args.min_train_days}")
    dates = [d for d, _ in days]
    cols_by_day = load_columns(days)
    warn_if_gaps(dates)
    cfg = WFConfig(min_train_days=args.min_train_days, val_days=args.val_days,
                   max_train_days=args.max_train_days, purge_rows=args.purge_s,
                   select_by=args.select_by,
                   clip_q=args.clip_q if args.clip_q > 0 else None)

    # default path is for runs with L=1s, IC selection -> data/walkforward
    out_dir = Path(args.out) if args.out else root / "walkforward"  
    analyse(dates, cols_by_day, cfg, tuple(args.horizons), args.latency, out_dir)

    # record folder metadata
    info = {**vars(args), "rhos": list(cfg.rhos), "first day": dates[0], "last day": dates[-1],
            "n_days": len(dates), "created": dt.datetime.now().isoformat(timespec= "seconds")}
    (out_dir / "run_config.json").write_text(json.dumps(info, indent=2))



if __name__ == "__main__":
    main()