"""Placebo and leakage checks on the walk-forward results (imbalance and ridge).

    python utils/placebo_checks.py
    python utils/placebo_checks.py --compare-dir data/wf_purge3600

Needs the output of 03_run_walkforward.py (default: data/walkforward, run with the same
--latency as given here, 1 s unless you changed it). Reads the saved out-of-sample ridge
predictions and daily ICs, plus `mid` and `imbalance` from data/features. Fits nothing.

1. REPRODUCE   Recompute the real IC from the saved predictions and freshly built labels and
               compare it with daily_ic.csv. Confirms this script's labels and the saved
               predictions line up with the run being checked. Must match to about 1e-4.
2. PLACEBO     Score the same signal against labels it cannot predict:
                 - other day: day i's signal against day i+s's labels (same time of day)
                 - rolled:    day i's signal against its own day's labels rotated by 1 h / 6 h
               Real signal -> clearly positive IC. Placebo -> about 0. A row is flagged CHECK
               when |t| > 3 and the placebo mean IC is more than 10% of the real one.
               This catches patterns shared by signal and label (e.g. time of day) and shows
               the size of IC noise under the null. It does NOT catch look-ahead in a feature
               or label leakage into training (see signal_research/placebo.py): those are
               covered by the perturbation tests and by the latency curve.
3. REPAIRED    Did the 18 days whose bookTicker file we re-sorted by update_id (2024-02-04 ..
   VS CLEAN    2024-02-21) behave like the others? Welch t-test on daily IC. Not conclusive on
               its own: the repaired days are also one calendar block, so a difference could
               be the market regime and not the repair. Equal means is reassuring.
4. COMPARE     (optional) Another run, e.g. with a much longer purge. Imbalance is a raw
               feature and must be identical; ridge should barely move (paired by day).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from signal_research import config
from signal_research.labels import make_labels
from signal_research.metrics import summarize_daily_ic
from signal_research.placebo import placebo_daily_ic, welch_t

BASELINE = "imbalance"
DAY_SHIFTS = (-7, -2, -1, 1, 2, 7)          # days
ROW_SHIFTS = (3600, 21600)                  # seconds: 1 hour, 6 hours
REPRODUCE_TOL = 1e-4                        # float32 predictions rank very slightly differently
T_FLAG = 3.0
SHARE_FLAG = 0.10


def load_days(root: Path, symbol: str, dates: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """mid and the baseline feature for each date (only these two columns are read)."""
    mids, imbs = [], []
    for i, d in enumerate(dates, 1):
        print(f"\r  loading [{i}/{len(dates)}] {d}", end="", flush=True)
        df = pl.read_parquet(root / "features" / f"{symbol}-features-{d}.parquet",
                             columns=["mid", BASELINE])
        mids.append(df["mid"].to_numpy())
        imbs.append(df[BASELINE].to_numpy())
    print()
    return mids, imbs


def reproduce_and_placebo(dates: list[str], horizons: list[int], preds: np.ndarray,
                          imbs: list[np.ndarray], mids: list[np.ndarray], latency_s: int,
                          daily: pd.DataFrame) -> None:
    labels = [make_labels(m, tuple(horizons), latency_s) for m in mids]
    rows, worst = [], 0.0
    for hi, h in enumerate(horizons):
        lab = [lb[h] for lb in labels]
        signals = {BASELINE: imbs, "ridge": [p[:, hi] for p in preds]}
        for name, sig in signals.items():
            real = placebo_daily_ic(sig, lab)
            saved = (daily[(daily.horizon_s == h) & (daily.signal == name)]
                     .set_index("day").loc[dates, "ic"].to_numpy())
            worst = max(worst, float(np.nanmax(np.abs(real - saved))))
            real_mean = float(np.nanmean(real))
            tests = [("real (aligned)", real)]
            tests += [(f"other day {s:+d}", placebo_daily_ic(sig, lab, day_shift=s))
                      for s in DAY_SHIFTS]
            tests += [(f"same day, rolled {m} s", placebo_daily_ic(sig, lab, row_shift=m))
                      for m in ROW_SHIFTS]
            for test, ic in tests:
                s = summarize_daily_ic(ic)
                flag = ""
                if test != "real (aligned)" and abs(s["t_stat"]) > T_FLAG \
                        and abs(s["mean_ic"]) > SHARE_FLAG * abs(real_mean):
                    flag = "CHECK"
                rows.append({"horizon_s": h, "signal": name, "test": test, "n_days": s["n_days"],
                             "mean_ic": s["mean_ic"], "std_ic": s["std_ic"],
                             "t_stat": s["t_stat"], "flag": flag})
    table = pd.DataFrame(rows)

    print(f"\n1. REPRODUCE: largest |recomputed - saved| daily IC = {worst:.2e}  "
          f"({'OK' if worst < REPRODUCE_TOL else 'MISMATCH: wrong --latency, or labels/preds differ'})")

    print("\n2. PLACEBO (a good signal: real row clearly positive, every other row about 0)")
    pd.set_option("display.width", 200)
    for h in horizons:
        print(f"\n--- horizon {h} s ---")
        print(table[table.horizon_s == h].drop(columns="horizon_s")
              .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    n_flag = int((table.flag == "CHECK").sum())
    print(f"\n{n_flag} placebo row(s) flagged CHECK" +
          (".  The placebo cannot see look-ahead in features; see the perturbation tests."
           if n_flag == 0 else ": look at these before trusting the headline IC."))


def repaired_vs_clean(daily: pd.DataFrame, start: str, end: str, horizons: list[int]) -> None:
    print(f"\n3. REPAIRED ({start} .. {end}) vs CLEAN test days, daily IC")
    rows = []
    for h in horizons:
        for name in (BASELINE, "ridge"):
            g = daily[(daily.horizon_s == h) & (daily.signal == name)]
            rep = g[(g.day >= start) & (g.day <= end)]["ic"].to_numpy()
            cln = g[(g.day < start) | (g.day > end)]["ic"].to_numpy()
            diff, t = welch_t(rep, cln)
            rows.append({"horizon_s": h, "signal": name, "n_rep": len(rep), "n_clean": len(cln),
                         "mean_rep": np.nanmean(rep) if len(rep) else np.nan,
                         "mean_clean": np.nanmean(cln) if len(cln) else np.nan,
                         "diff": diff, "welch_t": t})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("(differences with |t| < 2 are unremarkable; neighbouring days are not independent)")


def compare_runs(base: pd.DataFrame, other_dir: Path, horizons: list[int]) -> None:
    print(f"\n4. COMPARE with {other_dir}")
    other = pd.read_csv(other_dir / "daily_ic.csv")
    m = base.merge(other, on=["day", "horizon_s", "signal"], suffixes=("_base", "_other"))
    if m.empty:
        print("  no common (day, horizon, signal) rows")
        return
    raw = m[m.signal == BASELINE]
    print(f"  {BASELINE} (raw, nothing fitted): largest |difference| = "
          f"{(raw.ic_base - raw.ic_other).abs().max():.2e}  (must be 0)")
    for h in horizons:
        r = m[(m.horizon_s == h) & (m.signal == "ridge")].sort_values("day")
        d = summarize_daily_ic((r.ic_other - r.ic_base).to_numpy())
        print(f"  ridge, horizon {h:>2} s: base {r.ic_base.mean():.4f}, other {r.ic_other.mean():.4f}, "
              f"other - base (paired by day) {d['mean_ic']:+.4f}, t = {d['t_stat']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--wf-dir", default=None, help="default <root>/walkforward")
    ap.add_argument("--latency", type=int, default=config.LATENCY_S,
                    help="must equal the --latency of the walk-forward run being checked")
    ap.add_argument("--repaired-start", default="2024-02-04")
    ap.add_argument("--repaired-end", default="2024-02-21")
    ap.add_argument("--compare-dir", nargs="*", default=[],
                    help="other walk-forward output folder(s) to compare against")
    args = ap.parse_args()

    root = Path(args.root)
    wf_dir = Path(args.wf_dir) if args.wf_dir else root / "walkforward"
    z = np.load(wf_dir / "preds_ridge.npz")
    dates = [str(d) for d in z["days"]]
    horizons = [int(h) for h in z["horizons"]]
    preds = z["preds"]                                   # (n_test_days, rows, n_horizons)
    daily = pd.read_csv(wf_dir / "daily_ic.csv")
    print(f"{len(dates)} test days {dates[0]} .. {dates[-1]}, horizons {horizons} s, "
          f"latency {args.latency} s")

    mids, imbs = load_days(root, args.symbol, dates)
    reproduce_and_placebo(dates, horizons, preds, imbs, mids, args.latency, daily)
    repaired_vs_clean(daily, args.repaired_start, args.repaired_end, horizons)
    for other in args.compare_dir:
        compare_runs(daily, Path(other), horizons)


if __name__ == "__main__":
    main()