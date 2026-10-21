"""Is the higher IC on the re-sorted days (Feb 4 .. 21) a data artefact or a market regime?

    python utils/diagnose_repaired_days.py
    python utils/diagnose_repaired_days.py --horizon 1

Reads data/walkforward/daily_ic.csv (imbalance rows) and, for each test day, four cheap
summaries of the market from data/features: volatility, book-update rate, spread, depth.

1. BOUNDARIES  Daily IC for the days around the start and end of the repaired block.
               A market regime drifts; a data artefact STEPS exactly on the boundary dates.
2. PROFILE     Do repaired and clean days look different as markets (volatility, update rate,
               share of silent seconds, spread, depth)?
3. REGRESSION  daily IC ~ 1 + log(volatility) + log(update rate) + weekend + repaired.
               If `repaired` stays large after controlling for the market, the extra IC is not
               explained by volatility or activity. Plain OLS on ~60 days that are not
               independent: use the t-values as a rough guide, not as p-values.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from signal_research import config

BASELINE = "imbalance"
WINDOW = 4                                   # days shown on each side of a boundary


def day_profile(root: Path, symbol: str, date: str) -> dict[str, float]:
    df = pl.read_parquet(root / "features" / f"{symbol}-features-{date}.parquet",
                         columns=["mid", "n_updates_1s", "spread_ticks", "depth"])
    mid = df["mid"].to_numpy().astype(np.float64)
    upd = df["n_updates_1s"].to_numpy().astype(np.float64)
    bad = ~(np.isfinite(mid) & (mid > 0))                              # rows with no usable mid
    r = np.diff(np.log(np.where(bad, np.nan, mid)))
    r = r[np.isfinite(r)]                                              # only steps between two good rows
    return {"vol_bps": float(np.std(r) * 1e4),                         # std of 1-s mid returns
            "upd_per_s": float(np.nanmean(upd)),
            "quiet_share": float(np.nanmean(upd == 0)),                # seconds with no book update
            "spread": float(np.nanmean(df["spread_ticks"].to_numpy())),
            "depth": float(np.nanmean(df["depth"].to_numpy())),
            "bad_mid": int(bad.sum()),                                 # rows where mid is NaN / <= 0
            "first_bad": int(np.argmax(bad)) if bad.any() else -1}


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares coefficients and their standard errors (classical formula)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--wf-dir", default=None, help="default <root>/walkforward")
    ap.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON_S)
    ap.add_argument("--repaired-start", default="2024-02-04")
    ap.add_argument("--repaired-end", default="2024-02-21")
    args = ap.parse_args()

    root = Path(args.root)
    wf_dir = Path(args.wf_dir) if args.wf_dir else root / "walkforward"
    daily = pd.read_csv(wf_dir / "daily_ic.csv")
    ic = (daily[daily.signal == BASELINE].pivot(index="day", columns="horizon_s", values="ic")
          .sort_index())
    prof = []
    for i, d in enumerate(ic.index, 1):
        print(f"\r  loading [{i}/{len(ic)}] {d}", end="", flush=True)
        prof.append(day_profile(root, args.symbol, d))
    print()
    tab = pd.DataFrame(prof, index=ic.index)
    tab.insert(0, f"ic_{args.horizon}s", ic[args.horizon])
    tab["repaired"] = ((tab.index >= args.repaired_start) & (tab.index <= args.repaired_end)).astype(int)
    tab["weekend"] = (pd.to_datetime(tab.index).dayofweek >= 5).astype(int)
    out = wf_dir / "repaired_diagnostic.csv"
    tab.to_csv(out)
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:9.4f}"

    rep_pos = np.flatnonzero(tab.repaired.to_numpy())
    if len(rep_pos) == 0:
        raise SystemExit("no repaired days among the test days; check --repaired-start/--repaired-end")
    s, e = rep_pos[0], rep_pos[-1]
    show = sorted(set(range(max(s - WINDOW, 0), s + WINDOW)) | set(range(max(e - WINDOW + 1, 0),
                                                                          min(e + WINDOW + 1, len(tab)))))
    print(f"\n1. BOUNDARIES: imbalance daily IC around the repaired block "
          f"(rows {s} and {e} are its first and last day; full table saved to {out})")
    print(tab.iloc[show].drop(columns=["bad_mid", "first_bad"]).to_string(float_format=fmt))
    bad_days = tab[tab.bad_mid > 0]
    if len(bad_days):
        print("\nDays whose mid has unusable rows (NaN or <= 0); first_bad is the first row index:")
        print(bad_days[["bad_mid", "first_bad", "repaired"]].to_string())

    print("\n2. PROFILE: mean of each daily summary, repaired vs clean")
    g = tab.drop(columns=["bad_mid", "first_bad"]).groupby("repaired").mean().T
    g.columns = ["clean", "repaired"]
    g["ratio"] = g.repaired / g.clean
    print(g.to_string(float_format=fmt))

    y = tab[f"ic_{args.horizon}s"].to_numpy()
    X = np.column_stack([np.ones(len(tab)), np.log(tab.vol_bps), np.log(tab.upd_per_s),
                         tab.weekend, tab.repaired])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if not ok.all():
        print(f"\n(dropping {int((~ok).sum())} day(s) with a missing value from the regression)")
    y, X = y[ok], X[ok]
    beta, se = ols(y, X)
    print(f"\n3. REGRESSION on {len(y)} days: IC({args.horizon} s) ~ 1 + log vol + log update rate "
          f"+ weekend + repaired")
    for name, b, s_ in zip(["intercept", "log vol_bps", "log upd_per_s", "weekend", "repaired"], beta, se):
        print(f"   {name:<14} {b:+9.4f}   (se {s_:.4f}, t {b / s_:+.2f})")
    print(f"   corr(IC, log vol) = {np.corrcoef(y, X[:, 1])[0, 1]:+.2f},  "
          f"corr(IC, log update rate) = {np.corrcoef(y, X[:, 2])[0, 1]:+.2f},  "
          f"corr(log vol, log update rate) = {np.corrcoef(X[:, 1], X[:, 2])[0, 1]:+.2f}")


if __name__ == "__main__":
    main()