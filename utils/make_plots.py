"""Figures for the README, saved as PNG in docs/figures/.

    python utils/make_plots.py
    python utils/make_plots.py --horizon 10

Each figure is built from result files written by the earlier steps and is skipped, with a
message, if its input is missing:
  ic_over_time.png        data/walkforward/daily_ic.csv
  ic_vs_volatility.png    data/walkforward/repaired_diagnostic.csv        (utils/diagnose_repaired_days.py)
  ic_vs_latency.png       data/walkforward/summary.csv and data/wf_L<L>/summary.csv
  edge_vs_cost.png        data/walkforward/pnl_ridge_h<h>.csv           (scripts/04_costs_and_pnl.py)
Needs matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                   # draw to files, no window
import matplotlib.dates as mdates                       # noqa: E402
import matplotlib.ticker                                # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"    # categorical slots 1-3 of the reference palette
LINE_W, MARKER_PT = 1.5, 6                              # 2 px line, 8 px marker at 96 dpi


def new_fig(title: str, subtitle: str | None = None):
    fig, ax = plt.subplots(figsize=(7.6, 4.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    if subtitle:
        fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.80 if subtitle else 0.88, left=0.10, right=0.97, bottom=0.15)
    return fig, ax


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out / name}")


def missing(path: Path, what: str) -> bool:
    if path.exists():
        return False
    print(f"  skipped {what}: {path} not found")
    return True


def fig_ic_over_time(root: Path, out: Path, h: int) -> None:
    p = root / "walkforward" / "daily_ic.csv"
    if missing(p, "ic_over_time"):
        return
    base = pd.read_csv(p)

    def daily(df, sig):
        return df[(df.horizon_s == h) & (df.signal == sig)].sort_values("day").set_index("day")["ic"]

    series = {"imbalance (raw feature)": (daily(base, "imbalance"), BLUE),
              "ridge (15 features)": (daily(base, "ridge"), ORANGE)}
    note = "Spearman IC per test day: the raw imbalance feature and ridge on all 15 features."
    fig, ax = new_fig(f"Daily out-of-sample IC, {h}-second horizon", note)
    for label, (d, color) in series.items():
        ax.plot(pd.to_datetime(d.index), d.to_numpy(), color=color, lw=LINE_W, label=label,
                solid_capstyle="round")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("IC")
    loc = mdates.AutoDateLocator(maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper right", ncol=len(series))
    save(fig, out, "ic_over_time.png")


def fig_ic_vs_volatility(root: Path, out: Path, h: int) -> None:
    p = root / "walkforward" / "repaired_diagnostic.csv"
    if missing(p, "ic_vs_volatility"):
        return
    d = pd.read_csv(p)
    col = f"ic_{h}s"
    if col not in d.columns:
        print(f"  skipped ic_vs_volatility: no column {col} in {p}")
        return
    x, y = np.log(d.vol_bps.to_numpy()), d[col].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = new_fig(f"Imbalance IC against daily volatility ({h}-second horizon)",
                      f"One dot per test day; correlation of IC with log volatility = {r:+.2f}")
    for flag, label, color in ((0, "other days", BLUE), (1, "re-sorted days (Feb 4-21)", ORANGE)):
        m = d.repaired.to_numpy() == flag
        ax.scatter(d.vol_bps[m], d[col][m], s=MARKER_PT ** 2, color=color, edgecolors=SURFACE,
                   linewidths=1.2, label=label, zorder=3)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(np.exp(xs), a + b * xs, color=INK2, lw=LINE_W, label="linear fit", zorder=2)
    ax.set_xscale("log")
    lo, hi = d.vol_bps.min(), d.vol_bps.max()
    ax.set_xticks([t for t in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0) if lo * 0.85 <= t <= hi * 1.15])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("same-day volatility (std of 1-second mid returns, bps)")
    ax.set_ylabel("IC")
    fig.subplots_adjust(bottom=0.24)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3)
    save(fig, out, "ic_vs_volatility.png")


def fig_ic_vs_latency(root: Path, out: Path, h: int, latencies: list[int]) -> None:
    pts = []
    for L in sorted(latencies):
        p = root / ("walkforward" if L == 1 else f"wf_L{L}") / "summary.csv"
        if not p.exists():
            print(f"  (latency {L} s: {p} not found)")
            continue
        s = pd.read_csv(p)
        s = s[(s.horizon_s == h) & (s.signal == "imbalance")]
        if len(s):
            pts.append((L, float(s.mean_ic.iloc[0])))
    if len(pts) < 2:
        print("  skipped ic_vs_latency: need at least two latencies")
        return
    L, ic = zip(*pts)
    fig, ax = new_fig(f"Imbalance IC decays as order latency grows ({h}-second horizon)",
                      "Mean daily IC over the out-of-sample days; latency = delay before an order reaches the book")
    ax.plot(L, ic, color=BLUE, lw=LINE_W, marker="o", markersize=MARKER_PT, markeredgecolor=SURFACE,
            markeredgewidth=1.2, solid_capstyle="round")
    ax.annotate(f"{ic[0]:.3f}", (L[0], ic[0]), textcoords="offset points", xytext=(8, 6), color=INK2, fontsize=9)
    ax.annotate(f"{ic[-1]:.3f}", (L[-1], ic[-1]), textcoords="offset points", xytext=(-6, 10),
                ha="right", color=INK2, fontsize=9)
    ax.set_xticks(list(L))
    ax.set_ylim(bottom=min(0.0, min(ic) * 1.2))
    ax.set_xlabel("latency L (seconds)")
    ax.set_ylabel("mean daily IC")
    save(fig, out, "ic_vs_latency.png")


def fig_edge_vs_cost(root: Path, out: Path, h: int, taker: float, maker: float) -> None:
    p = root / "walkforward" / f"pnl_ridge_h{h}.csv"
    if missing(p, "edge_vs_cost"):
        return
    d = pd.read_csv(p)
    fig, ax = new_fig(f"Gross edge per trade against trading costs (ridge, {h}-second horizon)",
                      "Bars: mean gross return per trade when trading only the strongest predictions. "
                      "Lines: round-trip cost.")
    xs = np.arange(len(d))
    ax.bar(xs, d.gross_bps_per_trade, width=0.5, color=BLUE, zorder=3)
    best = int(d.gross_bps_per_trade.idxmax())
    ax.annotate(f"{d.gross_bps_per_trade[best]:.2f} bps", (best, d.gross_bps_per_trade[best]),
                textcoords="offset points", xytext=(0, 5), ha="center", color=INK2, fontsize=9)
    for y, color, label in ((2 * taker, ORANGE, f"taker fees only, round trip: {2 * taker:g} bps"),
                            (2 * maker, AQUA, f"maker fees only, round trip: {2 * maker:g} bps")):
        ax.axhline(y, color=color, lw=LINE_W, zorder=2)
        ax.text(len(d) - 0.55, y + 0.15, label, ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(xs)
    if "signal_share" in d.columns:
        ax.set_xticklabels([f"{q * 100:g}%\n(actual {a * 100:.1f}%)" for q, a in zip(d.top_fraction, d.signal_share)])
    else:
        ax.set_xticklabels([f"{q * 100:g}%" for q in d.top_fraction])
    ax.set_xlabel("threshold: top fraction of |prediction| over the previous 5 days (share of rows that cleared it)")
    ax.set_ylabel("bps per trade")
    ax.set_ylim(0, max(2 * taker, d.gross_bps_per_trade.max()) * 1.25)
    save(fig, out, "edge_vs_cost.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--latencies", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    ap.add_argument("--taker-fee-bps", type=float, default=5.0)
    ap.add_argument("--maker-fee-bps", type=float, default=2.0)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    fig_ic_over_time(root, out, args.horizon)
    fig_ic_vs_volatility(root, out, args.horizon)
    fig_ic_vs_latency(root, out, args.horizon, args.latencies)
    fig_edge_vs_cost(root, out, args.horizon, args.taker_fee_bps, args.maker_fee_bps)


if __name__ == "__main__":
    main()