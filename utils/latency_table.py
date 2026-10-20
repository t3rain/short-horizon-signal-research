"""IC-versus-latency table from the saved walk-forward summaries.

    python utils/latency_table.py                  # horizon 10 s, latencies 0 1 2 5 10
    python utils/latency_table.py --horizon 30

Reads data/walkforward/summary.csv for L = 1 (the default run) and data/wf_L<L>/summary.csv for
the other latencies (written by:  --latency L --out data\\wf_L<L>).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--latencies", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for L in sorted(args.latencies):
        path = root / ("walkforward" if L == 1 else f"wf_L{L}") / "summary.csv"
        if not path.exists():
            print(f"missing: {path}")
            continue
        s = pd.read_csv(path)
        s = s[s.horizon_s == args.horizon].set_index("signal")
        if s.empty:
            print(f"no horizon {args.horizon} s in {path}")
            continue
        rows.append({"L_s": L,
                     "imb_ic": s.loc["imbalance", "mean_ic"], "imb_t": s.loc["imbalance", "t_stat"],
                     "ridge_ic": s.loc["ridge", "mean_ic"], "ridge_t": s.loc["ridge", "t_stat"],
                     "ridge_minus_imb": s.loc["ridge", "mean_ic"] - s.loc["imbalance", "mean_ic"],
                     "n_days": int(s.loc["imbalance", "n_days"])})
    if not rows:
        raise SystemExit("nothing to show")
    t = pd.DataFrame(rows)
    # decay rate between consecutive latencies, assuming IC ~ exp(-k L)
    ic = t["imb_ic"].to_numpy()
    L = t["L_s"].to_numpy(dtype=float)
    k = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        if ic[i] > 0 and ic[i - 1] > 0:
            k[i] = np.log(ic[i - 1] / ic[i]) / (L[i] - L[i - 1])
    t["decay_per_s"] = k
    t["half_life_s"] = np.log(2) / t["decay_per_s"]
    pd.set_option("display.width", 200)
    print(f"\nimbalance and ridge, horizon {args.horizon} s, mean daily IC over out-of-sample days")
    print(t.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("\n(decay_per_s / half_life_s: between this row and the previous latency, assuming "
          "IC ~ exp(-k * L).)")


if __name__ == "__main__":
    main()