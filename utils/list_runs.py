"""Table of every walk-forward results folder under data/: how it was made and how it scored.

    python utils/list_runs.py                  # horizon 10 s
    python utils/list_runs.py --horizon 30

A folder counts as a run if it holds summary.csv. Settings (latency, select_by, purge) come
from run_config.json, which 03_run_walkforward.py writes into each output folder; folders made
before that show '?'. Imbalance is a raw feature, so its IC does not depend on rho, selection
or purge: it only changes with the latency, so it also tells old runs apart.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for d in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        sp = d / "summary.csv"
        if not sp.exists():
            continue                                       # raw/, parquet/, features/ ...
        s = pd.read_csv(sp)
        ic = s[s.horizon_s == args.horizon].set_index("signal")["mean_ic"]
        n = s[s.horizon_s == args.horizon].set_index("signal")["n_days"]
        cp = d / "run_config.json"
        cfg = json.loads(cp.read_text()) if cp.exists() else {}
        rows.append({"folder": d.name,
                     "modified": dt.datetime.fromtimestamp(sp.stat().st_mtime).strftime("%m-%d %H:%M"),
                     "latency": cfg.get("latency", "?"),
                     "select_by": cfg.get("select_by", "?"),
                     "purge_s": cfg.get("purge_s", "?"),
                     "n_days": int(n.get("imbalance", 0)),
                     "imbalance_ic": ic.get("imbalance", float("nan")),
                     "ridge_ic": ic.get("ridge", float("nan"))})
    if not rows:
        raise SystemExit(f"no folder with summary.csv under {args.root}")
    pd.set_option("display.width", 200)
    print(f"mean daily IC at horizon {args.horizon} s\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()