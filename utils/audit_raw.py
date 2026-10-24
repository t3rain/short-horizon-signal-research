"""Scan the raw parquet days for ordering / coverage problems.

    python utils/audit_raw.py                                  # every day found
    python utils/audit_raw.py --start 2024-01-01 --end 2024-03-30
    python utils/audit_raw.py --all --csv audit.csv            # show every day, save a CSV

For each (day, dataset) it reports:
    back_steps   how many times transaction_time goes BACKWARDS (row n < row n-1)
    worst_ms     the largest backwards jump, in milliseconds
    id_not_inc   how many times the id column (update_id / agg_trade_id) does not increase
    outside      rows whose timestamp is not inside that UTC day
    span_h       hours between the first and last timestamp (~24 for a full day)

Prints days with ordering problems only.
"""
import argparse
import datetime as dt
import re
from pathlib import Path

import polars as pl

ID_COL = {"bookTicker": "update_id", "aggTrades": "agg_trade_id"}
NAME = re.compile(r"^(?P<sym>[A-Z0-9]+)-(?P<kind>bookTicker|aggTrades)-(?P<date>\d{4}-\d{2}-\d{2})\.parquet$")


def day_bounds_ms(date: str) -> tuple[int, int]:
    d = dt.date.fromisoformat(date)
    start = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    return start, start + 86_400_000


def audit_file(path: Path, kind: str, date: str) -> dict:
    id_col = ID_COL[kind]
    df = pl.read_parquet(path, columns=[id_col, "transaction_time"])
    out = {"date": date, "kind": kind, "rows": df.height}
    if df.height < 2:
        out.update(back_steps=0, worst_ms=0.0, id_not_inc=0, outside=0, span_h=0.0)
        return out

    t = df["transaction_time"]
    digits = len(str(int(t[0])))
    to_ms = {13: 1.0, 16: 1e-3}.get(digits, 1.0)           # raw unit -> milliseconds

    step = t.diff()
    back = step.filter(step < 0)
    id_step = df[id_col].diff()
    lo, hi = day_bounds_ms(date)
    t_ms = t.cast(pl.Float64) * to_ms
    outside = int(((t_ms < lo) | (t_ms >= hi)).sum())

    out.update(
        back_steps=back.len(),
        worst_ms=float(-back.min() * to_ms) if back.len() else 0.0,
        id_not_inc=int((id_step <= 0).sum() or 0),
        outside=outside,
        span_h=float((t_ms.max() - t_ms.min()) / 3_600_000),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", help="YYYY-MM-DD (default: earliest file found)")
    ap.add_argument("--end", help="YYYY-MM-DD (default: latest file found)")
    ap.add_argument("--all", action="store_true", help="print every day, not only problem days")
    ap.add_argument("--csv", help="also write the full results to this CSV file")
    args = ap.parse_args()

    files = []
    for p in sorted((Path(args.root) / "parquet").glob("*.parquet")):
        m = NAME.match(p.name)
        if not m or m["sym"] != args.symbol:
            continue
        if args.start and m["date"] < args.start:
            continue
        if args.end and m["date"] > args.end:
            continue
        files.append((p, m["kind"], m["date"]))
    if not files:
        raise SystemExit("no raw parquet files found (they are deleted by --delete-raw)")

    rows = []
    for i, (p, kind, date) in enumerate(files, 1):
        print(f"\r[{i}/{len(files)}] {p.name}      ", end="", flush=True)
        rows.append(audit_file(p, kind, date))
    print()

    res = pl.DataFrame(rows).with_columns(
        problem=(pl.col("back_steps") > 0) | (pl.col("id_not_inc") > 0)
                | (pl.col("outside") > 0) | (pl.col("span_h") < 23.5)
    ).select("date", "kind", "rows", "back_steps", "worst_ms", "id_not_inc",
             "outside", "span_h", "problem").sort("date", "kind")

    pl.Config.set_tbl_rows(500)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    shown = res if args.all else res.filter(pl.col("problem"))
    print(shown if shown.height else "no problems found")
    n_bad = res.filter(pl.col("problem")).select(pl.col("date").n_unique()).item()
    print(f"\n{res.select(pl.col('date').n_unique()).item()} days scanned, {n_bad} with a problem")
    if args.csv:
        res.write_csv(args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()