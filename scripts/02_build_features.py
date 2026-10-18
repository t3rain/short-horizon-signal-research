"""Build the one-row-per-second feature table for a range of UTC days.

    python scripts/02_build_features.py --start 2024-01-01 --end 2024-03-30
    python scripts/02_build_features.py --start 2024-03-15 --end 2024-03-15   # one day + quick look

Reads   data/parquet/BTCUSDT-bookTicker-DATE.parquet and ...-aggTrades-DATE.parquet
Writes  data/features/BTCUSDT-features-DATE.parquet   (86,400 rows, no labels)

Days whose features file already exists are skipped (use --force to rebuild them; that
needs the raw files, so fetch them again first if they were deleted with --delete-raw).
A day that fails (unsorted timestamps, missing raw file) is reported at the end and the
batch carries on.

Labels are NOT saved: they depend on the latency and horizon settings, so they are
computed from the saved `mid` column when needed (labels.make_labels).

The quick look (sanity checks plus single-feature rank correlations) is printed only
when a single day is built. It is ONE day, in-sample: a smoke test, not a result.

Needs RAM for one day of book updates (roughly 2-3 GB for BTCUSDT).
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from signal_research import config
from signal_research.features import SECONDS_PER_DAY, build_features
from signal_research.labels import make_labels

BOOK_COLS = ["transaction_time", "best_bid_price", "best_bid_qty",
             "best_ask_price", "best_ask_qty"]
TRADE_COLS = ["transaction_time", "quantity", "is_buyer_maker"]
NOT_FEATURES = {"t_end_ms", "bid_px", "ask_px", "mid"}


def day_start_ms(date: str) -> int:
    d = dt.date.fromisoformat(date)
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def quick_look(df: pl.DataFrame, horizons: tuple[int, ...], latency_s: int) -> None:
    pdf = df.to_pandas()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    n_nan_state = int(pdf["mid"].isna().sum())
    print(f"\nrows: {len(pdf):,}   rows with no book state yet: {n_nan_state}")
    print(f"seconds with zero book updates: {int((pdf['n_updates_1s'] == 0).sum())}")

    # OFI over a window should line up with the mid change over the
    # SAME window. This is contemporaneous not a prediction. (Cont et al.)
    same = pdf[["ofi_1s", "ret_1s"]].dropna().corr(method="spearman").iloc[0, 1]
    print(f"\nsanity: Spearman(OFI_1s, mid change over the same second) = {same:.3f}")
    print("        (should be clearly positive; near zero would mean a bug)")

    # for each feature, find the spearman correlation over all horizons
    labels = make_labels(pdf["mid"].to_numpy(), horizons, latency_s)
    feats = [c for c in pdf.columns if c not in NOT_FEATURES]
    table = {}
    for h in horizons:
        y = pd.Series(labels[h], index=pdf.index)
        row = {}
        for c in feats:
            both = pd.concat([pdf[c], y], axis=1).dropna()
            row[c] = both.corr(method="spearman").iloc[0, 1] if len(both) > 100 else np.nan
        table[f"y_{h}s"] = row
    print(f"\nsingle-feature Spearman IC vs forward label (latency L = {latency_s} s), ONE day:")
    print(pd.DataFrame(table).round(4))
    print("\nlabel std (bps):", {f"{h}s": round(float(np.nanstd(labels[h])), 2) for h in horizons})


def read_sorted(path: Path, id_col: str, cols: list[str]) -> pl.DataFrame:
    """Read a raw day file, restoring the exchange's event order via its id column."""
    df = pl.read_parquet(path, columns=[id_col, *cols])
    if not df[id_col].is_sorted():
        print(f"  note: {path.name} is not in {id_col} order; sorting")
        if df[id_col].n_unique() != df.height:
            raise ValueError(f"{path.name}: duplicate {id_col} values")
        df = df.sort(id_col)
    return df.drop(id_col)


def build_day(date: str, root: Path, symbol: str, latency: int, delete_raw: bool, force: bool, verbose: bool) -> None:
    book_path = root / "parquet" / f"{symbol}-bookTicker-{date}.parquet"
    trade_path = root / "parquet" / f"{symbol}-aggTrades-{date}.parquet"
    out_path = root / "features" / f"{symbol}-features-{date}.parquet"

    if out_path.exists() and not force:
        print(f"[skip] {date}: features already built")
        return

    for p in (book_path, trade_path):
        if not p.exists():
            raise FileNotFoundError(f"missing raw file {p.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"reading {book_path.name} and {trade_path.name} ...")
    book = read_sorted(book_path, "update_id", BOOK_COLS)
    trades = read_sorted(trade_path, "agg_trade_id", TRADE_COLS)

    cols = build_features(
        book["transaction_time"].to_numpy(),
        book["best_bid_price"].to_numpy(), book["best_bid_qty"].to_numpy(),
        book["best_ask_price"].to_numpy(), book["best_ask_qty"].to_numpy(),
        trades["transaction_time"].to_numpy(), trades["quantity"].to_numpy(),
        trades["is_buyer_maker"].to_numpy(),
        day_start_ms=day_start_ms(date),
        tick_size=config.TICK_SIZE,
        n_rows=SECONDS_PER_DAY,
    )
    df = pl.DataFrame(cols)
    df.write_parquet(out_path, compression="zstd")
    print(f"saved {out_path}  ({df.height:,} rows x {df.width} columns)")

    if verbose:
        quick_look(df, config.HORIZONS_S, latency)

    if delete_raw:
        book_path.unlink()
        trade_path.unlink()
        print("deleted raw parquet files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--latency", type=int, default=config.LATENCY_S,
                    help="latency L in seconds used for the quick-look labels")
    ap.add_argument("--delete-raw", action="store_true",
                    help="delete the raw bookTicker/aggTrades parquet after a successful save")
    ap.add_argument("--force", action="store_true",
                    help="rebuild days whose features file already exists")
    args = ap.parse_args()
    
    root = Path(args.root)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if  end < start:
        raise SystemExit("--end is before start--")
    
    failed : list[tuple[str, str]] = []
    for k in range((end-start).days + 1):
        day = start + dt.timedelta(days=k)
        date = day.isoformat()
        try:
            build_day(date, root, args.symbol, args.latency, args.delete_raw, args.force, (start == end))
        except (ValueError, FileNotFoundError) as e:
            print(f"[FAIL] {date}: {e}")
            failed.append((date, str(e)))
    print(f"\n {len(failed)} failed days(s)")
    for date, err in failed:
        print(f"\n {date} : {err}")
    if failed:
        raise SystemExit(1)
    

if __name__ == "__main__":
    main()
