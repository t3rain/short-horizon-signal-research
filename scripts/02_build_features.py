"""Build the one-row-per-second feature table for one UTC day.

    python scripts/02_build_features.py --date 2024-03-15

Reads   data/parquet/BTCUSDT-bookTicker-DATE.parquet and ...-aggTrades-DATE.parquet
Writes  data/features/BTCUSDT-features-DATE.parquet   (86,400 rows, no labels)

Labels are NOT saved: they depend on the latency and horizon settings, so they are
computed from the saved `mid` column when needed (labels.make_labels). Then a
latency sweep never has to touch the raw data again.

After saving it prints a quick look: sanity checks plus single-feature rank
correlations with the labels. That table is ONE day and in-sample, so treat it as
a smoke test, not a result.

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--latency", type=int, default=config.LATENCY_S,
                    help="latency L in seconds used for the quick-look labels")
    ap.add_argument("--delete-raw", action="store_true",
                    help="delete the raw bookTicker/aggTrades parquet after a successful save")
    args = ap.parse_args()

    root = Path(args.root)
    book_path = root / "parquet" / f"{args.symbol}-bookTicker-{args.date}.parquet"
    trade_path = root / "parquet" / f"{args.symbol}-aggTrades-{args.date}.parquet"
    out_path = root / "features" / f"{args.symbol}-features-{args.date}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {book_path.name} and {trade_path.name} ...")
    book = pl.read_parquet(book_path, columns=BOOK_COLS)
    trades = pl.read_parquet(trade_path, columns=TRADE_COLS)

    cols = build_features(
        book["transaction_time"].to_numpy(),
        book["best_bid_price"].to_numpy(), book["best_bid_qty"].to_numpy(),
        book["best_ask_price"].to_numpy(), book["best_ask_qty"].to_numpy(),
        trades["transaction_time"].to_numpy(), trades["quantity"].to_numpy(),
        trades["is_buyer_maker"].to_numpy(),
        day_start_ms=day_start_ms(args.date),
        tick_size=config.TICK_SIZE,
        n_rows=SECONDS_PER_DAY,
    )
    df = pl.DataFrame(cols)
    df.write_parquet(out_path, compression="zstd")
    print(f"saved {out_path}  ({df.height:,} rows x {df.width} columns)")

    quick_look(df, config.HORIZONS_S, args.latency)

    if args.delete_raw:
        book_path.unlink()
        trade_path.unlink()
        print("deleted raw parquet files")


if __name__ == "__main__":
    main()
