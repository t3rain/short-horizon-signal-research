"""Download Binance USD-M futures bookTicker + aggTrades (daily files), convert
them to compressed parquet, and sanity-check them.

Usage (start with ONE day, inspect it, then fetch the rest):
    python 01_fetch_binance.py download --start 2026-10-09 --end 2026-10-09
    python 01_fetch_binance.py inspect  --date 2026-10-09
    python 01_fetch_binance.py download --start 2026-10-03 --end 2026-10-09

Requires:  pip install -U polars
Output:    data/parquet/BTCUSDT-bookTicker-YYYY-MM-DD.parquet (and aggTrades)
"""
import argparse
import datetime as dt
import hashlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

BASE = "https://data.binance.vision/data/futures/um/daily"
KINDS = ("bookTicker", "aggTrades")
COLS = {
    "bookTicker": ["update_id", "best_bid_price", "best_bid_qty",
                   "best_ask_price", "best_ask_qty",
                   "transaction_time", "event_time"],
    "aggTrades": ["agg_trade_id", "price", "quantity", "first_trade_id",
                  "last_trade_id", "transaction_time", "is_buyer_maker"],
}


# download
def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def file_url(kind: str, symbol: str, day: dt.date) -> str:
    return f"{BASE}/{kind}/{symbol}/{symbol}-{kind}-{day.isoformat()}.zip"


def http_get(url: str, dest: Path) -> bool:
    """Stream url to dest. Returns False on 404, raises on other errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            # Read 1 MB at a time. The := operator assigns and tests, which means an 
            # empty bytes object will stop execution. Never holds the whole file in memory.
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:8.1f} / {total/1e6:.1f} MB", end="")  # progress counter with \r so line overwrites itself
            print()
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise

# hash by 1 MB chunks to compare with checksum
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def zip_to_parquet(zip_path: Path, kind: str, out_path: Path) -> None:
    """Extract the CSV from the zip, stream it into zstd parquet, delete the CSV."""
    csv_path = None
    try:
        with zipfile.ZipFile(zip_path) as z:
            name = z.namelist()[0]
            z.extract(name, zip_path.parent)
            csv_path = zip_path.parent / name
        with open(csv_path, "r") as f:
            first = f.readline().strip()
        has_header = not first[:1].isdigit()
        ncols = len(first.split(","))
        if ncols != len(COLS[kind]):
            raise ValueError(f"{kind}: expected {len(COLS[kind])} columns, "
                             f"found {ncols} ({first!r})")
        # polars lazy scan, new_columns renames the columns
        # Look at many rows before fixing column types, so a size like "5" early
        # in the file is not locked in as an integer when "5.123" appears later.
        lf = pl.scan_csv(csv_path, has_header=has_header, new_columns=COLS[kind],
                         infer_schema_length=100_000)
        # streams the CSV in chunks to write compressed parquet
        # uses zstandard compression algorithm 
        lf.sink_parquet(out_path, compression="zstd")   # calling collect here would load the entire CSV into RAM
    finally:
        if csv_path is not None and csv_path.exists():
            csv_path.unlink()


def cmd_download(args) -> None:
    raw = Path(args.root) / "raw"
    pq = Path(args.root) / "parquet"
    raw.mkdir(parents=True, exist_ok=True)
    pq.mkdir(parents=True, exist_ok=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    for day in daterange(start, end):
        for kind in KINDS:
            stem = f"{args.symbol}-{kind}-{day.isoformat()}"
            out = pq / f"{stem}.parquet"
            if out.exists():
                print(f"[skip] {out.name} already exists")
                continue
            zip_path = raw / f"{stem}.zip"
            chk_path = raw / f"{stem}.zip.CHECKSUM"
            url = file_url(kind, args.symbol, day)
            print(f"[get ] {url}")
            if not http_get(url, zip_path):
                print(f"[miss] not published (404): {stem}")
                continue
            if http_get(url + ".CHECKSUM", chk_path):
                # pull the hash out of the checksum file
                expected = chk_path.read_text().split()[0].lower()
                actual = sha256(zip_path)
                if expected != actual:
                    print(f"[FAIL] checksum mismatch for {stem}; deleting")
                    zip_path.unlink()
                    continue
                print("[ ok ] checksum verified")
            else:
                print("[warn] no CHECKSUM file; skipping verification")
            t0 = time.time()
            zip_to_parquet(zip_path, kind, out)
            print(f"[conv] {out.name}  {out.stat().st_size/1e6:.1f} MB "
                  f"({time.time()-t0:.0f}s)")
            if not args.keep_zip:
                zip_path.unlink()
                chk_path.unlink(missing_ok=True)


# fetch timestamp unit 
def ts_divisor(lf: pl.LazyFrame) -> tuple[str, int]:
    first = lf.select(pl.col("transaction_time").first()).collect().item()
    digits = len(str(int(first)))
    if digits == 13:
        return "milliseconds", 1_000
    if digits == 16:
        return "microseconds", 1_000_000
    return f"UNKNOWN ({digits} digits)", 1_000


def inspect_book(path: Path) -> None:
    lf = pl.scan_parquet(path)
    n = lf.select(pl.len()).collect().item()
    unit, div = ts_divisor(lf)
    print(f"\n=== {path.name} ===")
    print(f"rows: {n:,}   timestamp unit: {unit}")
    print(lf.head(5).collect())

    # group by transaction times into a new group "sec", with a len for each group
    # and then collects these as a series in per_sec
    # MOTE: only seconds with an update will be present in per_sec
    per_sec = (lf.group_by((pl.col("transaction_time") // div).alias("sec"))
                 .len().collect()["len"])
    print(f"\nupdates per second: median {per_sec.median():.0f}, "
          f"mean {per_sec.mean():.1f}, p99 {per_sec.quantile(0.99):.0f}, "
          f"max {per_sec.max()}")
    print(f"seconds with at least one update: {per_sec.len():,} of 86,400")

    crossed = lf.filter(pl.col("best_bid_price") >= pl.col("best_ask_price")) \
                .select(pl.len()).collect().item()
    zero = lf.filter((pl.col("best_bid_qty") <= 0) | (pl.col("best_ask_qty") <= 0)) \
             .select(pl.len()).collect().item()
    print(f"\ncrossed/locked books (bid >= ask): {crossed:,}")
    print(f"rows with a non-positive best size: {zero:,}")

    # finds diff between consecutive rows 
    # finds out of order rows, if any, and prints
    dts = lf.select(pl.col("transaction_time").diff().alias("dt")).collect()["dt"]
    to_s = div
    print(f"out-of-order transaction_time rows: {(dts < 0).sum():,}")
    print(f"largest gap between updates: {dts.max() / to_s:.2f} s")

    lag = lf.select((pl.col("event_time") - pl.col("transaction_time")).alias("lag")) \
            .collect()["lag"] / (div / 1_000)
    print(f"event_time - transaction_time (ms): median {lag.median():.1f}, "
          f"p99 {lag.quantile(0.99):.1f}, max {lag.max():.1f}")

    
    spread = (lf.group_by((pl.col("best_ask_price") - pl.col("best_bid_price"))
                      .round(4).alias("spread"))
            .len()
            .sort("len", descending=True)
            .head(5)
            .collect())
    print("\nmost common spreads (price units):")
    print(spread)


def inspect_trades(path: Path) -> None:
    lf = pl.scan_parquet(path)
    n = lf.select(pl.len()).collect().item()
    unit, div = ts_divisor(lf)
    print(f"\n=== {path.name} ===")
    print(f"rows: {n:,}   timestamp unit: {unit}")
    print(lf.head(5).collect())
    per_sec = (lf.group_by((pl.col("transaction_time") // div).alias("sec"))
                 .len().collect()["len"])
    print(f"\naggTrades per second: median {per_sec.median():.0f}, "
          f"p99 {per_sec.quantile(0.99):.0f}, max {per_sec.max()}")
    share = lf.select(pl.col("is_buyer_maker").cast(pl.Boolean).mean()).collect().item()
    print(f"share with is_buyer_maker = True: {share:.3f}")
    print("(is_buyer_maker=True means the BUYER was resting, so the aggressor "
          "was a SELLER: sign = -1)")
    print(lf.select(pl.col("price").min().alias("min_price"),
                    pl.col("price").max().alias("max_price"),
                    pl.col("quantity").sum().alias("total_qty")).collect())


def cmd_inspect(args) -> None:
    pq = Path(args.root) / "parquet"
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    stem = f"{args.symbol}-{{}}-{args.date}.parquet"
    book = pq / stem.format("bookTicker")
    trades = pq / stem.format("aggTrades")
    for p, fn in ((book, inspect_book), (trades, inspect_trades)):
        if p.exists():
            fn(p)
        else:
            print(f"missing: {p} (run the download step first)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default="BTCUSDT")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", required=True, help="YYYY-MM-DD")
    d.add_argument("--end", required=True, help="YYYY-MM-DD")
    d.add_argument("--keep-zip", action="store_true")
    d.set_defaults(fn=cmd_download)
    i = sub.add_parser("inspect")
    i.add_argument("--date", required=True, help="YYYY-MM-DD")
    i.set_defaults(fn=cmd_inspect)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())