"""Is the row disorder in Binance's own file, or did OUR conversion cause it?
 
    python scripts/check_csv_order.py --date 2024-02-05                     # bookTicker
    python scripts/check_csv_order.py --date 2024-02-05 --kind aggTrades
    python scripts/check_csv_order.py --date 2024-02-03                     # a clean day, for contrast
 
What it does (pure Python for the CSV part, so no polars tricks can reorder anything):
  1. downloads the daily zip into data/check/ (checksum verified), unless already there
  2. streams the CSV inside the zip line by line, in file order, and counts how often
     the id column (update_id / agg_trade_id) and transaction_time go BACKWARDS
  3. if data/parquet/<same day>.parquet exists, compares the id at the same row positions
     in the CSV and in the parquet (first rows + a regular sample through the whole file)
 
Verdict:
  CSV itself out of order            -> Binance's file is built that way (not our bug)
  CSV in order, parquet not matching -> our CSV -> parquet conversion reordered rows
  both in order and matching         -> nothing wrong with this day
"""
import argparse
import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from itertools import chain
from pathlib import Path
 
BASE = "https://data.binance.vision/data/futures/um/daily"
ID_COL = {"bookTicker": "update_id", "aggTrades": "agg_trade_id"}
N_FIELDS = {"bookTicker": 7, "aggTrades": 7}
HEAD_ROWS = 1000      # compare this many leading rows with the parquet
STRIDE = 9973         # ...and every STRIDE-th row after that (a prime, to avoid patterns)
 
 
def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:8.1f} / {total/1e6:.1f} MB", end="")
            print()
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
 
 
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
 
 
def scan_zip(zip_path: Path, kind: str) -> dict:
    """Walk the CSV in file order. Returns counts plus the ids needed for the parquet check."""
    res = {"rows": 0, "id_back": 0, "t_back": 0, "worst_t_back": 0,
           "examples": [], "head": [], "ids_head": [], "ids_sample": []}
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        res["member"] = name
        with z.open(name) as raw:
            f = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            first = f.readline()
            has_header = not first[:1].isdigit()
            lines = f if has_header else chain([first], f)
            res["has_header"] = has_header
            prev_id = prev_t = None
            prev_line = ""
            for n, line in enumerate(lines):
                parts = line.split(",", 6)
                if len(parts) < N_FIELDS[kind]:
                    continue                       # blank / partial trailing line
                i = int(parts[0])
                t = int(parts[5])
                if n < 8:
                    res["head"].append((n, i, t))
                if n < HEAD_ROWS:
                    res["ids_head"].append(i)
                if n % STRIDE == 0:
                    res["ids_sample"].append((n, i))
                if prev_id is not None:
                    if i <= prev_id:
                        res["id_back"] += 1
                    if t < prev_t:
                        res["t_back"] += 1
                        res["worst_t_back"] = max(res["worst_t_back"], prev_t - t)
                        if len(res["examples"]) < 3:
                            res["examples"].append((n, prev_line.strip(), line.strip()))
                prev_id, prev_t, prev_line = i, t, line
                res["rows"] = n + 1
    return res
 
 
def compare_with_parquet(res: dict, pq_path: Path, kind: str) -> tuple[int, int, int]:
    """(rows in parquet, head mismatches, sample mismatches) - needs polars."""
    import numpy as np
    import polars as pl
    ids = pl.read_parquet(pq_path, columns=[ID_COL[kind]])[ID_COL[kind]].to_numpy()
    head = np.asarray(res["ids_head"])
    head_bad = int((ids[: len(head)] != head).sum())
    idx = np.asarray([n for n, _ in res["ids_sample"] if n < len(ids)])
    want = np.asarray([i for n, i in res["ids_sample"] if n < len(ids)])
    sample_bad = int((ids[idx] != want).sum())
    return len(ids), head_bad, sample_bad
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--kind", default="bookTicker", choices=list(ID_COL))
    ap.add_argument("--root", default="data")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--zip", help="use this local zip instead of downloading")
    args = ap.parse_args()
 
    stem = f"{args.symbol}-{args.kind}-{args.date}"
    if args.zip:
        zip_path = Path(args.zip)
    else:
        check_dir = Path(args.root) / "check"
        check_dir.mkdir(parents=True, exist_ok=True)
        zip_path = check_dir / f"{stem}.zip"
        if not zip_path.exists():
            url = f"{BASE}/{args.kind}/{args.symbol}/{stem}.zip"
            print(f"[get ] {url}")
            if not download(url, zip_path):
                raise SystemExit("404: that day is not published")
            chk = check_dir / f"{stem}.zip.CHECKSUM"
            if download(url + ".CHECKSUM", chk):
                ok = chk.read_text().split()[0].lower() == sha256(zip_path)
                print("[ ok ] checksum verified" if ok else "[FAIL] checksum mismatch")
                if not ok:
                    raise SystemExit(1)
 
    print(f"scanning {zip_path.name} in file order (pure Python, can take a minute) ...")
    res = scan_zip(zip_path, args.kind)
    print(f"\nCSV member: {res['member']}   header row: {res['has_header']}   rows: {res['rows']:,}")
    print("first rows (row, id, transaction_time):")
    for row in res["head"]:
        print("   ", row)
    print(f"\nCSV id goes backwards (or repeats): {res['id_back']:,}")
    print(f"CSV transaction_time goes backwards: {res['t_back']:,}"
          f"   (worst jump {res['worst_t_back']/1000:.1f} s if ms)")
    for n, a, b in res["examples"]:
        print(f"  first breaks, row {n}:\n     prev: {a}\n     this: {b}")
 
    csv_bad = res["id_back"] > 0 or res["t_back"] > 0
    pq_path = Path(args.root) / "parquet" / f"{stem}.parquet"
    pq_note = ""
    pq_bad = None
    if pq_path.exists():
        try:
            n_pq, head_bad, sample_bad = compare_with_parquet(res, pq_path, args.kind)
            pq_bad = head_bad > 0 or sample_bad > 0 or n_pq != res["rows"]
            print(f"\nparquet rows: {n_pq:,} (CSV {res['rows']:,})   "
                  f"id mismatches vs CSV at same row: head {head_bad}, sample {sample_bad}")
        except ImportError:
            pq_note = "(polars/numpy not importable, parquet comparison skipped)"
    else:
        pq_note = f"(no {pq_path.name}, parquet comparison skipped)"
    if pq_note:
        print("\n" + pq_note)
 
    print("\nVERDICT:")
    if csv_bad:
        print("  The CSV inside Binance's zip is itself out of order -> it is the DATA, not our code.")
        if pq_bad is False:
            print("  The parquet matches the CSV row for row, so the conversion kept the order faithfully.")
        elif pq_bad:
            print("  The parquet ALSO differs from the CSV order -> the conversion changed it further.")
    elif pq_bad:
        print("  The CSV is in order but the parquet is not the same sequence -> OUR conversion reordered rows.")
        print("  Fix fetch_binance.py (e.g. sink_parquet(..., maintain_order=True) or collect then write).")
    elif pq_bad is None:
        print("  The CSV is in order. Run again with the parquet present to check the conversion.")
    else:
        print("  CSV in order and parquet matches it: nothing wrong with this day.")
 
 
if __name__ == "__main__":
    main()
 