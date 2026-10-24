"""Loading the per-day feature files and turning them into model inputs X and labels Y.
"""

from pathlib import Path
import polars as pl  
import numpy as np
from .labels import make_labels

NOT_FEATURES = {"t_end_ms", "bid_px", "ask_px", "mid"}      # bookkeeping columns, not model inputs


def find_days(root: Path, symbol: str, start: str | None = None,
              end: str | None = None) -> list[tuple[str, Path]]:
    prefix = f"{symbol}-features-"
    found = []
    for p in sorted((root / "features").glob(f"{prefix}*.parquet")):
        date = p.stem[len(prefix):]
        if start and date < start:
            continue
        if end and date > end:
            continue
        found.append((date, p))
    return found

# columns = None reads every column
def load_columns(days: list[tuple[str, Path]], columns: list[str] | None = None,
                 progress: bool = True) -> list[dict[str, np.ndarray]]:
    out = []
    for i, (date, path) in enumerate(days, 1):
        if progress:
            print(f"\r  loading [{i}/{len(days)}] {date}", end="", flush=True)
        df = pl.read_parquet(path, columns=columns)
        out.append({c: df[c].to_numpy() for c in df.columns})
    if progress:
        print()
    return out

# gives feature names, X_days, Y_days
# X -> (n, p); Y -> (n, H)
def build_xy(cols_by_day: list[dict[str, np.ndarray]], horizons: list[int] | tuple[int, ...],
             latency_s: int) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    feature_cols = [c for c in cols_by_day[0] if c not in NOT_FEATURES]
    X_days, Y_days = [], []
    for cols in cols_by_day:
        X_days.append(np.column_stack([cols[c] for c in feature_cols]).astype(np.float32))
        labels = make_labels(cols["mid"], tuple(horizons), latency_s)
        Y_days.append(np.column_stack([labels[h] for h in horizons]).astype(np.float64))
    return feature_cols, X_days, Y_days