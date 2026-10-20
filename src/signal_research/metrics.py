"""Evaluation metrics: the information coefficient (IC) and its summary over days.

IC = Spearman rank correlation between a signal and the realised forward return:

    rank both columns separately (smallest value -> rank 1, ties share the average rank),
    then take the ordinary Pearson correlation of the two rank columns:

        IC = sum (ra - mean(ra)) (rb - mean(rb))
             -----------------------------------------------
             sqrt( sum (ra - mean(ra))^2 * sum (rb - mean(rb))^2 )

Ranks make it insensitive to the scale of the signal and to a few huge return outliers.
"""

import numpy as np
from scipy.stats import rankdata

# Spearman correlation of two 1-D arrays
# if either less than 3 non-NaN rows remai or when either column is constant return NaN
# Walk-forward is NumPy only, calling pandas correlation method would involve wrapping these
# to df objects, which increases overhead; also makes the calculation here transparent
# tested against scipy.stats.spearmanr
def spearman_ic(signal: np.ndarray, label: np.ndarray) -> float:
    a = np.asarray(signal, dtype=np.float64)
    b = np.asarray(label, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = rankdata(a[ok])
    rb = rankdata(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    if den == 0.0:
        return float("nan")
    return float((ra @ rb) / den)


# Mean daily IC, its t-statistic across days, and the share of days with IC > 0.
# t = mean / (std/sqrt(N)) with sample std (ddof) of the N daily ICs
def summarize_daily_ic(daily_ic: np.ndarray) -> dict[str, float]:   
    x = np.asarray(daily_ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "t_stat": float("nan"),
                "pos_days": float("nan"), "n_days": 0}
    mean = float(x.mean())
    if n < 2:
        return {"mean_ic": mean, "std_ic": float("nan"), "t_stat": float("nan"),
                "pos_days": float((x > 0).mean()), "n_days": n}
    std = float(x.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return {"mean_ic": mean, "std_ic": std, "t_stat": float(t),
            "pos_days": float((x > 0).mean()), "n_days": n}