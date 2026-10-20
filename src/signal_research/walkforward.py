"""Walk-forward evaluation: train on past days, predict the next day, roll forward.

Everything here is NumPy only so it can be tested on synthetic data.

Layout of one fold (days are whole UTC days, in time order):

    days:   0 ........................ t-V | t-V+1 ... t-1 |      t
            [             fit              ][      val     ][ test day ]
            [          train = fit + val: final            ]
    
  * The test day t is predicted by a model that only ever saw days < t.
  * We iterate over all ridge strengths we have to train on the fit set. This is then tested
    on the val set. The model with the least test error on val is chosen.
  * This model is trained again on the whole of fit+val set, and then scored on the immediate next test day.   
  * Since IC is blind to scale of predictions, heavy shrinkage can leave rankings intact and IC will not be able
  * to tell a calibrated model fro an over-shrunk one, and PnL requires predictions in bps.
  * Purge: the last `purge_rows` seconds of every fitting day are dropped.

Ridge is fitted from per-day sufficient statistics (n, sum x, sum xx', sum y, sum xy'),
which add up exactly across days. Any training window, and every rho, then costs one tiny
15x15 solve. rho is the penalty per training row:
    minimise  (1/N) * sum (y - ybar - z.theta)^2  +  rho * |theta|^2
"""
from dataclasses import dataclass

import numpy as np

from .metrics import spearman_ic

DEFAULT_RHOS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# config
@dataclass(frozen=True)
class WFConfig:
    min_train_days: int = 30            # first test day is day index min_train_days
    val_days: int = 5                   # last days of the training window, used to pick rho
    max_train_days: int | None = None   # None = expanding window; a number = rolling window
    rhos: tuple[float, ...] = DEFAULT_RHOS
    val_stride: int = 5                 # score validation stride for validation days; test days are scored on every sec
    clip_q: float | None = 0.001        # winsorise features at [q, 1-q] of the initial window
    purge_rows: int = 60                # seconds dropped at the end of every fitting day
    select_by: str = "ic"              # rho criterion on validation days: "mse" or "ic"

    def __post_init__(self) -> None:
        if self.val_days < 1 or self.min_train_days <= self.val_days:
            raise ValueError("need 1 <= val_days < min_train_days")
        if self.max_train_days is not None and self.max_train_days < self.min_train_days:
            raise ValueError("max_train_days must be >= min_train_days")
        if not self.rhos or any(r <= 0 for r in self.rhos):
            raise ValueError("rhos must be positive")
        if self.val_stride < 1 or self.purge_rows < 0:
            raise ValueError("bad val_stride / purge_rows")
        if self.select_by not in ("mse", "ic"):
            raise ValueError("select_by must be 'mse' or 'ic'")


@dataclass(frozen=True)
class Fold:
    test_day: int
    train: range        # every day the final model is fitted on
    fit: range          # train minus the validation days
    val: range          # last val_days of train


def fold_plan(n_days: int, cfg: WFConfig) -> list[Fold]:
    folds = []
    for t in range(cfg.min_train_days, n_days):
        start = 0 if cfg.max_train_days is None else max(0, t - cfg.max_train_days)
        folds.append(Fold(test_day=t,
                          train=range(start, t),
                          fit=range(start, t - cfg.val_days),
                          val=range(t - cfg.val_days, t)))
    return folds


# boolean mask for rows usable for fitting - purge
def fit_mask(X: np.ndarray, Y: np.ndarray, purge_rows: int) -> np.ndarray:
    m = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if purge_rows > 0:
        m[-purge_rows:] = False
    return m

# find q and 1-q quantiles
def fit_clip_bounds(X_days: list[np.ndarray], q: float, stride: int = 10
                    ) -> tuple[np.ndarray, np.ndarray]:
    sample = np.concatenate([np.asarray(X[::stride], dtype=np.float64) for X in X_days])
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) == 0:
        raise ValueError("no finite rows to estimate clip bounds from")
    return np.quantile(sample, q, axis=0), np.quantile(sample, 1.0 - q, axis=0)


# sufficient statistics
@dataclass
class Stats:
    n: float
    sx: np.ndarray       # (p,)    sum x
    sxx: np.ndarray      # (p, p)  sum x x'
    sy: np.ndarray       # (H,)    sum y
    sxy: np.ndarray      # (p, H)  sum x y'

    def __add__(self, other: "Stats") -> "Stats":
        return Stats(self.n + other.n, self.sx + other.sx, self.sxx + other.sxx,
                     self.sy + other.sy, self.sxy + other.sxy)


def day_stats(X: np.ndarray, Y: np.ndarray) -> Stats:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return Stats(float(len(X)), X.sum(axis=0), X.T @ X, Y.sum(axis=0), X.T @ Y)


def sum_stats(items: list[Stats]) -> Stats:
    total = items[0]
    for s in items[1:]:
        total = total + s
    return total


# ridge
@dataclass
class RidgeModel:
    mu: np.ndarray          # (p,)   training mean of each feature
    inv_sigma: np.ndarray   # (p,)   1 / training std (0 for a constant feature)
    ybar: np.ndarray        # (H,)   training mean of each label (the unpenalised intercept)
    theta: np.ndarray       # (p, H) coefficients on standardised features

    # predictions is (n, H)
    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu) * self.inv_sigma
        return Z @ self.theta + self.ybar

# Ridge of standardised features from raw sums
# With z = (x - mu)/sigma and centred labels, the normal equations are
#   (G + rho*N*I) theta = b,   G = sum z z',   b = sum z (y - ybar),
# and both come from the raw sums:
#    sum (x-mu)(x-mu)' = Sxx - N mu mu',      sum (x-mu)(y-ybar) = Sxy - N mu ybar'.
def fit_ridge(stats: Stats, rho: float) -> RidgeModel:
    n = stats.n
    if n < 10:
        raise ValueError(f"too few rows to fit ({n:.0f})")
    p = len(stats.sx)   # sum over n for each feature p
    mu = stats.sx / n   # mu is (p, )
    var = np.maximum(np.diag(stats.sxx) / n - mu ** 2, 0.0)    # sxx diagonals will be all sum xi^2 for all features
    sigma = np.sqrt(var)
    live = sigma > 1e-9 * np.maximum(np.abs(mu), 1.0)
    inv = np.where(live, 1.0 / np.where(live, sigma, 1.0), 0.0) # np.where evaluates both the inputs before choosing, so nest it
    ybar = stats.sy / n # ybar is (H, )
    G = (stats.sxx - n * np.outer(mu, mu)) * np.outer(inv, inv) # * is elementwise multiplication
    b = (stats.sxy - n * np.outer(mu, ybar)) * inv[:, None] # inv[:, None] turns inv from (p,) to (p, 1), which broadcasts over (p, H)
    theta = np.linalg.solve(G + (rho * n) * np.eye(p), b)   
    return RidgeModel(mu, inv, ybar, theta)


# the experiment
@dataclass
class WFResult:
    test_days: list[int]
    ic_model: np.ndarray            # (n_test, H)     ridge, out of sample
    ic_feature: np.ndarray          # (n_test, p, H)  each raw feature used as the signal
    rho: np.ndarray                 # (n_test, H)     rho chosen on the validation days
    preds: list[np.ndarray]         # per test day (n_rows, H) float32 out-of-sample predictions

# Spearman correlation array of size (H, )
def _day_ic(pred: np.ndarray, Y: np.ndarray, stride: int = 1) -> np.ndarray:
    return np.array([spearman_ic(pred[::stride, h], Y[::stride, h])         # per-horizon columns
                     for h in range(Y.shape[1])])

# MSE array of size (H, )
def _day_mse(pred: np.ndarray, Y: np.ndarray, stride: int = 1) -> np.ndarray:
    p, y = pred[::stride], Y[::stride]
    ok = np.isfinite(p) & np.isfinite(y)
    n = ok.sum(axis=0)
    ss = (np.where(ok, p - y, 0.0) ** 2).sum(axis=0)
    return np.where(n > 0, ss / np.maximum(n, 1), np.nan)

# X_days is nxp, Y_days is nxh; both time-ordered
def walk_forward(X_days: list[np.ndarray], Y_days: list[np.ndarray],
                cfg: WFConfig) -> WFResult:
    n_days = len(X_days)
    if n_days != len(Y_days):
        raise ValueError("X_days and Y_days must have the same length")
    if n_days <= cfg.min_train_days:
        raise ValueError(f"need more than min_train_days={cfg.min_train_days} days, "
                         f"got {n_days}")
    p = X_days[0].shape[1]      # features
    H = Y_days[0].shape[1]      # horizons

    if cfg.clip_q is None:
        lo, hi = np.full(p, -np.inf), np.full(p, np.inf)
    else:
        lo, hi = fit_clip_bounds(X_days[:cfg.min_train_days], cfg.clip_q)

    def clip(X: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(X, dtype=np.float64), lo, hi)

    # Per-day sufficient statistics, computed once. A fold only ever sums the days of its
    # own training window (checked by the perturbation test).
    stats = []
    for X, Y in zip(X_days, Y_days):
        m = fit_mask(X, Y, cfg.purge_rows)
        stats.append(day_stats(clip(X[m]), Y[m]))

    folds = fold_plan(n_days, cfg)
    ic_model = np.full((len(folds), H), np.nan)
    ic_feature = np.full((len(folds), p, H), np.nan)
    rho_chosen = np.full((len(folds), H), np.nan)
    preds: list[np.ndarray] = []

    for k, fold in enumerate(folds):
        # choose rho per horizon: fit on `fit` days, score on `val` days
        fit_stats = sum_stats([stats[i] for i in fold.fit])
        val_score = np.full((len(cfg.rhos), H), np.nan) # each row is avg val score per horizon for the rho
        for r, rho in enumerate(cfg.rhos):
            model = fit_ridge(fit_stats, rho)
            per_day = []
            for v in fold.val:
                pv = model.predict(clip(X_days[v]))
                per_day.append(_day_ic(pv, Y_days[v], cfg.val_stride) if cfg.select_by == "ic"
                               else -_day_mse(pv, Y_days[v], cfg.val_stride))       # score based on ic or mse
            val_score[r] = np.nanmean(np.array(per_day), axis=0)    # average score per horizon
        best = np.nanargmax(np.where(np.isnan(val_score), -np.inf, val_score), axis=0)  # best rho value per horizon, (H, ) array
        rho_chosen[k] = np.asarray(cfg.rhos)[best]

        # refit on the whole training window with the chosen rho
        full_stats = sum_stats([stats[i] for i in fold.train])
        models = [fit_ridge(full_stats, rho) for rho in cfg.rhos]   # retrain for all rhos
        theta = np.stack([models[best[h]].theta[:, h] for h in range(H)], axis=1)   # for each h, choose the model using best rho
        final = RidgeModel(models[0].mu, models[0].inv_sigma, models[0].ybar, theta) # model for 1-s horizon

        # predict the test day once and score it
        Xt, Yt = X_days[fold.test_day], Y_days[fold.test_day]
        pred = final.predict(clip(Xt))
        ic_model[k] = _day_ic(pred, Yt)
        for j in range(p):
            for h in range(H):
                ic_feature[k, j, h] = spearman_ic(Xt[:, j], Yt[:, h])
        preds.append(pred.astype(np.float32))

    return WFResult([f.test_day for f in folds], ic_model, ic_feature, rho_chosen, preds)