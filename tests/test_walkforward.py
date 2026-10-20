import numpy as np
import pytest

from signal_research.metrics import spearman_ic
from signal_research.walkforward import (
    WFConfig, day_stats, fit_clip_bounds, fit_mask, fit_ridge, fold_plan, sum_stats,
    walk_forward,
)


# folds
def test_fold_plan_train_preceeds_test():
    cfg = WFConfig(min_train_days=10, val_days=3)
    folds = fold_plan(15, cfg)
    assert [f.test_day for f in folds] == [10, 11, 12, 13, 14]
    for f in folds:
        assert max(f.train) < f.test_day                  
        assert list(f.val) == list(range(f.test_day - 3, f.test_day))
        assert set(f.fit).isdisjoint(f.val)                 
        assert list(f.fit) + list(f.val) == list(f.train)
        assert f.train.start == 0                  


def test_fold_plan_rolling_window():
    cfg = WFConfig(min_train_days=10, val_days=3, max_train_days=12)
    folds = fold_plan(30, cfg)
    assert all(len(f.train) <= 12 for f in folds)
    assert folds[-1].train.start == 29 - 12


def test_config_validation():
    with pytest.raises(ValueError):
        WFConfig(min_train_days=3, val_days=3)
    with pytest.raises(ValueError):
        WFConfig(min_train_days=10, max_train_days=5)
    with pytest.raises(ValueError):
        WFConfig(rhos=(0.0, 1.0))


def test_fit_mask_drop_and_purge():
    X = np.ones((100, 2))
    Y = np.ones((100, 2))
    X[3, 0] = np.nan
    Y[50, 1] = np.nan
    m = fit_mask(X, Y, purge_rows=10)
    assert not m[3] and not m[50]
    assert not m[90:].any() and m[:3].all() and m[4:50].all()
    assert m.sum() == 100 - 2 - 10


def test_clip_bounds_use_only_the_rows_given():
    rng = np.random.default_rng(0)
    lo, hi = fit_clip_bounds([rng.normal(size=(5000, 3))], q=0.01, stride=1)
    assert np.all(lo < -2.0) and np.all(hi > 2.0) and np.all(lo > -3.5) and np.all(hi < 3.5)


# ridge
def _zscored_ridge(X, Y, rho):
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    ybar = Y.mean(0)
    n = len(X)
    theta = np.linalg.solve(Z.T @ Z + rho * n * np.eye(X.shape[1]), Z.T @ (Y - ybar))
    return mu, sd, ybar, theta


def test_ridge_from_sums_matches_brute_force():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 4)) * [1, 10, 0.1, 5] + [3, -2, 0.5, 100]
    Y = np.column_stack([X @ [0.3, 0.02, 2.0, 0.0] + rng.normal(size=4000),
                         rng.normal(size=4000)])
    parts = [day_stats(X[i::4], Y[i::4]) for i in range(4)]
    for rho in (1e-4, 1e-2, 1.0):
        m = fit_ridge(sum_stats(parts), rho)
        mu, sd, ybar, theta = _zscored_ridge(X, Y, rho)
        assert m.mu == pytest.approx(mu)
        assert m.ybar == pytest.approx(ybar)
        assert m.theta == pytest.approx(theta, rel=1e-6, abs=1e-9)
        assert m.predict(X) == pytest.approx(((X - mu) / sd) @ theta + ybar)


def test_ridge_matches_sklearn():
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 3)) * [1, 4, 0.2]
    y = X @ [1.0, 0.1, 5.0] + rng.normal(size=3000)
    rho = 0.01
    m = fit_ridge(day_stats(X, y[:, None]), rho)
    mu, sd = X.mean(0), X.std(0)
    sk = linear_model.Ridge(alpha=rho * len(X), fit_intercept=True).fit((X - mu) / sd, y)
    assert m.theta[:, 0] == pytest.approx(sk.coef_, rel=1e-6)
    assert m.ybar[0] == pytest.approx(sk.intercept_)


def test_ridge_limits():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 2))
    y = (X @ [2.0, -1.0] + 5.0)[:, None]                     
    st = day_stats(X, y)
    small = fit_ridge(st, 1e-12)
    assert small.predict(X)[:, 0] == pytest.approx(y[:, 0], abs=1e-6)     
    big = fit_ridge(st, 1e9)                               
    assert np.abs(big.theta).max() < 1e-6
    assert big.predict(X)[:, 0] == pytest.approx(np.full(2000, y.mean()), abs=1e-3) # predicts mean for large rho


def test_constant_feature_is_harmless():
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(size=500), np.full(500, 1.0)])
    y = (2 * X[:, 0] + rng.normal(size=500) * 0.1)[:, None]
    m = fit_ridge(day_stats(X, y), 1e-3)
    assert np.all(np.isfinite(m.theta)) and m.theta[1, 0] == 0.0


# walk-forward
def _synthetic(n_days=12, n=1500, seed=0, signal=0.3):
    rng = np.random.default_rng(seed)
    X_days, Y_days = [], []
    for _ in range(n_days):
        X = rng.normal(size=(n, 4))
        y = signal * X[:, 0] + rng.normal(size=n)
        Y = np.column_stack([y, rng.normal(size=n)])          # horizon 0: signal; horizon 1: noise
        Y[-5:] = np.nan                                       # labels missing at the day's end
        X_days.append(X)
        Y_days.append(Y)
    return X_days, Y_days


CFG = WFConfig(min_train_days=6, val_days=2, purge_rows=10, val_stride=1)


def test_walk_forward_finds_real_signal():
    X_days, Y_days = _synthetic()
    res = walk_forward(X_days, Y_days, CFG)
    assert res.test_days == list(range(6, 12))
    assert res.ic_model[:, 0].mean() > 0.15                   # true corr = 0.3/sqrt(1.09) ~ 0.29
    assert abs(res.ic_model[:, 1].mean()) < 0.06              # pure-noise label: no skill
    assert res.ic_feature[:, 0, 0].mean() > 0.15              # raw feature 0 carries it
    assert res.preds[0].shape == (1500, 2) and res.preds[0].dtype == np.float32


def test_no_look_ahead():
    X_days, Y_days = _synthetic(seed=5)
    base = walk_forward(X_days, Y_days, CFG)
    t = 8
    rng = np.random.default_rng(99)
    X2 = [x.copy() for x in X_days]
    Y2 = [y.copy() for y in Y_days]
    Y2[t] = rng.normal(size=Y2[t].shape) * 100                # day t labels scrambled
    for d in range(t + 1, len(X2)):                           # everything after t scrambled
        X2[d] = rng.normal(size=X2[d].shape) * 50
        Y2[d] = rng.normal(size=Y2[d].shape) * 50
    new = walk_forward(X2, Y2, CFG)
    for k, day in enumerate(base.test_days):
        if day <= t:
            assert new.preds[k] == pytest.approx(base.preds[k]), f"prediction for day {day} changed"
            assert new.rho[k] == pytest.approx(base.rho[k])
    # test should not vacuous: scrambling a TRAINING day does change later predictions
    X3 = [x.copy() for x in X_days]
    Y3 = [y.copy() for y in Y_days]
    Y3[t - 1] = rng.normal(size=Y3[t - 1].shape) * 100
    other = walk_forward(X3, Y3, CFG)
    k_t = base.test_days.index(t)
    assert not np.allclose(other.preds[k_t], base.preds[k_t])


def test_select_by_validation_and_ic_mode_runs():
    with pytest.raises(ValueError):
        WFConfig(select_by="r2")
    X_days, Y_days = _synthetic(seed=8)
    cfg = WFConfig(min_train_days=6, val_days=2, purge_rows=10, val_stride=1, select_by="ic")
    res = walk_forward(X_days, Y_days, cfg)
    assert res.ic_model[:, 0].mean() > 0.15


def test_too_few_days_raises():
    X_days, Y_days = _synthetic(n_days=6)
    with pytest.raises(ValueError):
        walk_forward(X_days, Y_days, CFG)