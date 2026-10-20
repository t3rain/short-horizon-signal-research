# Short-horizon signal research: working notes

Running log of decisions, data findings, results and open items.

Last updated: 2026-10-20.

---

## 1. What this project is

Predict the forward mid-price return of Binance USD-M BTCUSDT (seconds horizon) from order-book
and trade features, with strict walk-forward validation, honest cost modelling, and a
"what did not work" section. 
Data: Binance public daily files (data.binance.vision), `bookTicker`
(top of book) and `aggTrades`, 2024-01-01 to 2024-03-30 (90 days; bookTicker on the site ends
around 2024-03-30).

Status: data pipeline done; walk-forward harness and ridge done; LightGBM, cost/PnL model and
write-up still to do.

## 2. Conventions

- **Clock:** `transaction_time` is the single clock for both datasets.
- **Grid:** 1-second rows labelled by END time `t_i = day_start + (i+1) s`, covering `[t_i - 1s, t_i)`.
- **State features** (bid/ask, imbalance, depth, micro_dev, returns) use the last book event
  strictly before `t_i` (`searchsorted(..., side="left") - 1`). **Flow features** (OFI, trade flow,
  counts) sum over the window. Nothing at or after `t_i` is used.
- **Prices** as integer ticks (0.1 for BTCUSDT) inside the OFI calculation.
- **Label:** `y_h[i] = (mid[i+L+h] / mid[i+L] - 1) * 1e4` bps. Latency `L` = 1 s (an order sent at
  `t_i` arrives at `t_i + L`). Horizons h = 1, 5, 10, 30 s. **Primary: L = 1 s, h = 10 s**. The last `L+h` rows of each day are dropped.
- **IC** = Spearman rank correlation between signal and label, computed per test day; reported as
  mean daily IC and t-statistic across days.

All of these were decided before start of the experiment and fixed. Latency sweeps were performed to see the variation but the initial values remain.

## 3. Data issues and decisions

**Interleaved bookTicker files (19 of 90 days).** `scripts/audit_raw.py` flagged
2024-01-02 and 2024-02-04 .. 2024-02-21 as having timestamps that go backwards (millions of
steps, jumps of hours).
- Diagnosis on 2024-02-05: no duplicate `update_id`; sorting by `update_id` gives zero backwards
  time steps. The file is two internally ordered streams (one from 00:00 UTC, one from about 15:24 UTC)
  interleaved row by row.
- **Checked in the raw CSV** (`scripts/check_csv_order.py`, checksum-verified zip, pure-Python scan):
  the CSV inside Binance's zip is itself interleaved (11,674,170 backwards steps), and the parquet
  matches the CSV row for row. So it is Binance's file, not our conversion.
- Only 2024-02-05 was checked directly in the CSV. The other 18 flagged days show the same
  signature in the parquet audit but were not each checked in the raw CSV.
- **Fix:** `read_sorted` in `02_build_features.py` sorts by `update_id` when the id column is not
  already sorted (raw files untouched, sorting is in memory only) and refuses to proceed on duplicate
  ids. The monotone-timestamp guard in `build_features` stays, so any day still out of order after
  sorting fails loudly. aggTrades passed the audit and need no sort.
- Sanity: a repaired day (2024-02-05) and a clean day (2024-02-03) give the same IC structure.

**Day-to-day volatility differs a lot.** 
One-day label std (bps) at 1/5/10/30 s: 0.54/1.34/1.94/3.3 on Mon 2024-02-05 and 
0.30/0.73/1.07/1.87 on Sat 2024-02-03.

## 4. Features (15 model inputs)

`spread_ticks, imbalance, depth, micro_dev_bps, ret_1s, ret_5s, ret_30s, n_updates_1s,
ofi_1s, ofi_5s, ofi_30s, tvol_1s, tflow_1s, tflow_5s, tflow_30s`

- `micro_dev_bps = (spread/2) * imbalance` (scaled by price), and the spread is almost always one
  tick, so it is nearly a rescaled copy of `imbalance`. Their ICs agree to the third decimal.
  Redundant for ridge; trees will split between them.
- Unsigned features (`spread_ticks, depth, n_updates_1s, tvol_1s`) have single-feature IC near 0
  against signed returns. That does not show they are useless; interactions are untested (a tree
  model can use them to modulate the signed features).

## 5. Validation design (`walkforward.py`, `03_run_walkforward.py`)

- Day-level walk-forward, expanding window. First test day is day 31 (2024-01-31): train on the
  30 prior days (25 fit + last 5 as validation to choose rho). 60 test days.
- **Purge:** the last 60 s of every fitting day is dropped (need >= L + h = 31 s for the longest
  horizon)
- Clipping bounds (0.1% / 99.9% quantiles) come from the first 30 days only; scaling is fitted on
  training days only; the final model for each test day is refitted from scratch on all its training
  days with the chosen rho.
- **rho is chosen by validation IC (default `--select-by ic`).** IC measures ranking quality and is
  blind to the scale of predictions; scale is handled in a separate calibration step (regress the label on
  the prediction over validation days, planned for the PnL stage). The earlier default was validation
  squared error. Squared error is noisy when the signal is weak (R^2 about 1-2%, heavy-tailed labels,
  5 validation days): it gave ridge IC 0.0131 at L = 10 s, against 0.0202 with IC selection and
  0.0197 for imbalance. Both rules give the same results at L = 1 s (within 0.001), and both result
  sets are kept (`data\walkforward` = MSE, `data\walkforward_ic` = IC).
- Ridge is solved in closed form from per-day sufficient statistics; matches a brute-force solve and
  scikit-learn's `Ridge` in tests. Grid for rho: 1e-6 .. 100 (rho is the penalty per training row).
- **Baseline declared in advance:** `imbalance` used raw as the signal.
- Tests: 55 pass (features, labels, metrics, walk-forward). Includes no-lookahead tests.

## 6. Results so far

### 6.1 Walk-forward IC, L = 1 s, 60 out-of-sample days (2024-01-31 .. 2024-03-30)

Ridge with rho chosen by validation IC (L = 1 s, 60 days):
| Horizon | Ridge | Ridge - imbalance (paired t) |
|---|---|---|
| 1 s | 0.1902 | -0.0026 (-6.8) |
| 5 s | 0.1608 | -0.0019 (-4.6) |
| 10 s | 0.1268 | -0.0010 (-1.5) |
| 30 s | 0.0712 | +0.0033 (+3.3) |

Chosen rho is scattered across the whole grid under IC selection (IC is flat in rho when the signal is
strong), so the per-fold counts carry no information.

### 6.2 Latency sensitivity (imbalance, 10 s horizon; raw feature, so independent of the rho grid)

| L | 0 s | 1 s | 2 s | 5 s | 10 s |
|---|---|---|---|---|---|
| Imbalance IC (10 s horizon) | 0.2092 | 0.1278 | 0.0938 | 0.0491 | 0.0197 |
| Imbalance t-stat | 18.2 | 11.2 | 9.5 | 7.4 | 5.1 |
| Ridge, rho by MSE | 0.2069 | 0.1267 | 0.0929 | 0.0479 | 0.0131 |
| Ridge, rho by IC | n/a | 0.1268 | n/a | n/a | 0.0202 |

The IC loses 39% in the first second (half-life about 1.4 s between L = 0 and 1), then the
half-life is 2.2 s (1 to 2), 3.2 s (2 to 5) and 3.8 s (5 to 10): a very fast component plus a
slower one. L = 0 is an optimistic upper bound. **Primary latency stays L = 1 s.**

### 6.3 Economics (back-of-envelope, NOT yet a result)

10 s label std is roughly 1-2 bps, so IC 0.128 corresponds to about 0.2 bps of expected move per
standard deviation of signal at L = 1 s (about 0.07 bps at L = 5 s). The assumed taker round-trip
cost is 8-10 bps (to be checked against the actual Binance fee tier). On these numbers a taker
strategy loses; only a maker with queue priority could capture a signal this fast, and this 1-second
grid cannot model that. The cost/PnL step will make this precise.

## 7. What did not work / negative findings (so far)

1. A 15-feature linear model is no better than `imbalance` alone at horizons up to 10 s.
2. `micro_dev_bps` is a rescaled copy of `imbalance`: no extra information.
3. OFI and signed trade flow have lower single-feature IC than book imbalance (e.g. 10 s: ofi_5s 0.073,
   ofi_1s 0.070, imbalance 0.128).
4. The signal decays within seconds of added latency.
5. Choosing rho by validation squared error hurt ridge at weak signal (L = 10 s: 0.0131 vs 0.0202 with
   IC selection). Not an issue at L = 1 s.
6. (Single-day observation, unconfirmed on the full sample) `ret_30s` flips sign against the 30 s label.

## 8. Lessons and corrections log

- Purge gap must be >= latency + horizon (not just horizon).
- Spearman IC ignores the scale of predictions; training loss (squared error) and evaluation metric
  (IC) can differ and should.
- Daily IC t-statistics treat days as independent; neighbouring days share regimes, so read them as
  a guide. 15 features x 4 horizons were looked at, so the best single numbers are optimistic;
  compare against the pre-declared baseline.
- Shuffled K-fold overstates IC because adjacent seconds share label windows and training days see
  the future regime; use walk-forward.
- Sorting by timestamp is unsafe for ties; sort by `update_id` and verify.
- Do not clamp timestamps with a running maximum to "fix" ordering; that hides real problems.
- IC ignores scale; squared error sees it but is noisy at weak signal. Choose hyperparameters by IC,
  calibrate magnitudes separately.
- When IC is flat in a hyperparameter, the "chosen" value is mostly noise; do not interpret counts of
  chosen values.
- I changed the rho-selection default after seeing the L = 10 s result. The reason (scale-blindness
  vs noise) does not depend on the outcome, and both sets of results are reported.

## 9. Open items

- [x] Latency curve with the final rho grid: `foreach ($L in 0,2,5,10) { python .\scripts\03_run_walkforward.py --latency $L --horizons 10 --out data\wf_L$L }`
- [ ] LightGBM on the same folds (decide: retrain every few days, number of trees by early stopping on the
      validation days only, training-row subsampling, strong leaf regularisation). Bar: 0.128 at L = 1 s, h = 10 s.
- [ ] Placebo / leakage checks: pair day d features with another day's labels (IC should be about 0);
      compare daily IC on repaired days (Feb 4 .. Feb 21) vs the rest.
- [ ] Calibration of predicted bps (regress y on yhat over recent days) before thresholds.
- [ ] Cost model: verify Binance fee tier, spread cost, net PnL with a validation-chosen threshold.
- [ ] Plot daily IC over time and IC vs latency for the README.
- [ ] README, CV bullet, optional C++ feature calculator last.

## 10. How to reproduce

```powershell
pip install -r requirements.txt
pip install -e .
python .\scripts\fetch_binance.py download --start 2024-01-01 --end 2024-03-30
python .\scripts\audit_raw.py                                   # ordering / coverage audit
python .\scripts\02_build_features.py --start 2024-01-01 --end 2024-03-30
pytest
python .\scripts\03_run_walkforward.py                          # L = 1 s, all horizons -> data\walkforward
python .\scripts\check_csv_order.py --date 2024-02-05           # raw CSV order check for one day
```