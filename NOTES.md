# Short-horizon signal research: working notes

Running log of decisions, data findings, results and open items.

Last updated: 2026-11-16.

---

## 1. What this project is

Predict the forward mid-price return of Binance USD-M BTCUSDT (seconds horizon) from order-book
and trade features, with strict walk-forward validation, honest cost modelling, and a
"what did not work" section.
Data: Binance public daily files (data.binance.vision), `bookTicker`
(top of book) and `aggTrades`, 2024-01-01 to 2024-03-30 (90 days; bookTicker on the site ends
around 2024-03-30).

Status: pipeline, walk-forward (ridge), checks and cost/PnL model done; README written. Remaining
items are in section 11.

Scope: the write-up covers ridge and the single-feature baseline (raw imbalance) only. Non-linear
models are out of scope for this write-up.

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

**Interleaved bookTicker files (19 of 90 days).** `utils/audit_raw.py` flagged
2024-01-02 and 2024-02-04 .. 2024-02-21 as having timestamps that go backwards (millions of
steps, jumps of hours).
- Diagnosis on 2024-02-05: no duplicate `update_id`; sorting by `update_id` gives zero backwards
  time steps. The file is two internally ordered streams (one from 00:00 UTC, one from about 15:24 UTC)
  interleaved row by row.
- **Checked in the raw CSV** (`utils/check_csv_order.py`, checksum-verified zip, pure-Python scan):
  the CSV inside Binance's zip is itself interleaved (11,674,170 backwards steps), and the parquet
  matches the CSV row for row. So it is Binance's file, not our conversion.
- Only 2024-02-05 was checked directly in the CSV. The other 18 flagged days show the same
  signature in the parquet audit but were not each checked in the raw CSV.
- **Fix:** `read_sorted` in `02_build_features.py` sorts by `update_id` when the id column is not
  already sorted (raw files untouched, sorting is in memory only) and refuses to proceed on duplicate
  ids. The monotone-timestamp guard in `build_features` stays, so any day still out of order after
  sorting fails loudly. aggTrades passed the audit and need no sort.
- Sanity: a repaired day (2024-02-05) and a clean day (2024-02-03) give the same IC structure.
- 2024-02-07 has 2 unusable mid rows (NaN or <= 0) at the start of the day; harmless, but an earlier
  "zero nulls" check missed them because they are NaN, not null.

**Day-to-day volatility differs a lot.**
One-day label std (bps) at 1/5/10/30 s: 0.54/1.34/1.94/3.3 on Mon 2024-02-05 and
0.30/0.73/1.07/1.87 on Sat 2024-02-03.

## 4. Features (15 model inputs)

`spread_ticks, imbalance, depth, micro_dev_bps, ret_1s, ret_5s, ret_30s, n_updates_1s,
ofi_1s, ofi_5s, ofi_30s, tvol_1s, tflow_1s, tflow_5s, tflow_30s`

- `micro_dev_bps = (spread/2) * imbalance` (scaled by price), and the spread is almost always one
  tick, so it is nearly a rescaled copy of `imbalance`. Their ICs agree to the third decimal.
  Redundant for ridge.
- Unsigned features (`spread_ticks, depth, n_updates_1s, tvol_1s`) have single-feature IC near 0
  against signed returns. That does not show they are useless; interactions are untested.

## 5. Validation design (`walkforward.py`, `03_run_walkforward.py`)

- Day-level walk-forward, expanding window. First test day is day 31 (2024-01-31): train on the
  30 prior days (25 fit + last 5 as validation to choose rho). 60 test days.
- **Purge:** the last 60 s of every fitting day is dropped (need >= L + h = 31 s for the longest
  horizon)
- Clipping bounds (0.1% / 99.9% quantiles) come from the first 30 days only; scaling is fitted on
  training days only; the final model for each test day is refitted from scratch on all its training
  days with the chosen rho.
- **rho is chosen by validation IC (default `--select-by ic`).** IC measures ranking quality and is
  blind to the scale of predictions; scale is handled in a separate calibration step. The earlier
  default was validation squared error. Squared error is noisy when the signal is weak (R^2 about
  1-2%, heavy-tailed labels, 5 validation days): it gave ridge IC 0.0131 at L = 10 s, against 0.0202
  with IC selection and 0.0197 for imbalance. Both rules give the same results at L = 1 s (within
  0.001). `data\walkforward` is the IC-selected run and is the canonical L = 1 s result; the earlier
  MSE-selected run was discarded.
- Ridge is solved in closed form from per-day sufficient statistics; matches a brute-force solve and
  scikit-learn's `Ridge` in tests. Grid for rho: 1e-6 .. 100 (rho is the penalty per training row).
- **Baseline declared in advance:** `imbalance` used raw as the signal.
- Tests (`pytest`) cover features, labels, metrics, walk-forward (including no-look-ahead tests),
  the placebo helpers and the trading helpers.
- Every results folder has a `run_config.json` recording the arguments that produced it;
  `utils/list_runs.py` tabulates them.

## 6. Results

### 6.1 Walk-forward IC, L = 1 s, 60 out-of-sample days (2024-01-31 .. 2024-03-30)

| Horizon | Imbalance | Ridge | Ridge - imbalance (paired t) |
|---|---|---|---|
| 1 s | 0.1928 | 0.1902 | -0.0026 (-6.8) |
| 5 s | 0.1627 | 0.1608 | -0.0019 (-4.6) |
| 10 s | 0.1278 | 0.1268 | -0.0010 (-1.5) |
| 30 s | 0.0679 | 0.0712 | +0.0033 (+3.3) |

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

### 6.3 Checks

- Cross-day and rolled-label placebos give IC about 0 (|t| < 3 everywhere). They do NOT test
  look-ahead or train/test leakage: a planted-leak test gave real IC 0.83, placebo 0. Look-ahead is
  covered by the perturbation tests and the latency decay.
- Purge 60 s vs 3600 s: ridge IC changes by at most 0.0005. Imbalance identical.
- Repaired days (19 days re-sorted by `update_id`): their higher IC is volatility. corr(daily IC,
  log same-day volatility) = -0.93; repaired-day effect after controlling for volatility, update rate
  and weekend = -0.0065 (t = -0.55). Daily imbalance IC at 10 s: weekdays mean 0.10 (43 days, range
  0.004 to 0.20), weekends mean 0.21 (17 days, range 0.05 to 0.34).

### 6.4 Costs and PnL (ridge, 10 s horizon, `04_costs_and_pnl.py`)

Fees: Binance USD-M regular-user example (maker 0.02%, taker 0.05%). The help page calls them
hypothetical; the tier still needs checking. Taker round trip = 2 x 5 bps + spread crossed (about
0.02 bps on BTCUSDT); maker round trip = 2 x 2 bps, assuming every order fills (an upper bound).

- Deciles of the prediction: mean realised 10 s return -0.27 bps (lowest tenth) to +0.29 bps
  (highest); top minus bottom +0.57 bps (t = 16.4).
- One position at a time, gross edge per trade (= break-even round-trip cost) by nominal top
  fraction 10% / 5% / 2% / 1% / 0.5% / 0.1%: 0.265 / 0.225 / 0.189 / 0.148 / 0.171 / 0.072 bps, with
  2,734 / 1,850 / 1,035 / 645 / 410 / 211 trades per day. Best: 0.27 bps at 10%.
- Net per trade at the best threshold: maker about -3.7 bps, taker about -9.8 bps. Break-even needs
  about 15x (maker) or 38x (taker) more edge.
- Selection (threshold chosen on the first 29 test days, reported on the last 30), daily PnL: zero cost
  +455 bps (t 3.6); maker -1,301 bps (t -1.2); taker -3,329 bps (t -1.2). With costs the rule picks
  0.1%, the fewest trades.
- Regimes by previous-day volatility, top 1%: IC 0.208 / 0.115 / 0.057 (low / middle / high), label
  std 2.1 / 3.0 / 4.3 bps, gross 0.27 / 0.16 / 0.10 bps per trade.
- Known issue: the threshold is the previous 5 days' quantile of |prediction|, so it drifts with the
  regime. The top 1% admitted about 2.8x the nominal rows (2,394 vs 864 per day) and the top 0.1%
  about 17x (1,498 vs 86). Edge per signal by threshold is noisy (0.07 bps at 5%, 0.79 at 0.5%); do not
  interpret its shape. Use the `signal_share` column.

## 7. What did not work / negative findings

1. A 15-feature linear model is no better than `imbalance` alone at horizons up to 10 s.
2. `micro_dev_bps` is a rescaled copy of `imbalance`: no extra information.
3. OFI and signed trade flow have lower single-feature IC than book imbalance (e.g. 10 s: ofi_5s 0.073,
   ofi_1s 0.070, imbalance 0.128).
4. The signal decays within seconds of added latency.
5. Choosing rho by validation squared error hurt ridge at weak signal (L = 10 s: 0.0131 vs 0.0202 with
   IC selection). Not an issue at L = 1 s.
6. (Single-day observation, unconfirmed on the full sample) `ret_30s` flips sign against the 30 s label.
7. Nothing survives costs: gross edge is 0.07 to 0.27 bps per trade against 4 (maker, optimistic) to
   10 (taker) bps.
8. Same-day volatility explains IC but is not usable as a filter; trailing volatility is.
9. Trailing-quantile thresholds drift with the regime (section 6.4).

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
  vs noise) does not depend on the outcome, and both sets of results were reported.
- A check only tests what it is built to test: say what it cannot see. (The cross-day placebo is not a
  leakage test.)
- IC is scale-free; edge in bps is IC x label std. Quote both.
- A big IC difference across a calendar block: look for the regime before blaming data.
- Do the cost arithmetic early: 0.3 bps against 10 bps decides the project.
- Quote numbers from the output files, not from memory: a README line about weekday and weekend IC was
  wrong until it was recomputed from `repaired_diagnostic.csv`.

## 9. Decisions

- Primary setting L = 1 s, h = 10 s, fixed in advance.
- Baseline = raw imbalance, fixed in advance.
- rho by validation IC (see section 5).
- Costs: taker and maker scenarios reported side by side; the maker one is an explicit upper bound.
- Final write-up is ridge only (see Scope in section 1).

## 10. How to reproduce

```powershell
pip install -r requirements.txt
pip install -e .
python .\scripts\01_fetch_binance.py download --start 2024-01-01 --end 2024-03-30
python .\utils\audit_raw.py                                     # ordering / coverage audit
python .\scripts\02_build_features.py --start 2024-01-01 --end 2024-03-30
pytest
python .\scripts\03_run_walkforward.py                          # L = 1 s, all horizons -> data\walkforward
python .\scripts\04_costs_and_pnl.py                            # deciles, edge, costs, regimes
python .\utils\placebo_checks.py
python .\utils\diagnose_repaired_days.py
python .\utils\check_csv_order.py --date 2024-02-05             # raw CSV order check for one day
python .\utils\make_plots.py                                    # -> docs\figures
python .\utils\list_runs.py                                     # what is in each data\ results folder
```

Latency runs for the latency figure, one folder each: `python .\scripts\03_run_walkforward.py
--latency 5 --out data\wf_L5` (and 0, 2, 10).

## 11. Still to do

- [ ] Check the Binance fee tier that would actually apply (the README uses the regular-user example).
- [ ] Optional experiment: classify the sign of the move with a linear model (logistic regression),
      weighting each row by |label|, and compare with ridge on the same walk-forward folds.
- [ ] Optional: a fill model for the maker case (fill probability, adverse selection) instead of the
      always-filled upper bound.
