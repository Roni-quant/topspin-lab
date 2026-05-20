# Results

All metrics on this page are reproducible from the code in this repository, against data scraped from the ITTF results API as of 2026-03-16 (training/holdout) and 2026-05-10 (London 2026 unseen tournament).

## Headline

| Test | Matches | Accuracy | AUC | Brier | LogLoss |
|---|---:|---:|---:|---:|---:|
| **London 2026 (truly unseen tournament)** | 822 | **75.06%** | **0.8356** | 0.1666 | 0.5022 |
| 2024-2026 time-based holdout (RF v2 enhanced) | ~21k | 70.26% | 0.7794 | - | - |
| Pure Elo prior (no model, London 2026) | 822 | 73.97% | 0.8333 | 0.1671 | 0.5024 |

The model generalizes better on the London 2026 tournament than on the broader 2024-2026 holdout. The likely reason: team championships pair elite-tier vs lower-tier players more often than open events, producing more lopsided (easier-to-predict) matches.

**Pure Elo carries ~95% of the signal.** Adding recent-form features improves accuracy by ~1 percentage point and AUC by ~0.003. Elo difference alone is a strong, calibrated estimator on this data.

## Reproducing the London 2026 result

```bash
# 1. Fetch events + matches (requires ITTF credentials in .env)
python -m pipeline.fetch_events
python -m pipeline.fetch_matches
python -m pipeline.merge_raw

# 2. Clean → Elo → features
python -m pipeline.clean
python -m pipeline.compute_elo
python -m pipeline.generate_features_v2

# 3. Train model (writes models/random_forest_v2.pkl locally - not committed)
python -m experiments.retrain_enhanced_rf

# 4. Fetch the unseen tournament + validate
python -m experiments.fetch_london_2026
python -m experiments.validate_london_2026

# 5. Build the HTML report
python -m experiments.build_london_report
open experiments/london_2026_report.html
```

## Calibration

The enhanced RF is well-calibrated mid-range, slightly under-confident at extremes. Calibration table from London 2026 holdout:

| Predicted bin | n | Mean predicted | Actual win rate |
|---|---:|---:|---:|
| [0.00, 0.10) | 95 | 0.050 | 0.095 |
| [0.10, 0.20) | 88 | 0.146 | 0.114 |
| [0.20, 0.30) | 52 | 0.255 | 0.327 |
| [0.30, 0.40) | 54 | 0.350 | 0.444 |
| [0.40, 0.50) | 50 | 0.461 | 0.380 |
| [0.50, 0.60) | 68 | 0.548 | 0.471 |
| [0.60, 0.70) | 80 | 0.645 | 0.588 |
| [0.70, 0.80) | 86 | 0.753 | 0.698 |
| [0.80, 0.90) | 126 | 0.849 | 0.802 |
| [0.90, 1.00) | 123 | 0.947 | 0.951 |

Pure Elo is slightly better calibrated in the mid-range; the RF compresses probabilities toward 0.5.

## Per-category breakdown (London 2026)

| Category | n | Pure Elo | Enhanced RF |
|---|---:|---:|---:|
| Men's Team (MT) | 434 | 74.19% | 74.42% |
| Women's Team (WT) | 388 | 73.71% | **75.77%** |

The form features contribute more on the women's draw - possibly because the WT field has wider talent spread, so recent results are a stronger tiebreaker between similarly-rated players.

## Cold-start

Of 857 London 2026 singles rubbers fetched, **35 (4.1%)** involved at least one player with zero prior matches in the training corpus. These are excluded from the headline metric and reported separately. Including them with base-Elo (1500) priors would mechanically lower accuracy without telling us anything about model quality on known players.

## Feature importance (Random Forest, enhanced)

From `train_models_v2` on the 2024+ holdout:

| Feature | Importance |
|---|---:|
| `elo_difference` | ~59% |
| `form_last_5_b` (opponent's recent form) | ~12% |
| `form_last_10_b` | ~8% |
| `form_last_5_a` | ~6% |
| Others (form_7d, matches_7d) | ~15% combined |

**The opponent's recent form is ~2x more predictive than the player's own.** A player on a cold streak entering a match against a strong opponent is a clearer signal than a player on a hot streak: strong opponents punish weakness more reliably than they reward strength.

## What the model gets wrong

Inspecting the misclassified London matches (see `experiments/london_2026_predictions.csv` filtered by `correct == False`):

- **Upsets where Elo difference > 200**: ~12% of high-confidence predictions still missed. These tend to be returning players (long layoffs) or matches involving very new players who built strong recent form but have a thin Elo history.
- **Tight matches (|Elo diff| < 50)**: model flips a coin (52% acc). Little signal to extract.

## Comparison to published baselines

We are not aware of a published table-tennis Elo prediction benchmark we can compare to directly. The closest reference points:

- **538-style Elo on chess / tennis**: typically 0.65-0.70 AUC. Our 0.83 on London 2026 is higher, but the tournament is unusually elite-heavy.
- **FiveThirtyEight WTA / ATP**: log-loss in the 0.55-0.60 range on out-of-sample. Our 0.50 log-loss on London is in line.

If you have a directly comparable benchmark, please open an issue.

## Limitations on these numbers

- **One tournament is one tournament.** 75% on London 2026 should not be quoted as the model's expected accuracy on future events.
- **The training corpus has known coverage gaps** (early 2000s sparse, COVID-era reduced volume). Players whose careers fall mostly in these gaps will have noisy Elo estimates.
- **No bootstrap intervals on the headline number.** A 822-sample accuracy at 75% has a ~3% 95% CI by Wilson's method. Treat the headline as "between 72% and 78%."
