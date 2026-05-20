# Methodology

This document describes the design decisions behind the pipeline. The bar is *reproducibility first, accuracy second*. If a metric improves but cannot be reproduced from raw data + code, it is not a result.

## 1. The leakage discipline

The single most common failure mode in sports prediction is **look-ahead bias** - letting information that did not exist at prediction time leak into features. We treat leakage as a correctness bug, not a performance issue.

Two rules:

1. **All features for a match at time `t` depend only on data with timestamp `< t`.** No exceptions.
2. **All evaluation splits are time-based.** Train on the past, test on the future. Never random splits.

How we enforce this:

- Matches are processed in strict `match_date` order through the Elo engine. Each match's `elo_a_before` / `elo_b_before` are captured *before* the engine updates ratings (see `ratings/elo.py:process_match`).
- Recent-form features (`form_last_5`, `form_last_7_days`, etc.) are computed by walking the history forward and only using `player_matches[pid]` entries appended *before* the current row (see `pipeline/generate_features_v2.py:_compute_recent_form_and_cumulative`).
- The train/test split cuts on calendar year, not row index (see `pipeline/train_models_v2.py:_time_based_split`).

## 2. Why Elo

Elo is a maximum-entropy estimator: it is the simplest model that produces calibrated win probabilities given only pairwise outcomes. It has three properties we want:

1. **Causal**: a player's rating at time `t` is fully determined by matches with `date < t`.
2. **Sequential**: no batch refit, no risk of in-batch contamination.
3. **Decomposable**: feature importance trivially attributes signal to "rating difference" vs "form" vs "experience" - useful for diagnosing the model.

We use the standard formulation with `K=32`, base rating `1500`. See `ratings/elo.py`.

## 3. The pipeline (Parquet end-to-end)

Stages are deliberately small, each writing a Parquet file with a fixed schema. No CSV, no Pickle, no in-memory hand-offs.

| Stage | Input | Output | Purpose |
|---|---|---|---|
| 1. `fetch_events` | ITTF API | `data/raw/events_index.parquet` | Event metadata |
| 2. `fetch_matches` | events index | `data/raw/matches_{year}.parquet` | Per-year match records |
| 3. `merge_raw` | yearly files | `data/raw_matches.parquet` | Deduped, sorted, name-joined |
| 4. `clean` | raw_matches | `data/matches_clean.parquet` | Player-ID validation, drop walkovers/RET |
| 5. `compute_elo` | clean | `data/matches_with_elo.parquet` | Pre-match Elo ratings |
| 6. `generate_features_v2` | with_elo | `data/model_features.parquet` | Form + cumulative features |
| 7. `train_models_v2` | features | metrics (no artifact) | Compare baseline vs enhanced |
| 8. `experiments/retrain_enhanced_rf` | features | `models/random_forest_v2.pkl` (local) | Save trained model |
| 9. `experiments/validate_london_2026` | unseen tournament | `experiments/london_2026_predictions.csv` | Out-of-sample validation |

Each stage is idempotent: re-running with the same inputs produces the same outputs.

## 4. Feature design

### Baseline features (5)
- `elo_difference` - `elo_a_before - elo_b_before`
- `cumulative_matches_a/b`, `cumulative_wins_a/b` - career counters

### Enhanced features (9)
- `elo_difference`
- `form_last_5_a/b`, `form_last_10_a/b` - win rate over the last N matches
- `form_7_days_a/b` - win rate over the last 7 days (fatigue signal)
- `matches_last_7_a/b` - match count over the last 7 days (workload signal)

Recent form is computed *per player*, walking each player's match history in time order. A player's first match has all form features as `NaN` (filled with 0 at training time - explicitly noted, not silently imputed).

## 5. Walk-forward validation

We do not trust a single train/test split. The reported holdout (WTTC London 2026 — the 2026 World Team Table Tennis Championships) is one slice; the time-based 2024-2026 holdout is another. Both should agree, and they do (75.06% vs ~70%).

For deeper validation, `pipeline/forward_test.py` runs walk-forward evaluation: roll the train cutoff forward month by month, refit, and score the next window. This catches regime shifts (e.g., post-COVID return-to-play distributions) that a single split would hide.

## 6. Cold-start handling

A "cold-start match" is one where at least one player has zero prior matches in our history. Two reasonable policies:

1. **Assign base Elo (1500) and predict normally.** Cheap but adds noise to evaluation - we are scoring the model on data the model has no information about.
2. **Exclude from evaluation and report the count separately.** What we do.

In London 2026, 35 of 857 singles matches (4.1%) involved at least one cold-start player. These are excluded from the headline metric but reported alongside it.

## 7. Doubles and team formats

The World Team Championships is a *team* event. Each tie is best-of-5 rubbers (4 singles + 1 doubles). Doubles matches are excluded from the model because:

1. Player identity in doubles is a pair, not an individual - Elo on individuals does not apply.
2. The ITTF API tags doubles differently (`MD`/`WD` vs `MT`/`WT`) and our scraper filters them out.

The 822 evaluated London matches are individual singles rubbers inside team contests, not team contest outcomes. To predict at the team-contest level, aggregate the per-rubber probabilities.

## 8. What this model is NOT

- **Not a betting recommendation.** Calibration is good but not perfect; bookmaker margins are tight.
- **Not a player ranking.** Elo is one signal; official rankings use different weighting (recency, event tier, etc.).
- **Not real-time.** Ratings are point-in-time snapshots; no in-match updating.

## 9. Known limitations

- **No surface / equipment effects.** Plastic balls vs celluloid, rubber type, etc. are not modeled.
- **No bo-3 vs bo-5 distinction.** All matches treated equally for Elo updates.
- **K-factor not tuned.** Could be a function of event tier or match importance.
- **No team-rubber order modeling.** In team ties, match order is strategically chosen - we ignore that.

PRs that close these gaps with rigorous backtests are welcome.
