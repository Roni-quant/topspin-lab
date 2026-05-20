# CLAUDE.md

Repo guidance for AI assistants working on this codebase.

## Project

Table tennis match prediction using sequential Elo ratings + Random Forest. Headline: 75.06% accuracy / 0.836 AUC on truly-unseen ITTF World Team Championships London 2026.

## Structure

- `pipeline/` - numbered scrape → clean → Elo → features → train stages
- `ratings/` - Elo engine (`elo.py`)
- `experiments/` - London 2026 validation, retrain script, HTML report builder
- `viz/` - plot generators (writes `docs/img/*.png`)
- `docs/` - `methodology.md` + `results.md` + generated images
- `tests/` - scraper / rate-limiter tests
- `data/`, `models/` - local artifacts (gitignored, regenerate)

## Pipeline Stages (run in order)

1. `pipeline.fetch_events` → `data/raw/events_index.parquet`
2. `pipeline.fetch_matches` → `data/raw/matches_{year}.parquet`
3. `pipeline.merge_raw` → `data/raw_matches.parquet`
4. `pipeline.clean` → `data/matches_clean.parquet`
5. `pipeline.compute_elo` → `data/matches_with_elo.parquet`
6. `pipeline.generate_features_v2` → `data/model_features.parquet`
7. `experiments.retrain_enhanced_rf` → `models/random_forest_v2.pkl`
8. `experiments.validate_london_2026` → out-of-sample metrics

## Key Rules

- Parquet only (no CSV for datasets)
- Strict chronological processing - never reorder matches
- No look-ahead - features at time `t` use only data with timestamp `< t`
- Time-based splits only (no random splits on time-series)
- `models/*.pkl` never committed - regenerate via retrain script

## Dataset Schemas

### `raw_matches.parquet`
`match_key`, `source_match_id`, `match_date`, `event_id`, `event_name`, `player_a_id`, `player_a_name`, `player_b_id`, `player_b_name`, `winner_id`, `result`, `games`, `category`

### `matches_clean.parquet`
`match_date`, `player_a_id`, `player_b_id`, `winner_id`

### `matches_with_elo.parquet`
`match_date`, `player_a_id`, `player_b_id`, `elo_a_before`, `elo_b_before`, `winner_id`

### `model_features.parquet` (9 enhanced features)
`elo_difference`, `form_last_5_a/b`, `form_last_10_a/b`, `form_7_days_a/b`, `matches_last_7_a/b`, `cumulative_matches_a/b`, `cumulative_wins_a/b`, `target`

## Current Results

- Training corpus: 157,836 matches up to 2026-03-16
- London 2026 holdout (822 singles rubbers): **75.06% acc / 0.8356 AUC**
- 2024–2026 time-based holdout: 70.26% acc / 0.7794 AUC
- Pure Elo prior alone: 73.97% acc - carries ~95% of signal
- Top feature: `elo_difference` (~59% importance). Opponent recent form ~2× player's own.

See `docs/results.md` for full numbers, `docs/methodology.md` for design decisions.

## Conventions

- Player IDs: integer (`Int64`)
- Base Elo: 1500, K-factor: 32
- Cold-start matches (≥1 player with zero prior history) excluded from headline metric
- Doubles filtered at scrape time - Elo on individuals only
