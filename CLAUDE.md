# CLAUDE.md

## Project

Table tennis match prediction research using Elo ratings. Python-based data pipeline with Parquet storage.

## Structure

- `data/` — all datasets (raw, cleaned, features) in Parquet format
- `notebooks/` — exploratory analysis and visualization
- `models/` — trained model artifacts
- `ratings/` — Elo engine implementation
- `evaluation/` — metrics, calibration, reporting
- `experiments/` — ad-hoc experiments and prototypes

## Pipeline Steps (in order)

1. **Data Collection** → `data/raw_matches.parquet`
2. **Data Cleaning** → `data/matches_clean.parquet`
3. **Elo Rating Engine** → `data/matches_with_elo.parquet`
4. **Feature Generation** → `data/model_features.parquet`
5. **Model Training & Testing** — time-based train/test split
6. **Forward Prediction Simulation** — walk-forward evaluation

## Key Rules

- All data stored as Parquet (not CSV)
- Matches must be processed in strict chronological order
- No future data leakage — features and ratings use only past data
- Time-based splits only (no random splits)
- Data quality is the top priority — bad data invalidates everything downstream

## Dataset Schema

### raw_matches.parquet
`match_key`, `source_match_id`, `match_date`, `event_id`, `event_name`, `player_a_id`, `player_a_name`, `player_b_id`, `player_b_name`, `winner_id`, `result`, `games`, `category`

### matches_clean.parquet
`match_date`, `player_a_id`, `player_b_id`, `winner_id`

### matches_with_elo.parquet
`match_date`, `player_a_id`, `player_b_id`, `elo_a_before`, `elo_b_before`, `winner_id`

### model_features.parquet
`elo_difference`, `recent_win_rate`, `matches_last_30_days`, `head_to_head` (optional), plus target

## Research Results (Iteration 2)

**Status:** Pipeline complete. All 6 steps finished with enhanced features.

**Data:** 158,185 clean matches, 12,700 players, 1988–2026

**Model Performance:**
- Random Forest (best): **70.26% accuracy, 0.7794 AUC**
- Logistic Regression (baseline): 69.42% accuracy
- Logistic Regression (enhanced w/ recent form): +1.21% AUC improvement (0.7805)

**Key Insights:**
- Elo difference: 59% feature importance (dominant predictor)
- Opponent's recent form: 11.57% (2× more predictive than player's form)
- No look-ahead bias; 26-month walk-forward validation confirms generalization
- Time-based splits (1988–2023 train, 2024–2026 test) prevent data leakage

## Conventions

- Player IDs must be consistent normalized strings
- Base Elo: 1500
- Check README.md for full results and research findings
