# topspin-lab

> **Elo-based prediction for table tennis, validated on truly unseen data.**
> 75.06% accuracy / 0.836 AUC on the ITTF World Team Championships London 2026 — a tournament that did not exist when the model was trained.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A research pipeline for sequential Elo rating and machine-learning prediction of table-tennis match outcomes. Built around three principles:

1. **No look-ahead bias.** All features and ratings at time `t` depend only on data with timestamp `< t`. Time-based splits only, never random.
2. **Reproducibility first.** Every metric in this repository can be regenerated from raw data + code. No "trust me" model artifacts.
3. **Honest evaluation.** Headline numbers come from a tournament that did not exist at training time, not a re-split of the training distribution.

## Headline result

The model was trained on 157,836 matches up to 2026-03-16, then frozen. It was then asked to predict every singles rubber in the ITTF World Team Championships London 2026 (April 28 – May 10, 2026):

| | n | Accuracy | AUC | Brier | LogLoss |
|---|---:|---:|---:|---:|---:|
| **London 2026, enhanced RF (9 features)** | 822 | **75.06%** | **0.8356** | 0.1666 | 0.5022 |
| Pure Elo prior (no model) | 822 | 73.97% | 0.8333 | 0.1671 | 0.5024 |

Pure Elo alone carries ~95% of the signal. The full per-match breakdown is in [`experiments/london_2026_report.html`](experiments/london_2026_report.html) — open it in a browser to see every prediction, the confidence, the actual outcome, and where the model was wrong.

See [`docs/results.md`](docs/results.md) for the full numbers and [`docs/methodology.md`](docs/methodology.md) for the design decisions.

## How it works (in 60 seconds)

```
ITTF API   →  raw_matches  →  clean  →  Elo ratings  →  features  →  RF model
(scrape)      (Parquet)       (dedupe)  (sequential)    (form + Elo)  (predict)
```

Each stage is a separate Python module under `pipeline/`, reads Parquet, writes Parquet, and is idempotent. The Elo engine (`ratings/elo.py`) is plain Python — standard K=32, base 1500. The Random Forest uses 9 features: Elo difference plus recent-form and workload indicators.

## Repository structure

```
topspin-lab/
├── pipeline/         # Numbered stages: fetch → clean → elo → features → train
├── ratings/          # Sequential Elo engine
├── experiments/      # London 2026 validation, retraining, HTML report
├── docs/             # Methodology + results
├── mvp/              # Live odds + paper-trade prototype (separate from research)
└── data/, models/    # Local artifacts (not committed — regenerate)
```

## Quick start

```bash
git clone https://github.com/roni-quant/topspin-lab.git
cd topspin-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # fill in ITTF credentials
```

Then run the pipeline end-to-end:

```bash
# Scrape (slow on first run — uses ITTF API)
python -m pipeline.fetch_events
python -m pipeline.fetch_matches
python -m pipeline.merge_raw

# Build features and train
python -m pipeline.clean
python -m pipeline.compute_elo
python -m pipeline.generate_features_v2
python -m experiments.retrain_enhanced_rf

# Validate on the unseen tournament
python -m experiments.fetch_london_2026
python -m experiments.validate_london_2026
python -m experiments.build_london_report
```

Open `experiments/london_2026_report.html` to see the result.

## What's in the model

| Feature | What it measures |
|---|---|
| `elo_difference` | Pre-match rating gap (signed: A − B) |
| `form_last_5_a/b`, `form_last_10_a/b` | Win rate over the most recent 5/10 matches |
| `form_7_days_a/b` | Win rate over the last 7 calendar days (fatigue signal) |
| `matches_last_7_a/b` | Match count over the last 7 days (workload) |

All features are computed walking each player's history forward in time. A player's first match has form features as `NaN`. There is no random shuffling anywhere in the pipeline.

## Why this is not a Kaggle toy

| Concern | How it's handled |
|---|---|
| Look-ahead bias | Strict chronological processing; pre-match Elo captured before update; features use only past entries |
| Random vs time splits | Time-based splits at the calendar-year boundary; walk-forward validation in `pipeline/forward_test.py` |
| Cold-start players | Excluded from headline metric (35 / 857 in London 2026); reported separately |
| Doubles vs singles | Doubles filtered at scrape time; Elo on individuals only |
| Calibration | Reliability tables in `docs/results.md`; mid-range well-calibrated, slight under-confidence at extremes |
| Reproducibility | Model artifact is *not* committed — `experiments/retrain_enhanced_rf.py` rebuilds it from features |
| Supply-chain hygiene | No `.pkl` in the repo; users regenerate locally. See `CONTRIBUTING.md`. |

## Known limitations

- No surface / equipment / ball-type modeling.
- K-factor not tuned by event tier.
- No team-rubber order modeling (in team formats, match order is a strategic choice).
- One tournament is one tournament — the 75% headline has a ~3% Wilson interval.

Full list in [`docs/methodology.md`](docs/methodology.md).

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the ground rules (no leakage, time-based splits only, Parquet over CSV, every metric must be reproducible).

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

Data scraped from the [ITTF results portal](https://results.ittf.link/). This project is not affiliated with or endorsed by the ITTF.
