# topspin-lab

> **An Elo + ML pipeline for table tennis, validated on a tournament that didn't exist at training time.**

![Elo trajectories — 12,700 careers, 3 stars highlighted](docs/img/all_players_overlay.png)

---

## The hook

I trained a model on every ITTF singles match I could find — **157,836 matches stretching back to 1988**. Then I froze it.

Six weeks later, the ITTF World Team Championships started in London. The model had never seen a single match from it. I asked it to predict all 822 singles rubbers.

It got **617 of them right.**

| | n | Accuracy | AUC | Brier | LogLoss |
|---|---:|---:|---:|---:|---:|
| **London 2026 — model on truly unseen tournament** | 822 | **75.06%** | **0.8356** | 0.1666 | 0.5022 |

That is the only number on this page that matters, and it is reproducible from the code in this repo.

---

## Why I built it

A popular tweet claimed an AI model trained on 95,491 sports matches predicted outcomes with 85% accuracy. I was suspicious. Most "AI predicts X" claims share a tell: the model is scored on a re-split of its own training distribution. The 85% comes from data the model has *already statistically seen*.

I wanted to do the honest version. Pick a sport I find under-explored (table tennis), build the simplest possible model (Elo + a small Random Forest), and then evaluate it the way you would evaluate a forecaster — on an event that *physically did not exist* when the model stopped learning.

**Three principles, in order:**

1. **No look-ahead bias.** Every feature at time `t` uses only data with timestamp `< t`. Always. No exceptions. Treated as a correctness bug, not a performance issue.
2. **Time-based splits only.** Never random splits on time-series data.
3. **Reproducibility first.** Every metric on this page can be regenerated from raw data + code. No `.pkl` artifacts are committed.

---

## What the model sees

The whole pipeline is built around one rating system: **Elo**. Standard formulation, K=32, base 1500. Every match in chronological order updates two players' ratings.

That gives every player a trajectory. Here's Ma Long's, the most-rated player in our corpus:

![Ma Long — Elo trajectory over his career](docs/img/player_elo_ma_long.png)

White line is his full career. Green is his peak window — the top 20% of matches by smoothed Elo. The green dot is his career peak rating in our data: **~2600**, roughly 1100 points above a base-rate player. That is what "world-class" looks like to the model.

Now look at all 1,486 players with 50+ matches, with three stars overlaid:

![All players with three stars](docs/img/all_players_overlay.png)

Two things to notice. **First**, the cloud has a ceiling — almost nobody crosses ~2400. **Second**, the stars escape that ceiling on a remarkably consistent trajectory: each builds from base Elo through their first ~100 matches, then climbs. The model doesn't know the names. It just sees three players whose ratings keep going up.

---

## The test

The model was trained on every match up to **2026-03-16**. Frozen.

Then on **April 28**, the ITTF World Team Championships London 2026 started. The model was asked to predict every singles rubber. Two weeks and 822 predictions later:

[![London 2026 — interactive dashboard](docs/img/london_2026_dashboard.png)](experiments/london_2026_report.html)

> *Click the image to open the [full interactive HTML report](experiments/london_2026_report.html) — every match, every prediction, every actual outcome.*

**75.1% accuracy. 0.836 AUC. 617 correct calls out of 822.** Calibration is on the diagonal — when the model says 80%, the model is right ~80% of the time. High-confidence (≥75%) predictions land at **85.9%**.

---

## What surprised me

Three findings I didn't expect:

1. **Pure Elo alone gets 73.97%.** No ML, no features. Just rating difference. The Random Forest adds *one percentage point*. Most of the signal in table tennis lives in a single number computed by 1960s arithmetic.

2. **Opponent's recent form is ~2× more predictive than the player's own.** A player on a cold streak entering a match against a strong opponent is a clearer signal than a player on a hot streak. Strong opponents punish weakness more reliably than they reward strength.

3. **The 9-feature model is well-calibrated mid-range but under-confident at the extremes.** When it says 90%, reality is 95%. This is the opposite failure mode from most ML models, which over-confidently shout 90% when reality is 75%.

The first finding is the one that made the whole project feel honest. If a model trained on six features (Elo difference + recent form) is one point better than Elo alone, that *is* the news. Many published "AI sports prediction" results would have stopped at the headline and never published the comparison.

---

## Reproduce it

```bash
git clone https://github.com/Roni-quant/topspin-lab.git
cd topspin-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # fill in ITTF credentials (Windows: `copy`)
```

Then the pipeline, end-to-end:

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

# Regenerate the plots in docs/img/
python -m viz.make_all
```

Then open `experiments/london_2026_report.html`.

---

## How it works (60 seconds)

```
ITTF API   →  raw_matches  →  clean  →  Elo ratings  →  features  →  RF model
(scrape)      (Parquet)       (dedupe)  (sequential)    (form + Elo)  (predict)
```

Each stage is a separate Python module under `pipeline/`, reads Parquet, writes Parquet, and is idempotent. The Elo engine (`ratings/elo.py`) is plain Python — standard K=32, base 1500. The Random Forest uses 9 features:

| Feature | What it measures |
|---|---|
| `elo_difference` | Pre-match rating gap (signed: A − B) |
| `form_last_5_a/b`, `form_last_10_a/b` | Win rate over the most recent 5/10 matches |
| `form_7_days_a/b` | Win rate over the last 7 calendar days |
| `matches_last_7_a/b` | Match count over the last 7 days (workload) |

All features computed by walking each player's history forward. A player's first match has form features as `NaN` (filled with 0 at training time — explicit, not silent imputation). No random shuffling anywhere.

---

## Repository structure

```
topspin-lab/
├── pipeline/         # Numbered stages: fetch → clean → elo → features → train
├── ratings/          # Sequential Elo engine
├── experiments/      # London 2026 validation, retraining, HTML report
├── viz/              # Plot generators (writes docs/img/*.png)
├── docs/             # methodology.md, results.md, generated images
└── data/, models/    # Local artifacts (not committed — regenerate)
```

---

## Why this is not a Kaggle toy

| Concern | How it's handled |
|---|---|
| Look-ahead bias | Strict chronological processing; pre-match Elo captured before update; features use only past entries |
| Random vs time splits | Time-based splits at the calendar-year boundary; walk-forward validation in `pipeline/forward_test.py` |
| Cold-start players | Excluded from the headline metric (35 / 857 in London 2026); reported separately |
| Doubles vs singles | Doubles filtered at scrape time; Elo on individuals only |
| Calibration | Reliability tables in `docs/results.md`; mid-range well-calibrated, slight under-confidence at extremes |
| Reproducibility | Model artifact is *not* committed — `experiments/retrain_enhanced_rf.py` rebuilds it from features |
| Supply-chain hygiene | No `.pkl` in the repo; users regenerate locally. See `CONTRIBUTING.md`. |

---

## Known limitations

- No surface / equipment / ball-type modeling.
- K-factor not tuned by event tier.
- No team-rubber order modeling (in team formats, match order is a strategic choice).
- One tournament is one tournament — the 75% headline has a ~3% Wilson interval (treat it as "between 72% and 78%").

Full list in [`docs/methodology.md`](docs/methodology.md).

---

## Scope and intent

This repository is a **research and educational project**. It demonstrates a clean, leakage-free Elo + ML pipeline for predicting table-tennis outcomes and reports honest out-of-sample numbers. It is:

- **Not** a betting tool. There is no odds-API integration, no staking logic, no money flow.
- **Not** financial advice. Predicted probabilities are model outputs, not signals to act on.
- **Not** affiliated with the ITTF or any bookmaker.

Use it to study the methodology, reproduce the numbers, or extend the model.

---

## Going deeper

- [`docs/methodology.md`](docs/methodology.md) — design decisions, leakage discipline, why Elo, walk-forward validation, cold-start handling
- [`docs/results.md`](docs/results.md) — full metrics, calibration tables, per-category breakdown, feature importance, comparison to published baselines
- [`experiments/london_2026_report.html`](experiments/london_2026_report.html) — every prediction in the holdout tournament, sortable / filterable
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ground rules for PRs

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

Data scraped from the [ITTF results portal](https://results.ittf.link/). This project is not affiliated with or endorsed by the ITTF.
