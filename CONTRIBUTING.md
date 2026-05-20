# Contributing

Thanks for your interest. This is a research repository — the goal is rigorous, reproducible work on Elo-based prediction in table tennis. Contributions that improve correctness, methodology, or clarity are welcome.

## Ground rules

1. **No data leakage.** All features and ratings must use only past data. Time-based splits only — never random splits on time-series.
2. **Strict chronological order.** Matches are processed in `match_date` order; never reorder.
3. **Parquet over CSV.** All datasets stored as Parquet for schema safety.
4. **Pin your assumptions.** If your change depends on a sklearn / pandas version, say so.
5. **Reproducibility is the bar.** Any quoted metric must be reproducible from the code and a reproducible data fetch.

## Development setup

```bash
git clone https://github.com/roni-quant/topspin-lab.git
cd topspin-lab
python -m venv .venv
.venv/Scripts/activate  # Windows; on Unix: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in ITTF credentials if you want to re-scrape
```

## Running the full pipeline

```bash
python -m pipeline.fetch_events          # Stage 1 — event index
python -m pipeline.fetch_matches         # Stage 2 — match scraper
python -m pipeline.merge_raw             # Stage 3 — merge yearly batches
python -m pipeline.clean                 # Stage 4 — clean + dedupe
python -m pipeline.compute_elo           # Stage 5 — Elo ratings
python -m pipeline.generate_features_v2  # Stage 6 — features
python -m experiments.retrain_enhanced_rf  # Train RF v2 model
python -m experiments.validate_london_2026 # Holdout validation
```

## Testing

```bash
pytest mvp/tests/ -v
```

## Submitting a PR

1. Open an issue first for non-trivial changes — saves churn.
2. Keep PRs focused. One concern per PR.
3. Include test coverage for new logic.
4. Document any methodology changes in `docs/methodology.md`.
5. If you change a published metric, also update `docs/results.md` and the README headline numbers.

## Reviewing PRs (maintainer note)

**Inspect binary diffs carefully.** PRs that modify `.pkl`, `.joblib`, or `.parquet` files can carry executable payloads (pickle deserialization = arbitrary code execution). Either:

- Reject binary changes in PRs (regenerate locally from the contributor's code change), or
- Inspect the binary contents before running any script against the PR branch.

The repo intentionally does not ship `models/*.pkl` in git — users regenerate via `experiments/retrain_enhanced_rf.py`. Keep it that way.

## Style

- Python: `ruff` for linting, `ruff format` for formatting. No strict line length enforced, but stay reasonable.
- Imports: stdlib → third-party → local.
- Type hints encouraged where they aid readability; not enforced.

## Reporting bugs

Open an issue with: minimum reproducible example, what you expected, what happened, and your environment (`python --version`, `pip freeze | grep -E "pandas|scikit|numpy"`).
