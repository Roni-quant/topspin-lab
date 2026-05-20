# TODO

Things on the list, not in any particular order. PRs welcome on any of them.

## Methodology

- [ ] Bootstrap 95% CI for the 75.06% headline (Wilson interval is ~3% wide on n=822; show the code, not just the number).
- [ ] Tune K-factor properly. Current K=32 is conventional, not optimized. Try K ∈ {16, 24, 32, 40} on walk-forward AUC.
- [ ] Bayesian Elo with rating variance. A player with 10 matches and a player with 1000 should have different confidence intervals.
- [ ] Investigate surface / ball-type regime change (~2014 plastic ball transition). Walk-forward should be sensitive to it.
- [ ] Bo-3 vs Bo-5 K-factor split. Bo-5 results probably carry more information per match.

## Features (tried, may revisit)

- [ ] H2H feature with smoothing. Cut in v2 because most pairs meet ≤ 1 time. Try a Bayesian prior centered on Elo expectation.
- [ ] Tournament tier as a feature (Grand Smash > WTT > Continental). Currently uniform.
- [ ] Team-rubber order in team-format ties. Position 1 vs position 5 has different strategic context.

## Engineering

- [ ] GitHub Actions CI running `pytest tests/`. The repo has tests but no CI badge.
- [ ] Confirm ITTF ToS allows redistribution of `experiments/london_2026_matches.parquet`. If not, move behind a fetch script.
- [ ] Bundle a tiny test fixture (~1000 matches, all anonymized) so `make_all.py` runs without the full scrape. Useful for CI.
- [ ] Type hints across `pipeline/`. Currently spotty.
- [ ] `pyproject.toml` instead of plain `requirements.txt`.

## Viz

- [ ] Walk-forward rolling-accuracy line chart. Currently only static aggregate metrics.
- [ ] Per-day London 2026 accuracy timeline.
- [ ] Feature importance stability across walk-forward refits.
- [ ] Elo distribution histogram across all 12,700 players.

## Reproducibility

- [ ] Pin sklearn version in `requirements.txt`. Currently `scikit-learn>=1.4,<2`; the RF pickle protocol changed between minor versions before.
- [ ] Docker / devcontainer setup for one-command reproduction.

## Known issues / weirdness

- [ ] Some player IDs have inconsistent name spellings across years (transliteration drift). `pipeline/merge_raw.py` picks the most recent canonical name; older logs may reference older spellings.
- [ ] First-match form features are `NaN` and currently filled with 0 at training time. Should probably be a sentinel value the model treats as "no signal," not a falsy zero.
- [ ] Cold-start matches (≥1 player with no prior history) excluded from headline but the cutoff is binary. A player with 3 prior matches isn't really "warm." Could threshold instead.
