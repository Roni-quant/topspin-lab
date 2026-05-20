# Research log

Unstructured notes from building this. Kept rough on purpose - the polished version is in `methodology.md` + `results.md`.

## 2026-03-28 - scrape concurrency

ITTF Joomla API is slow. Sequential year-by-year fetch took 4 hours on first run. Parallelized per-year with `concurrent.futures` (default 8 workers). Down to 35 minutes. Watch for 429s; rate limiter at `pipeline/http.py` backs off exponentially.

Tests at `tests/test_fetch_matches_concurrent.py` are the only async-ish coverage in the repo.

## 2026-04-05 - pipeline complete, first numbers

Sequential Elo over 158,185 matches: ~6 minutes on this laptop. K=32, base=1500. Mean Elo settles around 1500 (as it must), std ~150.

First model results (5 baseline features: elo_diff + cumulative_matches + cumulative_wins for both players):
- LR: 69.42% acc, 0.7712 AUC
- RF: 70.35% acc, 0.7824 AUC

RF wins on both. Tried `min_samples_leaf=1` (overfit, holdout dropped), `max_depth=None` (overfit), `n_estimators=500` (no gain over 200). Settled on `n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2`.

## 2026-04-05 - recent form (v2 features)

Hypothesis: cumulative counts are too coarse. A player on a 7-match losing streak should look different from a player on a 7-match winning streak even if their career win rates match.

Added 8 features: form_last_5, form_last_10, form_7_days (win rate), matches_last_7 (workload). Both A and B variants.

Result:
- LR: 69.34% acc, 0.7805 AUC (+0.27% acc, +1.21% AUC)
- RF: 70.31% acc, 0.7829 AUC (-0.04% acc, +0.05% AUC)

Interpretation: RF was already getting most of this signal from cumulative counts via tree interactions. LR couldn't, because it's linear. The +1.21% AUC on LR is the strongest evidence that recent form matters in principle - RF just hides it.

Feature importance on RF v2:
- elo_difference: ~59%
- form_last_5_b (opponent): ~12%
- form_last_10_b (opponent): ~8%
- form_last_5_a (player): ~6%
- everything else: ~15% combined

Opponent form > player form. Counter-intuitive but stable: I re-ran with different random seeds and the ratio held.

## 2026-04-12 - things tried, things cut

Quick log of dead ends from a single afternoon of poking:

- **Per-event-tier K.** Tried K=24 for opens, K=40 for Grand Smashes. Walk-forward AUC barely moved (<0.001). Reverted.
- **K boosted for last 90 days.** Made hot players over-rated. Walk-forward stability collapsed post-major-tournaments. Reverted.
- **Head-to-head feature.** `h2h_win_rate_a_over_b` over prior meetings. Training AUC jumped; holdout AUC fell. Diagnosis: median pair has 1 prior meeting in the corpus. Sparse, high-variance. Cut.
- **Form window = 5 matches alone.** Strictly worse than 10. Kept both because the RF extracts some marginal signal from the combination.
- **Form last 30 days.** Tried as a wider window. Mostly redundant with form_last_10; small importance, no AUC change. Dropped.

## 2026-04-XX - calibration

Initial RF was over-confident at the tails. Predicted 95% → actual 85%. Investigated:
- More trees: no help.
- `class_weight='balanced'`: no help.
- Isotonic post-calibration: helped on validation but the holdout (London 2026) is small enough that the calibrator overfit. Reverted to raw probabilities.

After feature work, calibration shifted to slightly under-confident at the tails (95% → 95.1% actual at the top bin). Acceptable. Could revisit with a larger holdout.

## Open ideas (not pursued)

- **Surface / equipment.** Plastic vs celluloid era split (~2014 transition). Plausible regime change. Untested.
- **Bo-3 vs Bo-5 weighting.** All matches updated Elo with K=32 regardless of format. A Bo-5 result should arguably carry more information.
- **Team rubber order modeling.** In team formats, match order is a coach's strategic call. Currently ignored. Could be a feature: "rubber position in the tie."
- **Bayesian Elo with uncertainty.** Current Elo is a point estimate. A player with 1000 matches and a player with 10 both get a single number; they should have different rating variances.
- **Player-style embeddings.** Latent factors from past matchups. Probably needs more data than the ITTF corpus has.

## What broke during development

- Sequential Elo writes the same float twice per match (once as winner, once as loser update). Easy to off-by-one when capturing pre-match ratings. Caught it because the very first regression showed elo_diff = post-match, not pre-match. Fix: always capture before calling `process_match`.
- Doubles matches snuck in early - the ITTF API tags them as `MD` / `WD`. Scraper now filters at fetch time. Found because Elo for a player like Lin Yun-Ju was being updated by his doubles results, which made no sense individually.
- Cold-start handling: first version assigned base Elo (1500) to unseen players in the holdout and predicted normally. Headline accuracy dropped 3-4 percentage points without saying anything about the model. Now they're reported separately.
- Time-based split at the year boundary missed events that ran across New Year. Edge case, ~12 matches in the corpus. Fixed by splitting on the actual `match_date` cutoff, not year.
