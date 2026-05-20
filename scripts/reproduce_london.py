"""One-shot reproduction: full pipeline -> London 2026 holdout -> HTML report -> plots.

Runs the canonical pipeline end-to-end and refuses to skip stages by default. Use
flags to resume from a partial run. Exits non-zero on any stage failure.

Usage:
    python scripts/reproduce_london.py                  # everything
    python scripts/reproduce_london.py --skip-scrape    # use existing data/raw/*
    python scripts/reproduce_london.py --only-validate  # data + Elo + features assumed present
    python scripts/reproduce_london.py --no-plots       # skip viz/make_all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRAPE_STAGES = [
    ("pipeline.fetch_events",   "Scrape ITTF event index"),
    ("pipeline.fetch_matches",  "Scrape per-year match records"),
    ("pipeline.merge_raw",      "Merge yearly batches"),
]
BUILD_STAGES = [
    ("pipeline.clean",                  "Clean + dedupe"),
    ("pipeline.compute_elo",            "Sequential Elo ratings"),
    ("pipeline.generate_features_v2",   "Build 9 features"),
    ("experiments.retrain_enhanced_rf", "Train enhanced RF on pre-London data"),
]
VALIDATE_STAGES = [
    ("experiments.fetch_london_2026",   "Fetch London 2026 matches"),
    ("experiments.validate_london_2026","Score model on the unseen tournament"),
    ("experiments.build_london_report", "Render HTML report"),
]


def run(module: str, label: str) -> None:
    print(f"\n>>> [{label}]  python -m {module}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n!!! {module} failed (exit {result.returncode}) after {dt:.1f}s", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"<<< [{label}] done in {dt:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip the ITTF scrape (use existing data/raw/* + raw_matches.parquet).")
    parser.add_argument("--only-validate", action="store_true",
                        help="Run only fetch/validate/report. Assumes Elo + features + model exist.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip viz.make_all at the end.")
    args = parser.parse_args()

    if args.only_validate:
        stages = VALIDATE_STAGES
    elif args.skip_scrape:
        stages = BUILD_STAGES + VALIDATE_STAGES
    else:
        stages = SCRAPE_STAGES + BUILD_STAGES + VALIDATE_STAGES

    t0 = time.time()
    for module, label in stages:
        run(module, label)

    if not args.no_plots:
        run("viz.make_all", "Regenerate plots")

    total = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Reproduction complete in {total/60:.1f} min")
    print("=" * 60)
    print("\n  HTML report: experiments/london_2026_report.html")
    print("  Plots:       docs/img/")
    print("  Headline:    see scoreboard printed by validate_london_2026 above\n")


if __name__ == "__main__":
    main()
