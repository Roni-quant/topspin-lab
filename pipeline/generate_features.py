"""Stage 4 — Feature Generation.

Reads matches_with_elo.parquet, generates prediction features using only
past data (no look-ahead), and outputs model_features.parquet.

Simplified features for efficiency:
- elo_difference: Elo_A - Elo_B before match
- cumulative_matches_a: Total matches played by player A up to this point
- cumulative_matches_b: Total matches played by player B up to this point
- cumulative_wins_a: Total wins by player A up to this point
- cumulative_wins_b: Total wins by player B up to this point

Target: winner_id == player_a_id (binary: 1 = player A wins, 0 = player B wins)

Usage:
    python -m pipeline.generate_features
"""

import logging
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

from pipeline.config import DATA_DIR
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import validate_schema

logger: logging.Logger | None = None

ELO_INPUT = DATA_DIR.parent / "matches_with_elo.parquet"
FEATURES_OUTPUT = DATA_DIR.parent / "model_features.parquet"

FEATURES_DTYPES = {
    "match_date": "datetime64[ns]",
    "player_a_id": "Int64",
    "player_b_id": "Int64",
    "elo_difference": "float64",
    "cumulative_matches_a": "Int64",
    "cumulative_matches_b": "Int64",
    "cumulative_wins_a": "Int64",
    "cumulative_wins_b": "Int64",
    "target": "int8",
}


def _load_elo_matches() -> pd.DataFrame:
    """Load matches_with_elo.parquet."""
    if not ELO_INPUT.exists():
        raise FileNotFoundError(f"matches_with_elo.parquet not found at {ELO_INPUT}")

    df = pd.read_parquet(ELO_INPUT)
    log_structured(
        logger, logging.INFO,
        f"Loaded {len(df)} Elo-rated matches from {ELO_INPUT.name}",
        entity_type="file", entity_id=str(ELO_INPUT.name),
    )
    return df


def _compute_elo_difference(df: pd.DataFrame) -> pd.Series:
    """Compute Elo difference (A - B) before match."""
    return df["elo_a_before"] - df["elo_b_before"]


def _compute_cumulative_stats(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute cumulative match counts and wins for each player (expanding window).

    Returns:
        (cumulative_matches_a, cumulative_matches_b, cumulative_wins_a, cumulative_wins_b)
    """
    matches_a = []
    matches_b = []
    wins_a = []
    wins_b = []

    player_match_counts = defaultdict(int)
    player_win_counts = defaultdict(int)

    for idx, (_, row) in enumerate(df.iterrows()):
        player_a_id = row["player_a_id"]
        player_b_id = row["player_b_id"]
        winner_id = row["winner_id"]

        # Get counts before this match
        matches_a.append(player_match_counts[player_a_id])
        matches_b.append(player_match_counts[player_b_id])
        wins_a.append(player_win_counts[player_a_id])
        wins_b.append(player_win_counts[player_b_id])

        # Update counts after this match
        player_match_counts[player_a_id] += 1
        player_match_counts[player_b_id] += 1
        if winner_id == player_a_id:
            player_win_counts[player_a_id] += 1
        else:
            player_win_counts[player_b_id] += 1

        if (idx + 1) % 20000 == 0:
            log_structured(
                logger, logging.INFO,
                f"Processed {idx + 1}/{len(df)} matches for cumulative stats",
                entity_type="dataset",
            )

    return (
        pd.Series(matches_a, index=df.index, dtype="Int64"),
        pd.Series(matches_b, index=df.index, dtype="Int64"),
        pd.Series(wins_a, index=df.index, dtype="Int64"),
        pd.Series(wins_b, index=df.index, dtype="Int64"),
    )


def _compute_target(df: pd.DataFrame) -> pd.Series:
    """Compute binary target: 1 if player_a wins, 0 if player_b wins."""
    return (df["winner_id"] == df["player_a_id"]).astype("int8")


def _generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all features from Elo-rated matches."""
    total_matches = len(df)

    log_structured(
        logger, logging.INFO,
        f"Computing features for {total_matches} matches...",
        entity_type="dataset",
    )

    # Elo difference
    df["elo_difference"] = _compute_elo_difference(df)

    # Cumulative statistics
    log_structured(logger, logging.INFO, "Computing cumulative match/win statistics...", entity_type="dataset")
    df["cumulative_matches_a"], df["cumulative_matches_b"], \
    df["cumulative_wins_a"], df["cumulative_wins_b"] = _compute_cumulative_stats(df)

    # Target
    df["target"] = _compute_target(df)

    log_structured(
        logger, logging.INFO,
        "All features computed successfully",
        entity_type="dataset",
    )

    return df


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder feature columns."""
    cols = list(FEATURES_DTYPES.keys())
    return df[cols].copy()


def _print_certification(df: pd.DataFrame) -> None:
    """Print and log the Stage 4 certification report."""
    row_count = len(df)

    # Feature statistics
    stats_lines = [
        "",
        "=" * 60,
        "  STAGE 4 CERTIFICATION — Feature Generation",
        "=" * 60,
        f"  Total samples:           {row_count:,}",
        "",
        "  Target distribution:",
    ]

    target_counts = df["target"].value_counts().sort_index()
    for target_val, count in target_counts.items():
        pct = 100.0 * count / row_count
        label = "Player A wins" if target_val == 1 else "Player B wins"
        stats_lines.append(f"    {label:20} {count:,} ({pct:.1f}%)")

    stats_lines += [
        "",
        "  Feature statistics:",
        f"    elo_difference:        mean={df['elo_difference'].mean():7.2f}, std={df['elo_difference'].std():7.2f}",
        f"    cum_matches_a:         mean={df['cumulative_matches_a'].mean():7.1f}, std={df['cumulative_matches_a'].std():7.1f}",
        f"    cum_matches_b:         mean={df['cumulative_matches_b'].mean():7.1f}, std={df['cumulative_matches_b'].std():7.1f}",
        f"    cum_wins_a:            mean={df['cumulative_wins_a'].mean():7.1f}, std={df['cumulative_wins_a'].std():7.1f}",
        f"    cum_wins_b:            mean={df['cumulative_wins_b'].mean():7.1f}, std={df['cumulative_wins_b'].std():7.1f}",
        "=" * 60,
    ]

    report_text = "\n".join(stats_lines)
    print(report_text)

    log_structured(
        logger, logging.INFO,
        report_text,
        entity_type="certification",
        status="ok",
    )


def generate_features() -> None:
    """Run the full Stage 4 feature generation pipeline."""
    global logger
    logger = setup_stage_logger("generate_features")

    log_structured(logger, logging.INFO, "Stage 4 — generate_features starting", status="start")

    # 1. Load Elo-rated matches
    df = _load_elo_matches()

    # 2. Generate all features
    df = _generate_all_features(df)

    # 3. Select feature columns
    df = _select_feature_columns(df)

    # 4. Validate schema
    df = validate_schema(df, FEATURES_DTYPES)

    # 5. Write output
    df.to_parquet(FEATURES_OUTPUT, index=False)
    log_structured(
        logger, logging.INFO,
        f"Wrote {len(df)} rows to {FEATURES_OUTPUT.name}",
        entity_type="file", entity_id=str(FEATURES_OUTPUT.name),
    )

    # 6. Certification
    _print_certification(df)

    log_structured(logger, logging.INFO, "Stage 4 — generate_features complete", status="done")


if __name__ == "__main__":
    try:
        generate_features()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("generate_features")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
