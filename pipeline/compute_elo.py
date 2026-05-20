"""Stage 3 — Elo Rating Engine.

Reads matches_clean.parquet, initializes players at base Elo (1500),
processes matches in strict chronological order, and outputs
matches_with_elo.parquet with pre-match ratings.

Usage:
    python -m pipeline.compute_elo
"""

import logging
import sys

import pandas as pd

from pipeline.config import DATA_DIR
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import validate_schema
from ratings.elo import EloRatingEngine, EloConfig

logger: logging.Logger | None = None

CLEAN_INPUT = DATA_DIR.parent / "matches_clean.parquet"
ELO_OUTPUT = DATA_DIR.parent / "matches_with_elo.parquet"

ELO_DTYPES = {
    "match_date": "datetime64[ns]",
    "player_a_id": "Int64",
    "player_b_id": "Int64",
    "elo_a_before": "float64",
    "elo_b_before": "float64",
    "winner_id": "Int64",
}


def _load_clean_matches() -> pd.DataFrame:
    """Load matches_clean.parquet."""
    if not CLEAN_INPUT.exists():
        raise FileNotFoundError(f"matches_clean.parquet not found at {CLEAN_INPUT}")

    df = pd.read_parquet(CLEAN_INPUT)
    log_structured(
        logger, logging.INFO,
        f"Loaded {len(df)} clean matches from {CLEAN_INPUT.name}",
        entity_type="file", entity_id=str(CLEAN_INPUT.name),
    )
    return df


def _compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Process matches in chronological order and compute Elo ratings.

    Args:
        df: DataFrame with columns [match_date, player_a_id, player_b_id, winner_id]
           Must be sorted by match_date.

    Returns:
        DataFrame with added columns [elo_a_before, elo_b_before]
    """
    # Ensure matches are sorted by date
    df = df.sort_values("match_date", kind="mergesort").reset_index(drop=True)

    # Initialize Elo engine
    config = EloConfig(base_rating=1500.0, k_factor=32.0)
    engine = EloRatingEngine(config)

    elo_a_before_list = []
    elo_b_before_list = []

    total_matches = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 10000 == 0:
            log_structured(
                logger, logging.INFO,
                f"Processing match {idx + 1}/{total_matches}",
                entity_type="dataset", progress=idx + 1,
            )

        player_a_id = int(row["player_a_id"])
        player_b_id = int(row["player_b_id"])
        winner_id = int(row["winner_id"])

        # Get pre-match ratings and update them
        elo_a, elo_b = engine.process_match(player_a_id, player_b_id, winner_id)
        elo_a_before_list.append(elo_a)
        elo_b_before_list.append(elo_b)

    # Add new columns
    df = df.copy()
    df["elo_a_before"] = elo_a_before_list
    df["elo_b_before"] = elo_b_before_list

    log_structured(
        logger, logging.INFO,
        f"Elo ratings computed for {total_matches} matches",
        entity_type="dataset",
    )

    return df


def _print_certification(df: pd.DataFrame) -> None:
    """Print and log the Stage 3 certification report."""
    row_count = len(df)

    # Elo statistics
    elo_a_stats = df["elo_a_before"].describe()
    elo_b_stats = df["elo_b_before"].describe()

    # Player statistics
    unique_players = set(
        pd.concat([df["player_a_id"], df["player_b_id"]]).dropna().unique()
    )

    report_lines = [
        "",
        "=" * 60,
        "  STAGE 3 CERTIFICATION — Elo Rating Engine",
        "=" * 60,
        f"  Matches processed:       {row_count:,}",
        f"  Unique players:          {len(unique_players):,}",
        "",
        "  Player A Elo statistics:",
        f"    Min:      {elo_a_stats['min']:.1f}",
        f"    Mean:     {elo_a_stats['mean']:.1f}",
        f"    Max:      {elo_a_stats['max']:.1f}",
        f"    Std Dev:  {elo_a_stats['std']:.1f}",
        "",
        "  Player B Elo statistics:",
        f"    Min:      {elo_b_stats['min']:.1f}",
        f"    Mean:     {elo_b_stats['mean']:.1f}",
        f"    Max:      {elo_b_stats['max']:.1f}",
        f"    Std Dev:  {elo_b_stats['std']:.1f}",
        "",
        f"  Base Elo:                1500.0",
        f"  K-factor:                32.0",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    log_structured(
        logger, logging.INFO,
        report_text,
        entity_type="certification",
        status="ok",
    )


def compute_elo() -> None:
    """Run the full Stage 3 Elo rating computation."""
    global logger
    logger = setup_stage_logger("compute_elo")

    log_structured(logger, logging.INFO, "Stage 3 — compute_elo starting", status="start")

    # 1. Load clean matches
    df = _load_clean_matches()

    # 2. Compute Elo ratings (sequential, chronological)
    df = _compute_elo_ratings(df)

    # 3. Validate schema
    df = validate_schema(df, ELO_DTYPES)

    # 4. Write output
    df.to_parquet(ELO_OUTPUT, index=False)
    log_structured(
        logger, logging.INFO,
        f"Wrote {len(df)} rows to {ELO_OUTPUT.name}",
        entity_type="file", entity_id=str(ELO_OUTPUT.name),
    )

    # 5. Certification
    _print_certification(df)

    log_structured(logger, logging.INFO, "Stage 3 — compute_elo complete", status="done")


if __name__ == "__main__":
    try:
        compute_elo()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("compute_elo")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
