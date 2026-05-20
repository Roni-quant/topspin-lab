"""Stage 2 — Data Cleaning & Normalization.

Reads raw_matches.parquet, validates data quality, removes duplicates,
normalizes player IDs, and outputs matches_clean.parquet.

Usage:
    python -m pipeline.clean
"""

import logging
import sys

import pandas as pd

from pipeline.config import DATA_DIR, MERGED_OUTPUT
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import validate_schema

logger: logging.Logger | None = None

CLEAN_OUTPUT = DATA_DIR.parent / "matches_clean.parquet"

# Expected schema for raw_matches.parquet
CLEAN_DTYPES = {
    "match_date": "datetime64[ns]",
    "player_a_id": "Int64",
    "player_b_id": "Int64",
    "winner_id": "Int64",
}


def _load_raw_matches() -> pd.DataFrame:
    """Load raw_matches.parquet."""
    if not MERGED_OUTPUT.exists():
        raise FileNotFoundError(f"raw_matches.parquet not found at {MERGED_OUTPUT}")

    df = pd.read_parquet(MERGED_OUTPUT)
    log_structured(
        logger, logging.INFO,
        f"Loaded {len(df)} raw matches from {MERGED_OUTPUT.name}",
        entity_type="file", entity_id=str(MERGED_OUTPUT.name),
    )
    return df


def _validate_basic(df: pd.DataFrame) -> tuple[int, int]:
    """Validate basic data integrity.

    Returns (null_player_count, null_winner_count).
    """
    null_a = int(df["player_a_id"].isna().sum())
    null_b = int(df["player_b_id"].isna().sum())
    null_winner = int(df["winner_id"].isna().sum())

    if null_a > 0 or null_b > 0:
        msg = f"Found {null_a} null player_a_id, {null_b} null player_b_id — dropping rows"
        log_structured(logger, logging.WARNING, msg, entity_type="dataset")
        df = df.dropna(subset=["player_a_id", "player_b_id"])

    if null_winner > 0:
        msg = f"Found {null_winner} null winner_id — this should not happen!"
        log_structured(logger, logging.ERROR, msg, entity_type="dataset")
        raise ValueError(msg)

    return len(df[df["player_a_id"].isna() | df["player_b_id"].isna()]), null_winner


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicates (same players, same date, same winner).

    Returns (deduplicated_df, duplicate_count).
    """
    subset = ["match_date", "player_a_id", "player_b_id", "winner_id"]
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    dup_count = pre_dedup - len(df)

    if dup_count > 0:
        log_structured(
            logger, logging.INFO,
            f"Removed {dup_count} exact duplicate matches",
            entity_type="dataset",
        )

    return df, dup_count


def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort matches by date (required for Elo sequential processing)."""
    df = df.sort_values("match_date", kind="mergesort").reset_index(drop=True)
    log_structured(
        logger, logging.INFO,
        "Sorted matches by date (required for sequential Elo processing)",
        entity_type="dataset",
    )
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder the minimal schema for downstream use."""
    cols = ["match_date", "player_a_id", "player_b_id", "winner_id"]
    df = df[cols].copy()
    return df


def _print_certification(
    df: pd.DataFrame,
    null_count: int,
    dup_count: int,
) -> None:
    """Print and log the Stage 2 certification report."""
    row_count = len(df)
    unique_players = set(
        pd.concat([df["player_a_id"], df["player_b_id"]]).dropna().unique()
    )
    year_distribution = (
        df["match_date"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report_lines = [
        "",
        "=" * 60,
        "  STAGE 2 CERTIFICATION — Data Cleaning",
        "=" * 60,
        f"  Final row count:         {row_count:,}",
        f"  Unique players:          {len(unique_players):,}",
        f"  Rows with null players:  {null_count}",
        f"  Duplicates removed:      {dup_count:,}",
        "",
        "  Year distribution:",
    ]
    for year, cnt in sorted(year_distribution.items()):
        report_lines.append(f"    {year}: {cnt:,}")

    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)

    log_structured(
        logger, logging.INFO,
        report_text,
        entity_type="certification",
        status="ok",
    )


def clean() -> None:
    """Run the full Stage 2 data cleaning pipeline."""
    global logger
    logger = setup_stage_logger("clean")

    log_structured(logger, logging.INFO, "Stage 2 — clean starting", status="start")

    # 1. Load raw data
    df = _load_raw_matches()

    # 2. Validate and drop null players (but fail on null winner)
    null_count, _ = _validate_basic(df)

    # 3. Remove duplicates
    df, dup_count = _remove_duplicates(df)

    # 4. Sort by date
    df = _sort_by_date(df)

    # 5. Select minimal schema
    df = _select_columns(df)

    # 6. Validate final schema
    df = validate_schema(df, CLEAN_DTYPES)

    # 7. Write output
    df.to_parquet(CLEAN_OUTPUT, index=False)
    log_structured(
        logger, logging.INFO,
        f"Wrote {len(df)} rows to {CLEAN_OUTPUT.name}",
        entity_type="file", entity_id=str(CLEAN_OUTPUT.name),
    )

    # 8. Certification
    _print_certification(df, null_count, dup_count)

    log_structured(logger, logging.INFO, "Stage 2 — clean complete", status="done")


if __name__ == "__main__":
    try:
        clean()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("clean")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
