"""Stage 4 — Merge raw yearly match batches into a single raw_matches.parquet.

Concatenates all matches_{year}.parquet files, filters out dropped matches,
joins player names from players.parquet for consistency, deduplicates by
match_key, sorts by match_date.

Usage:
    python -m pipeline.merge_raw
"""

import json
import logging
import sys

import pandas as pd

from pipeline.config import (
    DATA_DIR,
    DUPLICATE_RATIO_THRESHOLD,
    MERGED_OUTPUT,
)
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import (
    MATCH_COLUMNS_MERGED,
    MATCH_DTYPES_MERGED,
    validate_schema,
)

# Module-level logger; initialised inside merge() via setup_stage_logger.
logger: logging.Logger | None = None

# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------


def _load_yearly_batches() -> pd.DataFrame:
    """Load and concatenate all matches_{year}.parquet files from DATA_DIR."""
    parquet_files = sorted(DATA_DIR.glob("matches_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No matches_*.parquet files found in {DATA_DIR}"
        )

    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        log_structured(logger, logging.INFO, f"loaded {len(df)} rows",
                       entity_type="file", entity_id=pf.name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log_structured(logger, logging.INFO,
                   f"concatenated {len(frames)} files, {len(combined)} total rows",
                   entity_type="dataset")
    return combined


def _load_players() -> pd.DataFrame | None:
    """Load players.parquet if it exists; return None otherwise."""
    players_path = DATA_DIR / "players.parquet"
    if not players_path.exists():
        log_structured(logger, logging.INFO,
                       "player name join will be skipped",
                       entity_type="file", entity_id="players.parquet",
                       status="skipped", reason="not_found")
        return None

    players = pd.read_parquet(players_path)
    log_structured(logger, logging.INFO, f"loaded {len(players)} players",
                   entity_type="file", entity_id="players.parquet")
    return players


def _join_player_names(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Override player_a_name / player_b_name with canonical names from players.parquet.

    If a player_id is not found in the players table, keep the existing name.
    """
    name_map = players.set_index("player_id")["player_name"]

    # Player A
    mapped_a = df["player_a_id"].map(name_map)
    df["player_a_name"] = mapped_a.fillna(df["player_a_name"])

    # Player B
    mapped_b = df["player_b_id"].map(name_map)
    df["player_b_name"] = mapped_b.fillna(df["player_b_name"])

    updated_a = mapped_a.notna().sum()
    updated_b = mapped_b.notna().sum()
    log_structured(logger, logging.INFO,
                   f"player name join: {updated_a} A-names, {updated_b} B-names updated",
                   entity_type="dataset")

    return df


def _print_certification(
    df: pd.DataFrame,
    dropped_count: int,
    duplicate_count: int,
) -> None:
    """Print and log the Stage 4 certification report."""
    row_count = len(df)

    unique_players = set(
        pd.concat([df["player_a_id"], df["player_b_id"]]).dropna().unique()
    )
    unique_events = df["event_id"].nunique()
    winner_null_count = int(df["winner_id"].isna().sum())

    category_split = df["category"].value_counts().to_dict()
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
        "  STAGE 4 CERTIFICATION — merge_raw",
        "=" * 60,
        f"  Final row count:           {row_count:,}",
        f"  Unique players:            {len(unique_players):,}",
        f"  Unique events:             {unique_events:,}",
        f"  Duplicate match_key removed: {duplicate_count:,}",
        f"  Winner null count:         {winner_null_count} (expect 0)",
        f"  Dropped matches (pre-filter): {dropped_count:,}",
        "",
        "  Category split:",
    ]
    for cat, cnt in sorted(category_split.items()):
        report_lines.append(f"    {cat}: {cnt:,}")

    report_lines.append("")
    report_lines.append("  Year distribution:")
    for year, cnt in sorted(year_distribution.items()):
        report_lines.append(f"    {year}: {cnt:,}")

    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)

    log_structured(
        logger, logging.INFO,
        json.dumps({
            "final_row_count": row_count,
            "unique_players": len(unique_players),
            "unique_events": unique_events,
            "duplicate_match_key_removed": duplicate_count,
            "winner_null_count": winner_null_count,
            "dropped_matches_total": dropped_count,
            "category_split": category_split,
            "year_distribution": {str(k): v for k, v in year_distribution.items()},
        }),
        entity_type="certification",
        status="ok",
    )


def merge() -> None:
    """Run the full Stage 4 merge pipeline."""
    global logger
    logger = setup_stage_logger("merge")

    # 1. Load all yearly batches
    combined = _load_yearly_batches()

    # 2. Count and filter dropped matches (drop_reason IS NOT NULL)
    if "drop_reason" in combined.columns:
        dropped_mask = combined["drop_reason"].notna()
        dropped_count = int(dropped_mask.sum())
        log_structured(logger, logging.INFO,
                       f"filtering out {dropped_count} dropped matches",
                       entity_type="dataset")
        df = combined[~dropped_mask].copy()
    else:
        dropped_count = 0
        df = combined.copy()

    log_structured(logger, logging.INFO,
                   f"{len(df)} matches after drop filter",
                   entity_type="dataset")

    # 3. Sort by match_date
    df = df.sort_values("match_date", kind="mergesort").reset_index(drop=True)

    # 4. Deduplicate by match_key (keep first after sort)
    pre_dedup_count = len(df)
    df = df.drop_duplicates(subset="match_key", keep="first").reset_index(drop=True)
    duplicate_count = pre_dedup_count - len(df)

    log_structured(logger, logging.INFO,
                   f"removed {duplicate_count} duplicate match_keys",
                   entity_type="dataset")

    # 5. Check duplicate ratio threshold
    if pre_dedup_count > 0:
        duplicate_ratio = duplicate_count / pre_dedup_count
        if duplicate_ratio > DUPLICATE_RATIO_THRESHOLD:
            msg = (
                f"Duplicate ratio {duplicate_ratio:.4f} ({duplicate_count}/{pre_dedup_count}) "
                f"exceeds threshold {DUPLICATE_RATIO_THRESHOLD}. "
                f"This indicates a pipeline bug."
            )
            log_structured(logger, logging.ERROR, msg,
                           entity_type="dataset", status="error",
                           reason="duplicate_ratio_exceeded")
            raise RuntimeError(msg)

    # 6. Join player names from players.parquet
    players = _load_players()
    if players is not None:
        df = _join_player_names(df, players)

    # 7. Select merged columns (drop drop_reason) and validate schema
    df = df[MATCH_COLUMNS_MERGED].copy()
    df = validate_schema(df, MATCH_DTYPES_MERGED)

    # 8. Write output
    df.to_parquet(MERGED_OUTPUT, index=False)
    log_structured(logger, logging.INFO,
                   f"wrote {len(df)} rows to {MERGED_OUTPUT}",
                   entity_type="file", entity_id=str(MERGED_OUTPUT.name))

    # 9. Print certification
    _print_certification(df, dropped_count, duplicate_count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        merge()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("merge")
        log_structured(logger, logging.ERROR, str(exc),
                       entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
