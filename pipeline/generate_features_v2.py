"""Stage 4b — Enhanced Feature Generation with Recent Form.

Reads matches_with_elo.parquet, generates improved prediction features including:
- Elo difference
- Recent form (last 5 matches, last 10 matches)
- Recent win rates (last 7 days, last 30 days)
- Cumulative experience (backup features)

All features use only past data (no look-ahead).

Usage:
    python -m pipeline.generate_features_v2
"""

import logging
import sys
from collections import defaultdict
from datetime import timedelta

import pandas as pd
import numpy as np

from pipeline.config import DATA_DIR
from pipeline.log import log_structured, setup_stage_logger

logger: logging.Logger | None = None

ELO_INPUT = DATA_DIR.parent / "matches_with_elo.parquet"
FEATURES_OUTPUT = DATA_DIR.parent / "model_features.parquet"

FEATURES_DTYPES = {
    "match_date": "datetime64[ns]",
    "player_a_id": "Int64",
    "player_b_id": "Int64",
    "elo_difference": "float64",
    "form_last_5_a": "float64",  # Win rate last 5 matches
    "form_last_10_a": "float64",  # Win rate last 10 matches
    "form_last_5_b": "float64",
    "form_last_10_b": "float64",
    "form_7_days_a": "float64",  # Win rate last 7 days
    "form_7_days_b": "float64",
    "matches_last_7_a": "Int64",  # Match count last 7 days
    "matches_last_7_b": "Int64",
    "cumulative_matches_a": "Int64",
    "cumulative_wins_a": "Int64",
    "cumulative_matches_b": "Int64",
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


def _compute_recent_form_and_cumulative(df: pd.DataFrame) -> tuple:
    """Compute recent form (last N matches, last N days) + cumulative stats.

    Returns all feature series.
    """
    form_last_5_a = []
    form_last_10_a = []
    form_last_5_b = []
    form_last_10_b = []
    form_7_days_a = []
    form_7_days_b = []
    matches_7_days_a = []
    matches_7_days_b = []
    cum_matches_a = []
    cum_matches_b = []
    cum_wins_a = []
    cum_wins_b = []

    # Track match history per player (in order)
    player_matches = defaultdict(list)  # player_id -> [(date, is_win), ...]
    player_match_counts = defaultdict(int)
    player_win_counts = defaultdict(int)

    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        player_a_id = int(row["player_a_id"])
        player_b_id = int(row["player_b_id"])
        winner_id = int(row["winner_id"])
        match_date = row["match_date"]

        # ===== RECENT FORM: Last N matches =====
        # Get last 5 and 10 matches for each player
        hist_a = player_matches[player_a_id]
        hist_b = player_matches[player_b_id]

        # Form A: last 5 matches
        if len(hist_a) >= 5:
            recent_5_a = hist_a[-5:]
            form_last_5_a.append(np.mean([w for _, w in recent_5_a]))
        elif len(hist_a) > 0:
            form_last_5_a.append(np.mean([w for _, w in hist_a]))
        else:
            form_last_5_a.append(np.nan)

        # Form A: last 10 matches
        if len(hist_a) >= 10:
            recent_10_a = hist_a[-10:]
            form_last_10_a.append(np.mean([w for _, w in recent_10_a]))
        elif len(hist_a) > 0:
            form_last_10_a.append(np.mean([w for _, w in hist_a]))
        else:
            form_last_10_a.append(np.nan)

        # Form B: last 5 matches
        if len(hist_b) >= 5:
            recent_5_b = hist_b[-5:]
            form_last_5_b.append(np.mean([w for _, w in recent_5_b]))
        elif len(hist_b) > 0:
            form_last_5_b.append(np.mean([w for _, w in hist_b]))
        else:
            form_last_5_b.append(np.nan)

        # Form B: last 10 matches
        if len(hist_b) >= 10:
            recent_10_b = hist_b[-10:]
            form_last_10_b.append(np.mean([w for _, w in recent_10_b]))
        elif len(hist_b) > 0:
            form_last_10_b.append(np.mean([w for _, w in hist_b]))
        else:
            form_last_10_b.append(np.nan)

        # ===== RECENT FORM: Last 7 days =====
        cutoff_7_days = match_date - timedelta(days=7)

        hist_a_7d = [w for d, w in hist_a if d >= cutoff_7_days]
        hist_b_7d = [w for d, w in hist_b if d >= cutoff_7_days]

        form_7_days_a.append(np.mean(hist_a_7d) if hist_a_7d else np.nan)
        form_7_days_b.append(np.mean(hist_b_7d) if hist_b_7d else np.nan)
        matches_7_days_a.append(len(hist_a_7d))
        matches_7_days_b.append(len(hist_b_7d))

        # ===== CUMULATIVE STATS =====
        cum_matches_a.append(player_match_counts[player_a_id])
        cum_matches_b.append(player_match_counts[player_b_id])
        cum_wins_a.append(player_win_counts[player_a_id])
        cum_wins_b.append(player_win_counts[player_b_id])

        # ===== UPDATE HISTORY =====
        is_win_a = 1.0 if winner_id == player_a_id else 0.0
        is_win_b = 1.0 if winner_id == player_b_id else 0.0

        player_matches[player_a_id].append((match_date, is_win_a))
        player_matches[player_b_id].append((match_date, is_win_b))

        player_match_counts[player_a_id] += 1
        player_match_counts[player_b_id] += 1
        player_win_counts[player_a_id] += int(is_win_a)
        player_win_counts[player_b_id] += int(is_win_b)

        if (idx + 1) % 20000 == 0:
            log_structured(
                logger, logging.INFO,
                f"Computed features for {idx + 1}/{total} matches",
                entity_type="dataset",
            )

    return (
        pd.Series(form_last_5_a, index=df.index, dtype="float64"),
        pd.Series(form_last_10_a, index=df.index, dtype="float64"),
        pd.Series(form_last_5_b, index=df.index, dtype="float64"),
        pd.Series(form_last_10_b, index=df.index, dtype="float64"),
        pd.Series(form_7_days_a, index=df.index, dtype="float64"),
        pd.Series(form_7_days_b, index=df.index, dtype="float64"),
        pd.Series(matches_7_days_a, index=df.index, dtype="Int64"),
        pd.Series(matches_7_days_b, index=df.index, dtype="Int64"),
        pd.Series(cum_matches_a, index=df.index, dtype="Int64"),
        pd.Series(cum_matches_b, index=df.index, dtype="Int64"),
        pd.Series(cum_wins_a, index=df.index, dtype="Int64"),
        pd.Series(cum_wins_b, index=df.index, dtype="Int64"),
    )


def _compute_target(df: pd.DataFrame) -> pd.Series:
    """Compute binary target: 1 if player_a wins, 0 if player_b wins."""
    return (df["winner_id"] == df["player_a_id"]).astype("int8")


def _generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all features from Elo-rated matches."""
    total_matches = len(df)

    log_structured(
        logger, logging.INFO,
        f"Computing enhanced features for {total_matches} matches...",
        entity_type="dataset",
    )

    # Elo difference
    df["elo_difference"] = _compute_elo_difference(df)

    # Recent form + cumulative
    (
        df["form_last_5_a"],
        df["form_last_10_a"],
        df["form_last_5_b"],
        df["form_last_10_b"],
        df["form_7_days_a"],
        df["form_7_days_b"],
        df["matches_last_7_a"],
        df["matches_last_7_b"],
        df["cumulative_matches_a"],
        df["cumulative_matches_b"],
        df["cumulative_wins_a"],
        df["cumulative_wins_b"],
    ) = _compute_recent_form_and_cumulative(df)

    # Target
    df["target"] = _compute_target(df)

    log_structured(
        logger, logging.INFO,
        "All enhanced features computed successfully",
        entity_type="dataset",
    )

    return df


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder feature columns."""
    cols = list(FEATURES_DTYPES.keys())
    return df[cols].copy()


def _print_certification(df: pd.DataFrame) -> None:
    """Print and log the Stage 4b certification report."""
    row_count = len(df)

    # Feature statistics
    stats_lines = [
        "",
        "=" * 70,
        "  STAGE 4b CERTIFICATION — Enhanced Feature Generation",
        "=" * 70,
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
        "  Recent Form Features:",
        f"    form_last_5_a:      mean={df['form_last_5_a'].mean():.3f}, nan={df['form_last_5_a'].isna().sum():,}",
        f"    form_last_10_a:     mean={df['form_last_10_a'].mean():.3f}, nan={df['form_last_10_a'].isna().sum():,}",
        f"    form_7_days_a:      mean={df['form_7_days_a'].mean():.3f}, nan={df['form_7_days_a'].isna().sum():,}",
        "",
        "  Cumulative Features (for comparison):",
        f"    cumulative_matches_a: mean={df['cumulative_matches_a'].mean():.1f}",
        f"    cumulative_wins_a:    mean={df['cumulative_wins_a'].mean():.1f}",
        "",
        "  Key Insight:",
        f"    Recent form captures momentum + current fitness vs cumulative experience",
        "=" * 70,
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
    """Run the full Stage 4b enhanced feature generation pipeline."""
    global logger
    logger = setup_stage_logger("generate_features_v2")

    log_structured(logger, logging.INFO, "Stage 4b — generate_features_v2 starting", status="start")

    # 1. Load Elo-rated matches
    df = _load_elo_matches()

    # 2. Generate all features
    df = _generate_all_features(df)

    # 3. Select feature columns
    df = _select_feature_columns(df)

    # 4. Validate schema
    from pipeline.schema import validate_schema
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

    log_structured(logger, logging.INFO, "Stage 4b — generate_features_v2 complete", status="done")


if __name__ == "__main__":
    try:
        generate_features()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("generate_features_v2")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
