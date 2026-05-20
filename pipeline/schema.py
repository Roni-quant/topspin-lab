import re

import pandas as pd

# --- Match schema (yearly batches, includes drop_reason) ---

MATCH_DTYPES = {
    "match_key": "string",
    "source_match_id": "Int64",
    "match_date": "datetime64[ns]",
    "event_id": "Int64",
    "event_name": "string",
    "player_a_id": "Int64",
    "player_a_name": "string",
    "player_b_id": "Int64",
    "player_b_name": "string",
    "winner_id": "Int64",
    "result": "string",
    "games": "string",
    "category": "string",
    "drop_reason": "string",
}

MATCH_COLUMNS_MERGED = [
    "match_key", "source_match_id", "match_date", "event_id", "event_name",
    "player_a_id", "player_a_name", "player_b_id", "player_b_name",
    "winner_id", "result", "games", "category",
]

# --- Player schema ---

PLAYER_DTYPES = {
    "player_id": "Int64",
    "player_name": "string",
    "birth_year": "Int64",
    "association": "string",
    "hand": "string",
    "grip": "string",
    "style": "string",
}

# --- Event schema ---

EVENT_DTYPES = {
    "event_id": "Int64",
    "tournament_id": "Int64",
    "event_name": "string",
    "event_date": "datetime64[ns]",
    "event_category": "string",
}

# --- Constants ---

ALLOWED_CATEGORIES = {"MS", "WS"}
ACCEPTED_RESULT_REGEX = re.compile(r"^\d+\s*:\s*\d+$")
GAME_SCORE_REGEX = re.compile(r"^\d+\s*:\s*\d+$")

# --- Merged output schema (no drop_reason) ---

MATCH_DTYPES_MERGED = {k: v for k, v in MATCH_DTYPES.items() if k in MATCH_COLUMNS_MERGED}

# --- Drop reason codes ---

DROP_REASONS = {
    "non_standard_result",
    "missing_result",
    "walkover",
    "retirement",
    "malformed_games",
    "missing_player_id",
    "winner_mismatch",
}


def validate_schema(df: pd.DataFrame, expected_dtypes: dict) -> pd.DataFrame:
    """Validate and cast a DataFrame to the expected schema.

    Ensures all expected columns are present and casts to the correct dtypes.
    Raises ValueError if a required column is missing.
    Returns the cast DataFrame.
    """
    missing = set(expected_dtypes) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col, dtype in expected_dtypes.items():
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot cast column '{col}' to {dtype}: {e}")

    return df
