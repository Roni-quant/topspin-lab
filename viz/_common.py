"""Shared helpers for viz scripts: paths, style, player-name lookup, series."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_MATCHES = DATA / "raw_matches.parquet"
ELO_MATCHES = DATA / "matches_with_elo.parquet"
IMG_DIR = ROOT / "docs" / "img"


def apply_dark_style() -> None:
    """Dark background, white text, sans-serif."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0d0d0d",
        "axes.facecolor": "#0d0d0d",
        "savefig.facecolor": "#0d0d0d",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#555",
        "axes.grid": True,
        "grid.color": "#222",
        "grid.linewidth": 0.5,
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
    })


def ensure_inputs() -> None:
    """Fail loudly if required Parquet files are missing."""
    missing = [p for p in (RAW_MATCHES, ELO_MATCHES) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing required data files: {names}\n"
            "Run the pipeline first:\n"
            "  python -m pipeline.fetch_events\n"
            "  python -m pipeline.fetch_matches\n"
            "  python -m pipeline.merge_raw\n"
            "  python -m pipeline.clean\n"
            "  python -m pipeline.compute_elo"
        )


@lru_cache(maxsize=1)
def player_name_map() -> dict[int, str]:
    """player_id -> most-recent name seen in raw_matches.parquet."""
    df = pd.read_parquet(RAW_MATCHES, columns=[
        "match_date", "player_a_id", "player_a_name", "player_b_id", "player_b_name"
    ])
    a = df[["match_date", "player_a_id", "player_a_name"]].rename(
        columns={"player_a_id": "pid", "player_a_name": "name"}
    )
    b = df[["match_date", "player_b_id", "player_b_name"]].rename(
        columns={"player_b_id": "pid", "player_b_name": "name"}
    )
    both = pd.concat([a, b], ignore_index=True).dropna(subset=["pid", "name"])
    both = both.sort_values("match_date").drop_duplicates("pid", keep="last")
    return dict(zip(both["pid"].astype(int), both["name"].astype(str)))


@lru_cache(maxsize=1)
def _elo_df() -> pd.DataFrame:
    """Cached load of matches_with_elo.parquet, sorted by date."""
    df = pd.read_parquet(ELO_MATCHES)
    return df.sort_values("match_date").reset_index(drop=True)


def player_elo_series(player_id: int) -> pd.DataFrame:
    """Return DataFrame[match_date, elo] for one player, chronological.

    Uses the pre-match Elo (the rating just before each match), so the series
    shows ratings the model would have seen at prediction time.
    """
    df = _elo_df()
    a = df[df["player_a_id"] == player_id][["match_date", "elo_a_before"]].rename(
        columns={"elo_a_before": "elo"}
    )
    b = df[df["player_b_id"] == player_id][["match_date", "elo_b_before"]].rename(
        columns={"elo_b_before": "elo"}
    )
    return pd.concat([a, b], ignore_index=True).sort_values("match_date").reset_index(drop=True)


def find_player_id(name_substring: str) -> int:
    """Look up a player_id by substring match on the latest name.

    Raises if zero or multiple matches.
    """
    names = player_name_map()
    needle = name_substring.lower()
    hits = [(pid, n) for pid, n in names.items() if needle in n.lower()]
    if not hits:
        raise LookupError(f"No player matches '{name_substring}'")
    if len(hits) > 1:
        sample = ", ".join(n for _, n in hits[:5])
        raise LookupError(
            f"Ambiguous '{name_substring}' — {len(hits)} matches: {sample}..."
        )
    return hits[0][0]
