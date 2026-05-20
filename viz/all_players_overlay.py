"""Viz 2 - Every active player's Elo over career, with 3 stars overlaid.

Matches the tweet aesthetic: a dense gray cloud of all players' careers, with
three named stars highlighted in red / blue / green. Shows where the ceiling
sits and how the top players escape the bulk.

Filter: players with at least MIN_MATCHES matches (default 50).

Usage:
    python -m viz.all_players_overlay
    python -m viz.all_players_overlay --stars "Ma Long" "Fan Zhendong" "Sun Yingsha"
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz._common import (
    ELO_MATCHES, IMG_DIR, apply_dark_style, ensure_inputs, find_player_id,
    player_name_map,
)

MIN_MATCHES = 50
DEFAULT_STARS = ["Ma Long", "Fan Zhendong", "Sun Yingsha"]
STAR_COLORS = ["#ff3b3b", "#3b9dff", "#39ff14"]  # red, blue, green


def build_player_series_all(min_matches: int) -> dict[int, np.ndarray]:
    """Return {player_id: array of pre-match Elo, chronological}.

    Single pass over matches_with_elo, grouping by player. Much faster than
    re-filtering the full frame per player.
    """
    df = pd.read_parquet(ELO_MATCHES).sort_values("match_date").reset_index(drop=True)
    series: dict[int, list[float]] = defaultdict(list)
    for row in df.itertuples(index=False):
        series[int(row.player_a_id)].append(float(row.elo_a_before))
        series[int(row.player_b_id)].append(float(row.elo_b_before))
    return {pid: np.array(vals) for pid, vals in series.items() if len(vals) >= min_matches}


def main() -> None:
    parser = argparse.ArgumentParser(description="All-players Elo overlay with stars.")
    parser.add_argument("--stars", nargs="+", default=DEFAULT_STARS,
                        help=f"Player name substrings to highlight (default: {DEFAULT_STARS})")
    parser.add_argument("--min-matches", type=int, default=MIN_MATCHES,
                        help=f"Minimum matches to include a player in the crowd (default: {MIN_MATCHES})")
    args = parser.parse_args()

    ensure_inputs()
    apply_dark_style()

    print(f"Building Elo series for all players (min {args.min_matches} matches)...")
    all_series = build_player_series_all(args.min_matches)
    print(f"  Players in crowd: {len(all_series):,}")

    # Resolve stars up front so a typo fails fast
    names = player_name_map()
    star_ids: list[tuple[int, str, str]] = []
    for needle, color in zip(args.stars, STAR_COLORS):
        pid = find_player_id(needle)
        star_ids.append((pid, names[pid], color))
        print(f"  Star: {names[pid]} (id={pid}, {len(all_series.get(pid, [])):,} matches)")

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Crowd - every qualifying player as a faint gray line
    for pid, elo in all_series.items():
        ax.plot(np.arange(1, len(elo) + 1), elo, color="#bbbbbb",
                linewidth=0.4, alpha=0.06, zorder=1)

    # Stars - thick colored overlay
    for pid, name, color in star_ids:
        elo = all_series.get(pid)
        if elo is None or len(elo) == 0:
            print(f"  WARN: {name} has no matches in the filtered crowd; plotting anyway")
            # Fall back to unfiltered: rebuild for this one player
            from viz._common import player_elo_series
            elo = player_elo_series(pid)["elo"].to_numpy()
        ax.plot(np.arange(1, len(elo) + 1), elo, color=color,
                linewidth=2.4, alpha=0.95, label=name, zorder=3)

    ax.set_title("Elo Ratings Over Career - All Players, Three Stars Highlighted")
    ax.set_xlabel("Match number")
    ax.set_ylabel("Elo rating (pre-match)")
    ax.legend(loc="lower right", framealpha=0.0, labelcolor="white")

    # Reference line at base Elo
    ax.axhline(1500, color="#444", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(5, 1505, "base Elo 1500", color="#666", fontsize=9)

    ax.text(
        0.01, -0.13,
        f"{len(all_series):,} players  ·  filter: ≥{args.min_matches} matches  ·  pre-match ratings only",
        transform=ax.transAxes, color="#888", fontsize=9,
    )

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "all_players_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(out.parent.parent.parent)}")


if __name__ == "__main__":
    main()
