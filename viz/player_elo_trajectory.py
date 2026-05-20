"""Viz 1 — One player's Elo over their career.

Matches the tweet aesthetic: dark background, x = match number, y = Elo rating.
The "peak window" — matches where the player's rolling Elo sits in the top
20% of their career — is highlighted in green; the rest is white.

Usage:
    python -m viz.player_elo_trajectory                 # default: Ma Long
    python -m viz.player_elo_trajectory "Fan Zhendong"
    python -m viz.player_elo_trajectory --id 12345
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz._common import (
    IMG_DIR, apply_dark_style, ensure_inputs, find_player_id,
    player_elo_series, player_name_map,
)


DEFAULT_PLAYER = "Ma Long"
ROLLING_WINDOW = 50      # matches, for the smoothed peak-detection signal
PEAK_PERCENTILE = 80     # top X% of rolling Elo counted as "peak window"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_player(player_id: int, name: str) -> Path:
    series = player_elo_series(player_id)
    if len(series) < 10:
        raise ValueError(
            f"{name} has only {len(series)} matches in the dataset — too few to plot."
        )

    match_num = np.arange(1, len(series) + 1)
    elo = series["elo"].to_numpy()

    # Smoothed signal — rolling mean over recent matches
    win = min(ROLLING_WINDOW, max(5, len(elo) // 10))
    rolling = (
        series["elo"].rolling(window=win, min_periods=1).mean().to_numpy()
    )
    threshold = np.percentile(rolling, PEAK_PERCENTILE)
    in_peak = rolling >= threshold

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # White trace = full career
    ax.plot(match_num, elo, color="white", linewidth=1.0, alpha=0.85, zorder=2)

    # Green overlay = peak window only (mask non-peak with NaN so the line breaks)
    peak_trace = np.where(in_peak, elo, np.nan)
    ax.plot(match_num, peak_trace, color="#39ff14", linewidth=2.0, zorder=3,
            label=f"Peak window (top {100 - PEAK_PERCENTILE}% of career)")

    # Annotate career-peak point
    peak_idx = int(np.argmax(elo))
    ax.scatter([match_num[peak_idx]], [elo[peak_idx]], color="#39ff14",
               s=60, zorder=4, edgecolor="white", linewidth=1.0)
    ax.annotate(
        f"Career peak: {elo[peak_idx]:.0f}",
        xy=(match_num[peak_idx], elo[peak_idx]),
        xytext=(8, 8), textcoords="offset points",
        color="#39ff14", fontsize=10, fontweight="bold",
    )

    ax.set_title(f"{name} — Elo Rating Over Career")
    ax.set_xlabel("Match number")
    ax.set_ylabel("Elo rating (pre-match)")
    ax.legend(loc="lower right", framealpha=0.0, labelcolor="white")
    ax.set_xlim(0, len(elo) + 1)

    # Footer caption: dataset transparency
    ax.text(
        0.01, -0.18,
        f"{len(elo):,} matches  ·  base Elo 1500  ·  K=32  ·  pre-match ratings only",
        transform=ax.transAxes, color="#888", fontsize=9,
    )

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / f"player_elo_{_slug(name)}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one player's Elo trajectory.")
    parser.add_argument("name", nargs="?", default=DEFAULT_PLAYER,
                        help="Player name substring (case-insensitive). Default: Ma Long.")
    parser.add_argument("--id", type=int, default=None,
                        help="Explicit player_id (overrides name lookup).")
    args = parser.parse_args()

    ensure_inputs()
    apply_dark_style()

    if args.id is not None:
        pid = args.id
        name = player_name_map().get(pid, f"player_{pid}")
    else:
        pid = find_player_id(args.name)
        name = player_name_map()[pid]

    out = plot_player(pid, name)
    print(f"Wrote {out.relative_to(out.parent.parent.parent)}")


if __name__ == "__main__":
    main()
