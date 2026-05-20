"""Regenerate every plot under docs/img/ in one shot.

Usage:
    python -m viz.make_all
"""

from __future__ import annotations

from viz import all_players_overlay, player_elo_trajectory
from viz._common import apply_dark_style, ensure_inputs, find_player_id, player_name_map


HEADLINE_PLAYERS = ["Ma Long", "Fan Zhendong", "Sun Yingsha", "Wang Chuqin"]


def main() -> None:
    ensure_inputs()
    apply_dark_style()

    names = player_name_map()
    print("=" * 60)
    print(" Regenerating all plots -> docs/img/")
    print("=" * 60)

    print("\n[1/2] Single-player Elo trajectories")
    for needle in HEADLINE_PLAYERS:
        try:
            pid = find_player_id(needle)
            out = player_elo_trajectory.plot_player(pid, names[pid])
            print(f"  ok  {names[pid]:20}  ->  {out.name}")
        except Exception as exc:
            print(f"  skip {needle:20}  ({exc})")

    print("\n[2/2] All-players overlay")
    # Reuse the script's argparse-driven main by faking argv? Cleaner: call its
    # internals directly. Keep it simple - shell out via the script.
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "viz.all_players_overlay"], check=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
