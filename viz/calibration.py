"""Viz 3 - Reliability diagram for the London 2026 holdout.

Reads experiments/london_2026_predictions.csv (committed, no scrape needed),
bins predicted probabilities into deciles, and plots actual win rate per bin
against the diagonal. The closer the dots are to the diagonal, the better
calibrated the model.

Two curves overlaid: Pure Elo prior vs Enhanced RF (9 features).

Usage:
    python -m viz.calibration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz._common import IMG_DIR, apply_dark_style

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "experiments" / "london_2026_predictions.csv"
BINS = 10


def reliability(y: np.ndarray, p: np.ndarray, bins: int = BINS) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin_center": float((edges[b] + edges[b + 1]) / 2),
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "actual": float(y[m].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS} not found. Run experiments.validate_london_2026 first."
        )
    apply_dark_style()

    df = pd.read_csv(PREDICTIONS)
    y = df["target"].to_numpy()
    p_elo = df["p_elo"].to_numpy()
    p_rf = df["p_rf_enhanced"].to_numpy()

    r_elo = reliability(y, p_elo)
    r_rf = reliability(y, p_rf)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Diagonal = perfect calibration
    ax.plot([0, 1], [0, 1], color="#666", linestyle="--", linewidth=1.2,
            zorder=1, label="perfect calibration")

    # Pure Elo
    ax.plot(r_elo["mean_pred"], r_elo["actual"],
            color="#3b9dff", linewidth=2.0, marker="o", markersize=7,
            zorder=3, label=f"Pure Elo (n={int(r_elo['n'].sum())})")

    # Enhanced RF
    ax.plot(r_rf["mean_pred"], r_rf["actual"],
            color="#39ff14", linewidth=2.0, marker="s", markersize=7,
            zorder=4, label=f"Enhanced RF (n={int(r_rf['n'].sum())})")

    # Bin-count bars at the bottom for transparency
    bar_y = -0.06
    bar_h = 0.04
    max_n = max(r_elo["n"].max(), r_rf["n"].max())
    for _, row in r_rf.iterrows():
        h = bar_h * row["n"] / max_n
        ax.add_patch(plt.Rectangle(
            (row["bin_center"] - 0.04, bar_y), 0.08, h,
            color="#39ff14", alpha=0.35, zorder=2,
        ))
    ax.text(0.02, bar_y - 0.01, "matches per bin (RF)",
            color="#888", fontsize=9, va="top")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bar_y - 0.04, 1.02)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Reliability diagram - London 2026 holdout (822 rubbers)")
    ax.legend(loc="upper left", framealpha=0.0, labelcolor="white")
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.02, -0.16,
        "Dots on the diagonal = predicted prob matches actual win rate. "
        "Below diagonal = overconfident. Above = underconfident.",
        transform=ax.transAxes, color="#aaa", fontsize=9,
    )

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "calibration_london_2026.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(out.parent.parent.parent)}")


if __name__ == "__main__":
    main()
