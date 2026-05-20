"""Validate model on truly unseen ITTF World Team Championships London 2026.

Uses last-known Elo per player (from full matches_clean history replayed through
the Elo engine) + recent-form features computed from the same history. No data
from London 2026 leaks into priors — priors freeze at the last match before
2026-05-10.

Evaluates:
  1. Pure Elo prior: P(A wins) = expected_score(elo_a, elo_b)
  2. Shipped Random Forest (models/random_forest_v2.pkl) — 5-feature baseline
  3. Freshly-trained enhanced Random Forest (9 features, trained on data
     strictly before London 2026)

Cold-start handling: players with no prior matches in our history are excluded
and reported separately (model failure ≠ cold-start failure).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from ratings.elo import EloConfig, EloRatingEngine, expected_score

ROOT = Path(__file__).resolve().parent.parent
CLEAN_MATCHES = ROOT / "data" / "matches_clean.parquet"
ELO_MATCHES = ROOT / "data" / "matches_with_elo.parquet"
FEATURES_PARQUET = ROOT / "data" / "model_features.parquet"
LONDON_RAW = Path(__file__).resolve().parent / "london_2026_matches.parquet"
MODEL_PKL = ROOT / "models" / "random_forest_v2.pkl"
LONDON_DATE = pd.Timestamp("2026-05-10")

BASELINE_FEATURES = [
    "elo_difference",
    "cumulative_matches_a",
    "cumulative_matches_b",
    "cumulative_wins_a",
    "cumulative_wins_b",
]
ENHANCED_FEATURES = [
    "elo_difference",
    "form_last_5_a",
    "form_last_10_a",
    "form_last_5_b",
    "form_last_10_b",
    "form_7_days_a",
    "form_7_days_b",
    "matches_last_7_a",
    "matches_last_7_b",
]


# ---------------------------------------------------------------------------
# Build priors from history (everything strictly before London 2026)
# ---------------------------------------------------------------------------


def build_history_state():
    """Replay full history up to (but not including) London 2026 date.

    Returns:
        elo_engine: EloRatingEngine populated with final ratings.
        player_history: dict[player_id -> list[(date, is_win)] sorted by date].
        cum_matches: dict[player_id -> int].
        cum_wins:    dict[player_id -> int].
    """
    df = pd.read_parquet(CLEAN_MATCHES).sort_values("match_date").reset_index(drop=True)
    df = df[df["match_date"] < LONDON_DATE]
    print(f"  History matches (< {LONDON_DATE.date()}): {len(df):,}")
    print(f"  History date range: {df['match_date'].min().date()} -> {df['match_date'].max().date()}")

    engine = EloRatingEngine(EloConfig(base_rating=1500.0, k_factor=32.0))
    history: dict[int, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    cum_m: dict[int, int] = defaultdict(int)
    cum_w: dict[int, int] = defaultdict(int)

    for row in df.itertuples(index=False):
        pa, pb, win = int(row.player_a_id), int(row.player_b_id), int(row.winner_id)
        engine.process_match(pa, pb, win)
        is_a = 1.0 if win == pa else 0.0
        is_b = 1.0 - is_a
        history[pa].append((row.match_date, is_a))
        history[pb].append((row.match_date, is_b))
        cum_m[pa] += 1
        cum_m[pb] += 1
        cum_w[pa] += int(is_a)
        cum_w[pb] += int(is_b)

    print(f"  Players with prior history: {len(engine.ratings):,}")
    return engine, history, cum_m, cum_w


# ---------------------------------------------------------------------------
# Feature computation for London 2026 matches
# ---------------------------------------------------------------------------


def _form_lastn(hist: list[tuple[pd.Timestamp, float]], n: int) -> float:
    if not hist:
        return np.nan
    recent = hist[-n:]
    return float(np.mean([w for _, w in recent]))


def _form_window(hist: list[tuple[pd.Timestamp, float]], cutoff: pd.Timestamp) -> tuple[float, int]:
    wins = [w for d, w in hist if d >= cutoff]
    return (float(np.mean(wins)) if wins else np.nan, len(wins))


def build_london_features(engine, history, cum_m, cum_w) -> pd.DataFrame:
    """For each London 2026 singles match, compute features using only pre-event priors."""
    df = pd.read_parquet(LONDON_RAW)
    # Keep only usable singles
    df = df[df["drop_reason"].isna() & ~df["is_doubles"]].copy()
    print(f"  London singles matches: {len(df):,}")

    rows = []
    cutoff_7 = LONDON_DATE - timedelta(days=7)

    for r in df.itertuples(index=False):
        pa, pb, win = int(r.player_a_id), int(r.player_b_id), int(r.winner_id)
        elo_a = engine.ratings.get(pa, np.nan)
        elo_b = engine.ratings.get(pb, np.nan)
        cold_a = pa not in engine.ratings
        cold_b = pb not in engine.ratings

        ha = history.get(pa, [])
        hb = history.get(pb, [])
        f7a, m7a = _form_window(ha, cutoff_7)
        f7b, m7b = _form_window(hb, cutoff_7)

        rows.append({
            "player_a_id": pa,
            "player_b_id": pb,
            "winner_id": win,
            "target": 1 if win == pa else 0,
            "cold_start_a": cold_a,
            "cold_start_b": cold_b,
            "elo_a_before": elo_a,
            "elo_b_before": elo_b,
            "elo_difference": elo_a - elo_b,
            "form_last_5_a": _form_lastn(ha, 5),
            "form_last_10_a": _form_lastn(ha, 10),
            "form_last_5_b": _form_lastn(hb, 5),
            "form_last_10_b": _form_lastn(hb, 10),
            "form_7_days_a": f7a,
            "form_7_days_b": f7b,
            "matches_last_7_a": m7a,
            "matches_last_7_b": m7b,
            "cumulative_matches_a": cum_m.get(pa, 0),
            "cumulative_matches_b": cum_m.get(pb, 0),
            "cumulative_wins_a": cum_w.get(pa, 0),
            "cumulative_wins_b": cum_w.get(pb, 0),
            "category": r.category,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(name: str, y_true, p) -> dict:
    y_pred = (p >= 0.5).astype(int)
    out = {
        "model": name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan"),
        "logloss": log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y_true, p),
    }
    return out


def _calibration_table(y_true, p, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"[{edges[b]:.2f},{edges[b+1]:.2f})",
            "n": int(mask.sum()),
            "mean_pred": float(p[mask].mean()),
            "actual_win_rate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fresh enhanced RF (trained strictly on data before London 2026)
# ---------------------------------------------------------------------------


def train_enhanced_rf_pre_london() -> RandomForestClassifier:
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["match_date"] < LONDON_DATE].copy()
    for c in ENHANCED_FEATURES:
        df[c] = df[c].fillna(0)
    X = df[ENHANCED_FEATURES].values
    y = df["target"].astype(int).values
    print(f"  Training enhanced RF on {len(df):,} matches before {LONDON_DATE.date()}")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  London 2026 Validation — truly unseen tournament")
    print("=" * 70)

    print("\n[1/4] Replaying history to build priors...")
    engine, history, cum_m, cum_w = build_history_state()

    print("\n[2/4] Building features for London 2026 matches...")
    feats = build_london_features(engine, history, cum_m, cum_w)

    cold_mask = feats["cold_start_a"] | feats["cold_start_b"]
    print(f"  Cold-start matches (>=1 unseen player): {cold_mask.sum()}")
    feats_eval = feats[~cold_mask].copy().reset_index(drop=True)
    print(f"  Evaluable matches: {len(feats_eval)}")
    print(f"    MT: {(feats_eval['category']=='MT').sum()}, WT: {(feats_eval['category']=='WT').sum()}")

    y = feats_eval["target"].astype(int).values

    # --- Model 1: Pure Elo prior ---
    print("\n[3/4] Scoring models...")
    p_elo = np.array([expected_score(a, b) for a, b in zip(feats_eval["elo_a_before"], feats_eval["elo_b_before"])])
    m_elo = _metrics("Pure Elo prior", y, p_elo)

    # --- Model 2: Shipped RF v2 (5-feature baseline) ---
    with open(MODEL_PKL, "rb") as f:
        rf_baseline = pickle.load(f)
    Xb = feats_eval[BASELINE_FEATURES].fillna(0).values
    p_rf_baseline = rf_baseline.predict_proba(Xb)[:, 1]
    m_rf_b = _metrics("Shipped RF (5-feature baseline)", y, p_rf_baseline)

    # --- Model 3: Fresh enhanced RF trained pre-London ---
    rf_enh = train_enhanced_rf_pre_london()
    Xe = feats_eval[ENHANCED_FEATURES].fillna(0).values
    p_rf_enh = rf_enh.predict_proba(Xe)[:, 1]
    m_rf_e = _metrics("Fresh enhanced RF (9 features)", y, p_rf_enh)

    # --- Report ---
    print("\n[4/4] Results")
    print("=" * 70)
    res = pd.DataFrame([m_elo, m_rf_b, m_rf_e])
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCalibration (Pure Elo):")
    print(_calibration_table(y, p_elo).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCalibration (Fresh enhanced RF):")
    print(_calibration_table(y, p_rf_enh).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Breakdown by category
    print("\nBy category — accuracy (Pure Elo / Shipped RF / Enhanced RF):")
    for cat in ("MT", "WT"):
        mask = (feats_eval["category"] == cat).values
        if mask.sum() == 0:
            continue
        a1 = accuracy_score(y[mask], (p_elo[mask] >= 0.5).astype(int))
        a2 = accuracy_score(y[mask], (p_rf_baseline[mask] >= 0.5).astype(int))
        a3 = accuracy_score(y[mask], (p_rf_enh[mask] >= 0.5).astype(int))
        print(f"  {cat}: n={mask.sum():<4} elo={a1:.4f}  rf_base={a2:.4f}  rf_enh={a3:.4f}")

    out_csv = Path(__file__).resolve().parent / "london_2026_predictions.csv"
    feats_eval = feats_eval.assign(p_elo=p_elo, p_rf_baseline=p_rf_baseline, p_rf_enhanced=p_rf_enh)
    feats_eval.to_csv(out_csv, index=False)
    print(f"\nPer-match predictions saved -> {out_csv}")


if __name__ == "__main__":
    main()
