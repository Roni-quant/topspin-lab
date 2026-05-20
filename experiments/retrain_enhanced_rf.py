"""Retrain enhanced 9-feature RF on full pre-London history, save to models/random_forest_v2.pkl.

Uses the canonical feature set defined in `pipeline.train_models_v2.FEATURE_COLS_ENHANCED`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline.train_models_v2 import FEATURE_COLS_ENHANCED

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "model_features.parquet"
MODEL_OUT = ROOT / "models" / "random_forest_v2.pkl"
LONDON_DATE = pd.Timestamp("2026-05-10")


def main() -> None:
    df = pd.read_parquet(FEATURES)
    df = df[df["match_date"] < LONDON_DATE].copy()
    for c in FEATURE_COLS_ENHANCED:
        df[c] = df[c].fillna(0)
    X = df[FEATURE_COLS_ENHANCED].values
    y = df["target"].astype(int).values
    print(f"Training enhanced RF on {len(df):,} matches (< {LONDON_DATE.date()})")
    print(f"Features ({len(FEATURE_COLS_ENHANCED)}): {FEATURE_COLS_ENHANCED}")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    print(f"n_features_in_: {rf.n_features_in_}")
    print(f"n_estimators: {rf.n_estimators}")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(rf, f)
    print(f"Saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
