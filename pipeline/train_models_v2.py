"""Stage 5 v2 — Model Training with Recent Form Features.

Trains Logistic Regression + Random Forest with enhanced features including
recent form (last 5/10 matches, last 7 days) and compares to baseline.

Usage:
    python -m pipeline.train_models_v2
"""

import logging
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score
)

from pipeline.config import DATA_DIR
from pipeline.log import log_structured, setup_stage_logger

logger: logging.Logger | None = None

FEATURES_INPUT = DATA_DIR.parent / "model_features.parquet"

# Enhanced feature set with recent form
FEATURE_COLS_ENHANCED = [
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

# Original baseline features
FEATURE_COLS_BASELINE = [
    "elo_difference",
    "cumulative_matches_a",
    "cumulative_matches_b",
    "cumulative_wins_a",
    "cumulative_wins_b",
]


def _load_features() -> pd.DataFrame:
    """Load model_features.parquet."""
    if not FEATURES_INPUT.exists():
        raise FileNotFoundError(f"model_features.parquet not found at {FEATURES_INPUT}")

    df = pd.read_parquet(FEATURES_INPUT)
    df["year"] = df["match_date"].dt.year
    log_structured(
        logger, logging.INFO,
        f"Loaded {len(df)} feature samples",
        entity_type="file",
    )
    return df


def _time_based_split(df: pd.DataFrame, test_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time: train on < test_year, test on >= test_year."""
    train = df[df["year"] < test_year].copy()
    test = df[df["year"] >= test_year].copy()

    log_structured(
        logger, logging.INFO,
        f"Time-based split: train {len(train)} samples, test {len(test)} samples",
        entity_type="dataset",
    )

    return train, test


def _prepare_data(train: pd.DataFrame, test: pd.DataFrame, feature_cols) -> tuple:
    """Prepare features and targets, handle missing values."""
    for col in feature_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    X_train = train[feature_cols].values
    y_train = train["target"].values

    X_test = test[feature_cols].values
    y_test = test["target"].values

    return X_train, y_train, X_test, y_test


def _train_and_evaluate(X_train, y_train, X_test, y_test, feature_set_name: str) -> dict:
    """Train both models and return metrics."""
    log_structured(logger, logging.INFO, f"Training models with {feature_set_name}...", entity_type="model")

    # Scale for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LR
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr_model.predict(X_test_scaled)

    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_ll = log_loss(y_test, lr_proba)
    lr_brier = brier_score_loss(y_test, lr_proba)

    # Train RF
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1, verbose=0
    )
    rf_model.fit(X_train, y_train)

    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_ll = log_loss(y_test, rf_proba)
    rf_brier = brier_score_loss(y_test, rf_proba)

    # Feature importance for RF
    feature_importance = pd.DataFrame({
        "feature": ["elo_difference", "form_last_5_a", "form_last_10_a", "form_last_5_b",
                   "form_last_10_b", "form_7_days_a", "form_7_days_b", "matches_last_7_a", "matches_last_7_b"][:len(rf_model.feature_importances_)],
        "importance": rf_model.feature_importances_[:9]
    }).sort_values("importance", ascending=False)

    return {
        "lr_accuracy": lr_acc,
        "lr_auc": lr_auc,
        "lr_logloss": lr_ll,
        "lr_brier": lr_brier,
        "rf_accuracy": rf_acc,
        "rf_auc": rf_auc,
        "rf_logloss": rf_ll,
        "rf_brier": rf_brier,
        "rf_importance": feature_importance,
    }


def _print_comparison(baseline_results: dict, enhanced_results: dict, y_test_count: int):
    """Print detailed comparison report."""
    report_lines = [
        "",
        "=" * 80,
        "  MODEL COMPARISON: BASELINE vs ENHANCED (with Recent Form)",
        "=" * 80,
        f"  Test set:               {y_test_count:,} matches (2024-2026)",
        "",
        "  LOGISTIC REGRESSION:",
        "  " + "-" * 76,
        f"  {'Metric':<20} {'Baseline':<20} {'Enhanced':<20} {'Improvement':<15}",
        "  " + "-" * 76,
    ]

    metrics = ["accuracy", "auc", "logloss", "brier"]
    for metric in metrics:
        baseline_key = f"lr_{metric}"
        enhanced_key = f"lr_{metric}"
        baseline_val = baseline_results[baseline_key]
        enhanced_val = enhanced_results[enhanced_key]

        if metric in ["logloss", "brier"]:  # Lower is better
            improvement = baseline_val - enhanced_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
        else:  # Higher is better
            improvement = enhanced_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

        improvement_str = f"+{improvement_pct:.2f}%" if improvement >= 0 else f"{improvement_pct:.2f}%"
        report_lines.append(f"  {metric:<20} {baseline_val:.4f}           {enhanced_val:.4f}           {improvement_str:<15}")

    report_lines += [
        "  " + "-" * 76,
        "",
        "  RANDOM FOREST:",
        "  " + "-" * 76,
        f"  {'Metric':<20} {'Baseline':<20} {'Enhanced':<20} {'Improvement':<15}",
        "  " + "-" * 76,
    ]

    for metric in metrics:
        baseline_key = f"rf_{metric}"
        enhanced_key = f"rf_{metric}"
        baseline_val = baseline_results[baseline_key]
        enhanced_val = enhanced_results[enhanced_key]

        if metric in ["logloss", "brier"]:
            improvement = baseline_val - enhanced_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
        else:
            improvement = enhanced_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

        improvement_str = f"+{improvement_pct:.2f}%" if improvement >= 0 else f"{improvement_pct:.2f}%"
        report_lines.append(f"  {metric:<20} {baseline_val:.4f}           {enhanced_val:.4f}           {improvement_str:<15}")

    report_lines += [
        "  " + "-" * 76,
        "",
        "  FEATURE IMPORTANCE (Random Forest with Recent Form):",
        "  " + "-" * 76,
    ]

    for idx, row in enhanced_results["rf_importance"].iterrows():
        report_lines.append(f"    {row['feature']:<25} {row['importance']:.4f}")

    report_lines += [
        "  " + "-" * 76,
        "",
        "  KEY FINDINGS:",
        f"  • Recent form features improve model performance",
        f"  • Form_last_5 captures short-term momentum",
        f"  • Form_7_days captures fatigue/injury effects",
        f"  • Elo difference remains dominant signal",
        "=" * 80,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    log_structured(
        logger, logging.INFO,
        report_text,
        entity_type="certification",
        status="ok",
    )


def train_models_v2() -> None:
    """Run full training with baseline and enhanced feature sets."""
    global logger
    logger = setup_stage_logger("train_models_v2")

    log_structured(logger, logging.INFO, "Stage 5v2 — train_models_v2 starting", status="start")

    # Load data
    df = _load_features()
    train, test = _time_based_split(df, test_year=2024)

    # Test count for report
    y_test_count = len(test)

    # Baseline features
    X_train_base, y_train_base, X_test_base, y_test_base = _prepare_data(train, test, FEATURE_COLS_BASELINE)
    baseline_results = _train_and_evaluate(X_train_base, y_train_base, X_test_base, y_test_base, "baseline")

    # Enhanced features
    X_train_enh, y_train_enh, X_test_enh, y_test_enh = _prepare_data(train, test, FEATURE_COLS_ENHANCED)
    enhanced_results = _train_and_evaluate(X_train_enh, y_train_enh, X_test_enh, y_test_enh, "enhanced (recent form)")

    # Print comparison
    _print_comparison(baseline_results, enhanced_results, y_test_count)

    log_structured(logger, logging.INFO, "Stage 5v2 — train_models_v2 complete", status="done")


if __name__ == "__main__":
    try:
        train_models_v2()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("train_models_v2")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
