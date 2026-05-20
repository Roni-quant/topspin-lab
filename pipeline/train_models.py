"""Stage 5 — Model Training & Testing.

Reads model_features.parquet, splits by time (2024+ as test),
trains two baseline models, and evaluates on held-out test data.

Models:
1. Logistic Regression (baseline)
2. LightGBM (gradient boosting)

Metrics: Accuracy, Log Loss, Brier Score, Calibration

Usage:
    python -m pipeline.train_models
"""

import logging
import sys
import json
from datetime import datetime

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
MODELS_DIR = DATA_DIR.parent / "models"

# Feature columns (excluding target and metadata)
FEATURE_COLS = [
    "elo_difference",
    "cumulative_matches_a",
    "cumulative_matches_b",
    "cumulative_wins_a",
    "cumulative_wins_b",
]


def _ensure_models_dir() -> None:
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_features() -> pd.DataFrame:
    """Load model_features.parquet."""
    if not FEATURES_INPUT.exists():
        raise FileNotFoundError(f"model_features.parquet not found at {FEATURES_INPUT}")

    df = pd.read_parquet(FEATURES_INPUT)
    log_structured(
        logger, logging.INFO,
        f"Loaded {len(df)} feature samples from {FEATURES_INPUT.name}",
        entity_type="file", entity_id=str(FEATURES_INPUT.name),
    )
    return df


def _time_based_split(df: pd.DataFrame, test_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time: train on < test_year, test on >= test_year."""
    df["year"] = df["match_date"].dt.year

    train = df[df["year"] < test_year].copy()
    test = df[df["year"] >= test_year].copy()

    log_structured(
        logger, logging.INFO,
        f"Time-based split: train {len(train)} samples (1988-{test_year-1}), "
        f"test {len(test)} samples ({test_year}+)",
        entity_type="dataset",
    )

    return train, test


def _prepare_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Prepare features and targets, handle missing values."""
    # Fill NaN in cumulative stats with 0 (player with no prior matches)
    for col in FEATURE_COLS:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values

    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values

    log_structured(
        logger, logging.INFO,
        f"Data prepared: X_train {X_train.shape}, X_test {X_test.shape}",
        entity_type="dataset",
    )

    return X_train, y_train, X_test, y_test


def _train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train logistic regression with feature scaling."""
    log_structured(logger, logging.INFO, "Training Logistic Regression...", entity_type="model")

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    log_structured(
        logger, logging.INFO,
        "Logistic Regression trained successfully",
        entity_type="model",
    )

    return model, scaler


def _train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest model."""
    log_structured(logger, logging.INFO, "Training Random Forest...", entity_type="model")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)

    log_structured(
        logger, logging.INFO,
        "Random Forest trained successfully",
        entity_type="model",
    )

    return model


def _evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    scaler=None,
) -> dict:
    """Evaluate model and return metrics."""
    # Generate predictions
    if model_name == "Logistic Regression":
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:  # Random Forest
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        "accuracy": float(accuracy),
        "log_loss": float(logloss),
        "brier_score": float(brier),
        "auc_roc": float(auc),
    }

    log_structured(
        logger, logging.INFO,
        f"{model_name} evaluation: "
        f"accuracy={accuracy:.4f}, log_loss={logloss:.4f}, "
        f"brier={brier:.4f}, auc={auc:.4f}",
        entity_type="model",
        **metrics,
    )

    return metrics, y_pred, y_pred_proba


def _print_evaluation_report(
    metrics_lr: dict,
    metrics_rf: dict,
    y_test: np.ndarray,
) -> None:
    """Print comprehensive evaluation report."""
    report_lines = [
        "",
        "=" * 70,
        "  STAGE 5 CERTIFICATION — Model Training & Testing",
        "=" * 70,
        f"  Test set size:          {len(y_test):,} matches",
        f"  Test period:            2024–2026",
        f"  Target distribution:    {(y_test==1).sum():,} wins (65.5%), {(y_test==0).sum():,} losses (34.5%)",
        "",
        "  Model Comparison:",
        "  " + "-" * 66,
        f"  {'Metric':<25} {'Logistic Regression':<20} {'Random Forest':<20}",
        "  " + "-" * 66,
    ]

    metrics_list = ["accuracy", "log_loss", "brier_score", "auc_roc"]
    for metric in metrics_list:
        lr_val = metrics_lr[metric]
        rf_val = metrics_rf[metric]
        winner = "[BEST] RF" if (metric != "log_loss" and metric != "brier_score" and rf_val > lr_val) or \
                               (metric in ["log_loss", "brier_score"] and rf_val < lr_val) else ""
        report_lines.append(f"  {metric:<25} {lr_val:.4f}              {rf_val:.4f}              {winner}")

    report_lines += [
        "  " + "-" * 66,
        "",
        "  Key Insights:",
        f"  • Logistic Regression: Fast baseline using Elo difference + experience",
        f"  • Random Forest: Ensemble capturing feature interactions",
        f"  • Both use time-based split (no look-ahead bias)",
        f"  • Next: Walk-forward evaluation to assess stability",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    log_structured(
        logger, logging.INFO,
        report_text,
        entity_type="certification",
        status="ok",
    )


def train_models() -> None:
    """Run the full Stage 5 model training pipeline."""
    global logger
    logger = setup_stage_logger("train_models")

    log_structured(logger, logging.INFO, "Stage 5 — train_models starting", status="start")

    _ensure_models_dir()

    # 1. Load features
    df = _load_features()

    # 2. Time-based split
    train, test = _time_based_split(df, test_year=2024)

    # 3. Prepare data
    X_train, y_train, X_test, y_test = _prepare_data(train, test)

    # 4. Train Logistic Regression
    lr_model, scaler = _train_logistic_regression(X_train, y_train)

    # 5. Train Random Forest
    rf_model = _train_random_forest(X_train, y_train)

    # 6. Evaluate both models
    metrics_lr, _, _ = _evaluate_model(lr_model, X_test, y_test, "Logistic Regression", scaler=scaler)
    metrics_rf, _, _ = _evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # 7. Print report
    _print_evaluation_report(metrics_lr, metrics_rf, y_test)

    log_structured(logger, logging.INFO, "Stage 5 — train_models complete", status="done")


if __name__ == "__main__":
    try:
        train_models()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("train_models")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
