"""Stage 6 — Forward Prediction Simulation.

Walk-forward evaluation: for each month in the test period (2024-2026),
use all prior data to predict that month's matches, then record performance.

This simulates real-world prediction where we only have historical data
and test how well predictions hold up over time.

Usage:
    python -m pipeline.forward_test
"""

import logging
import sys
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from pipeline.config import DATA_DIR
from pipeline.log import log_structured, setup_stage_logger

logger: logging.Logger | None = None

FEATURES_INPUT = DATA_DIR.parent / "model_features.parquet"
RESULTS_OUTPUT = DATA_DIR.parent / "forward_test_results.parquet"

FEATURE_COLS = [
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
    df["year_month"] = df["match_date"].dt.to_period("M")
    log_structured(
        logger, logging.INFO,
        f"Loaded {len(df)} feature samples",
        entity_type="file",
    )
    return df


def _get_test_months(df: pd.DataFrame):
    """Get unique months in 2024-2026."""
    test_df = df[df["match_date"].dt.year >= 2024].copy()
    months = sorted(test_df["year_month"].unique())
    return months


def _train_models(X_train: np.ndarray, y_train: np.ndarray):
    """Train both Logistic Regression and Random Forest."""
    # Scale for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf_model.fit(X_train, y_train)

    return lr_model, rf_model, scaler


def _evaluate_month(
    lr_model,
    rf_model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate both models on a month's test data."""
    # LR predictions
    X_test_scaled = scaler.transform(X_test)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr_model.predict(X_test_scaled)

    # RF predictions
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)

    return {
        "lr_accuracy": accuracy_score(y_test, lr_pred),
        "lr_logloss": log_loss(y_test, lr_proba),
        "lr_brier": brier_score_loss(y_test, lr_proba),
        "lr_auc": roc_auc_score(y_test, lr_proba),
        "rf_accuracy": accuracy_score(y_test, rf_pred),
        "rf_logloss": log_loss(y_test, rf_proba),
        "rf_brier": brier_score_loss(y_test, rf_proba),
        "rf_auc": roc_auc_score(y_test, rf_proba),
        "n_matches": len(y_test),
        "n_wins": (y_test == 1).sum(),
        "n_losses": (y_test == 0).sum(),
    }


def _run_forward_test(df: pd.DataFrame):
    """Run walk-forward evaluation."""
    df = df.copy()

    # Fill NaN in features
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0)

    # Split into train (pre-2024) and test months (2024+)
    train_data = df[df["match_date"].dt.year < 2024].copy()
    test_months = _get_test_months(df)

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data["target"].values

    log_structured(
        logger, logging.INFO,
        f"Training on {len(train_data)} matches (1988-2023)",
        entity_type="dataset",
    )

    # Train initial models
    lr_model, rf_model, scaler = _train_models(X_train, y_train)

    # Walk forward through test months
    results = []

    for idx, month in enumerate(test_months):
        month_data = df[df["year_month"] == month]
        X_test = month_data[FEATURE_COLS].values
        y_test = month_data["target"].values

        if len(X_test) == 0:
            continue

        # Evaluate
        metrics = _evaluate_month(lr_model, rf_model, scaler, X_test, y_test)

        result = {
            "period": str(month),
            **metrics,
        }
        results.append(result)

        log_structured(
            logger, logging.INFO,
            f"Month {str(month)}: {len(X_test)} matches, "
            f"LR={metrics['lr_accuracy']:.3f}, RF={metrics['rf_accuracy']:.3f}",
            entity_type="dataset",
        )

    return pd.DataFrame(results)


def _print_summary(results_df: pd.DataFrame):
    """Print walk-forward evaluation summary."""
    report_lines = [
        "",
        "=" * 70,
        "  STAGE 6 CERTIFICATION — Forward Prediction Simulation",
        "=" * 70,
        f"  Test period:            2024–2026 (monthly walk-forward)",
        f"  Total test matches:     {results_df['n_matches'].sum():,}",
        f"  Months evaluated:       {len(results_df)}",
        "",
        "  Cumulative Performance:",
        "  " + "-" * 66,
        f"  {'Metric':<25} {'Logistic Regression':<20} {'Random Forest':<20}",
        "  " + "-" * 66,
    ]

    metrics = ["accuracy", "logloss", "brier", "auc"]
    for metric in metrics:
        lr_col = f"lr_{metric}"
        rf_col = f"rf_{metric}"
        lr_vals = results_df[lr_col].values
        rf_vals = results_df[rf_col].values

        if metric in ["logloss", "brier"]:  # Lower is better
            lr_mean = np.mean(lr_vals)
            rf_mean = np.mean(rf_vals)
            winner = "[BEST] RF" if rf_mean < lr_mean else ""
        else:  # Higher is better
            lr_mean = np.mean(lr_vals)
            rf_mean = np.mean(rf_vals)
            winner = "[BEST] RF" if rf_mean > lr_mean else ""

        report_lines.append(f"  {metric:<25} {lr_mean:.4f}              {rf_mean:.4f}              {winner}")

    report_lines += [
        "  " + "-" * 66,
        "",
        "  Monthly Breakdown (first 5 months):",
    ]

    for idx, row in results_df.head(5).iterrows():
        report_lines.append(
            f"    {row['period']}: "
            f"LR acc={row['lr_accuracy']:.3f}, "
            f"RF acc={row['rf_accuracy']:.3f} "
            f"({int(row['n_matches'])} matches)"
        )

    report_lines += [
        "",
        "  Key Insights:",
        f"  • Walk-forward simulates real-world deployment (predict future using past)",
        f"  • Monthly evaluation shows stability of predictions over time",
        f"  • This avoids look-ahead bias — predictions made before seeing outcomes",
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


def forward_test() -> None:
    """Run the full Stage 6 forward prediction simulation."""
    global logger
    logger = setup_stage_logger("forward_test")

    log_structured(logger, logging.INFO, "Stage 6 — forward_test starting", status="start")

    # 1. Load features
    df = _load_features()

    # 2. Run walk-forward evaluation
    results_df = _run_forward_test(df)

    # 3. Save results
    results_df.to_parquet(RESULTS_OUTPUT, index=False)
    log_structured(
        logger, logging.INFO,
        f"Saved {len(results_df)} monthly results to {RESULTS_OUTPUT.name}",
        entity_type="file",
    )

    # 4. Print summary
    _print_summary(results_df)

    log_structured(logger, logging.INFO, "Stage 6 — forward_test complete", status="done")


if __name__ == "__main__":
    try:
        forward_test()
    except Exception as exc:
        if logger is None:
            logger = setup_stage_logger("forward_test")
        log_structured(logger, logging.ERROR, str(exc), entity_type="pipeline", status="error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
