# mvp/models.py
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_model(model_path: Path):
    """
    Load trained Random Forest model from disk.

    Args:
        model_path (Path): Absolute path to pickled Random Forest model file

    Returns:
        RandomForestClassifier: Trained model object ready for predictions

    Raises:
        FileNotFoundError: If model_path does not exist on disk
        pickle.UnpicklingError: If file exists but is not a valid pickle

    Example:
        >>> from pathlib import Path
        >>> model = load_model(Path("models/random_forest_v2.pkl"))
        >>> print(model.n_estimators)  # Access model properties
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded model from {model_path}")
    return model

def predict(model, features: dict) -> float:
    """
    Predict probability that player A wins using trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained Random Forest classifier
        features (dict): Feature dictionary with keys matching FEATURE_NAMES config.
                        Keys should include: elo_difference, recent_win_rate,
                        matches_last_30_days, opponent_recent_form, momentum

    Returns:
        float: Probability [0.0-1.0] that player A wins the match

    Raises:
        KeyError: If required feature keys are missing from features dict
        ValueError: If model cannot generate predictions

    Example:
        >>> features = {
        ...     "elo_difference": 50.0,
        ...     "recent_win_rate": 0.65,
        ...     "matches_last_30_days": 8.0,
        ...     "opponent_recent_form": 0.60,
        ...     "momentum": 0.05,
        ... }
        >>> prob_a_wins = predict(model, features)
        >>> prob_b_wins = 1.0 - prob_a_wins
    """
    # Order features to match training
    from mvp.config import FEATURE_NAMES

    feature_vector = [features[name] for name in FEATURE_NAMES]

    # predict_proba returns [[prob_loss, prob_win]]
    prob_win = model.predict_proba([feature_vector])[0][1]

    return prob_win
