# mvp/tests/test_model_loading.py
import pytest
from pathlib import Path
from mvp.models import load_model, predict

def test_model_loads():
    """Test that model loads without error."""
    from mvp.config import MODEL_PATH

    # This will fail initially (model path incorrect)
    # Fix by adjusting MODEL_PATH in config.py
    model = load_model(MODEL_PATH)
    assert model is not None

def test_predict_returns_valid_probability():
    """Test that predict returns [0-1]."""
    from mvp.config import MODEL_PATH, FEATURE_NAMES

    model = load_model(MODEL_PATH)

    features = {name: 0.5 for name in FEATURE_NAMES}
    features["elo_difference"] = 50.0

    prob = predict(model, features)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
