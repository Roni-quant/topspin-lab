# mvp/tests/test_predict_and_log.py
import pytest
from mvp.predict_and_log import (
    compute_edge_and_select_side,
    should_place_bet,
    update_trade_row,
    load_features,
)

def test_should_place_bet():
    """Test that bets placed only when edge > threshold."""
    # Edge below threshold
    assert should_place_bet(edge=0.02) == False

    # Edge above threshold (3%)
    assert should_place_bet(edge=0.05) == True

def test_compute_edge_and_select_side():
    """Test edge calculation and side selection."""
    model_prob_a = 0.68
    model_prob_b = 0.32
    implied_prob_a = 0.51
    implied_prob_b = 0.49

    side, edge = compute_edge_and_select_side(
        model_prob_a, model_prob_b,
        implied_prob_a, implied_prob_b
    )

    assert side == "player_a"
    assert edge == pytest.approx(0.17, abs=0.01)

def test_compute_edge_and_select_side_player_b():
    """Test edge calculation when player_b has better edge."""
    model_prob_a = 0.45
    model_prob_b = 0.55
    implied_prob_a = 0.60
    implied_prob_b = 0.40

    side, edge = compute_edge_and_select_side(
        model_prob_a, model_prob_b,
        implied_prob_a, implied_prob_b
    )

    assert side == "player_b"
    assert edge == pytest.approx(0.15, abs=0.01)

def test_update_trade_row_with_bet():
    """Test updating trade row when betting."""
    row = {
        "match_id": "match_123",
        "player_a": "player_1",
        "player_b": "player_2",
        "event": "london-2026",
    }

    updated = update_trade_row(
        row,
        model_prob_a=0.65,
        model_prob_b=0.35,
        selected_side="player_a",
        edge=0.10,
        stake=10.0
    )

    assert updated["model_prob_a"] == 0.65
    assert updated["model_prob_b"] == 0.35
    assert updated["selected_side"] == "player_a"
    assert updated["edge"] == 0.10
    assert updated["stake"] == 10.0
    assert updated["status"] == "placed"

def test_update_trade_row_no_bet():
    """Test updating trade row when not betting."""
    row = {
        "match_id": "match_123",
        "player_a": "player_1",
        "player_b": "player_2",
        "event": "london-2026",
    }

    updated = update_trade_row(
        row,
        model_prob_a=0.55,
        model_prob_b=0.45,
        selected_side="",
        edge=0.01,
        stake=0.0
    )

    assert updated["selected_side"] == ""
    assert updated["stake"] == ""
    assert updated["status"] == "pending"

@pytest.mark.xfail(reason="load_features() is a stub; integration with parent feature pipeline is a TODO")
def test_load_features():
    """Test that load_features returns a dict with expected feature keys."""
    from mvp.config import FEATURE_NAMES
    features = load_features("fan_zhendong")
    assert isinstance(features, dict)
    for name in FEATURE_NAMES:
        assert name in features, f"Missing feature: {name}"
    assert isinstance(features["elo_difference"], float)
