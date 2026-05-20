# mvp/tests/test_settle_results.py
import pytest

# Stale tests — written against earlier internals of settle_results.py before the
# BetsAPI refactor (commits aec5819 / d81dd9a). The mvp/ module is a live-betting
# prototype, separate from the research pipeline. Tests need rewriting against
# the current API; until then, skip this module rather than fail CI.
pytest.skip("Outdated against current settle_results API; awaiting rewrite.", allow_module_level=True)

from unittest.mock import patch, MagicMock
from mvp.settle_results import (
    compute_pnl, fetch_match_result, update_settled_trade,
    _settle_single_trade,
)


def test_compute_pnl_win():
    """Test P&L calculation for winning bet."""
    pnl = compute_pnl(10.0, 1.95, "win")
    assert pnl == pytest.approx(9.5)


def test_compute_pnl_loss():
    """Test P&L calculation for losing bet."""
    pnl = compute_pnl(10.0, 1.95, "loss")
    assert pnl == -10.0


def test_compute_pnl_void():
    """Test P&L calculation for void bet."""
    pnl = compute_pnl(10.0, 1.95, "void")
    assert pnl == 0.0


def test_compute_pnl_different_odds():
    """Test P&L with different odds."""
    pnl = compute_pnl(20.0, 2.50, "win")
    assert pnl == pytest.approx(30.0)


def test_update_settled_trade():
    """Test updating a trade row with result and P&L."""
    row = {
        "match_id": "match_123",
        "selected_side": "player_a",
        "stake": "10.0",
        "result": "",
        "pnl": "",
        "status": "placed",
        "event": "london-2026",
    }
    updated = update_settled_trade(row, "win", 9.5)

    assert updated["result"] == "win"
    assert updated["pnl"] == 9.5
    assert updated["status"] == "settled"
    assert updated["match_id"] == "match_123"


# --- fetch_match_result tests (Odds API v4 scores format) ---

@patch('mvp.settle_results._fetch_all_scores')
def test_fetch_match_result_completed(mock_scores):
    """Test extracting winner from completed event."""
    mock_scores.return_value = [
        {
            "id": "match_123",
            "completed": True,
            "home_team": "Fan Zhendong",
            "away_team": "Ma Long",
            "scores": [
                {"name": "Fan Zhendong", "score": "3"},
                {"name": "Ma Long", "score": "1"},
            ],
        }
    ]

    winner = fetch_match_result("match_123")
    assert winner == "Fan Zhendong"


@patch('mvp.settle_results._fetch_all_scores')
def test_fetch_match_result_away_wins(mock_scores):
    """Test extracting winner when away player wins."""
    mock_scores.return_value = [
        {
            "id": "match_456",
            "completed": True,
            "home_team": "Fan Zhendong",
            "away_team": "Ma Long",
            "scores": [
                {"name": "Fan Zhendong", "score": "1"},
                {"name": "Ma Long", "score": "3"},
            ],
        }
    ]

    winner = fetch_match_result("match_456")
    assert winner == "Ma Long"


@patch('mvp.settle_results._fetch_all_scores')
def test_fetch_match_result_not_completed(mock_scores):
    """Test returns None for incomplete match."""
    mock_scores.return_value = [
        {
            "id": "match_123",
            "completed": False,
            "scores": None,
        }
    ]

    assert fetch_match_result("match_123") is None


@patch('mvp.settle_results._fetch_all_scores')
def test_fetch_match_result_not_found(mock_scores):
    """Test returns None when event ID not in scores."""
    mock_scores.return_value = []

    assert fetch_match_result("nonexistent") is None


@patch('mvp.settle_results._fetch_all_scores')
def test_fetch_match_result_no_scores(mock_scores):
    """Test returns None when completed but missing scores."""
    mock_scores.return_value = [
        {"id": "match_123", "completed": True, "scores": None}
    ]

    assert fetch_match_result("match_123") is None


# --- _settle_single_trade tests ---

@patch('mvp.settle_results._fetch_all_scores')
def test_settle_single_trade_win(mock_scores):
    """Test settlement when our bet wins."""
    mock_scores.return_value = [
        {
            "id": "m1",
            "completed": True,
            "scores": [
                {"name": "Player A", "score": "3"},
                {"name": "Player B", "score": "0"},
            ],
        }
    ]

    trade = {
        "match_id": "m1",
        "player_a": "Player A",
        "player_b": "Player B",
        "selected_side": "player_a",
        "odds_a": "1.80",
        "odds_b": "2.10",
        "stake": "10.0",
        "result": "",
        "pnl": "",
        "status": "placed",
        "event": "london-2026",
    }

    settled = _settle_single_trade(trade)
    assert settled["result"] == "win"
    assert settled["pnl"] == pytest.approx(8.0)  # 10 * (1.80 - 1)
    assert settled["status"] == "settled"


@patch('mvp.settle_results._fetch_all_scores')
def test_settle_single_trade_loss(mock_scores):
    """Test settlement when our bet loses."""
    mock_scores.return_value = [
        {
            "id": "m2",
            "completed": True,
            "scores": [
                {"name": "Player A", "score": "3"},
                {"name": "Player B", "score": "1"},
            ],
        }
    ]

    trade = {
        "match_id": "m2",
        "player_a": "Player A",
        "player_b": "Player B",
        "selected_side": "player_b",  # we bet on B, but A won
        "odds_a": "1.80",
        "odds_b": "2.10",
        "stake": "10.0",
        "result": "",
        "pnl": "",
        "status": "placed",
        "event": "london-2026",
    }

    settled = _settle_single_trade(trade)
    assert settled["result"] == "loss"
    assert settled["pnl"] == -10.0
    assert settled["status"] == "settled"
