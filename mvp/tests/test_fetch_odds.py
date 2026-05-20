# mvp/tests/test_fetch_odds.py
import pytest
from unittest.mock import patch, MagicMock
from mvp.fetch_odds import parse_odds_response, build_trade_row

def test_parse_odds_response():
    """Test parsing odds API response."""
    # Mock response from The Odds API
    mock_odds = {
        "id": "london_2026_001",
        "home_team": "Lin Gaoyuan",
        "away_team": "Hugo Calderano",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Lin Gaoyuan", "price": 1.95},
                            {"name": "Hugo Calderano", "price": 1.85},
                        ]
                    }
                ]
            }
        ]
    }

    odds_a, odds_b = parse_odds_response(mock_odds)

    assert odds_a == 1.95
    assert odds_b == 1.85

def test_build_trade_row():
    """Test building a trade row from odds."""
    row = build_trade_row(
        match_id="london_2026_001",
        player_a="Lin Gaoyuan",
        player_b="Hugo Calderano",
        odds_a=1.95,
        odds_b=1.85,
    )

    assert row["match_id"] == "london_2026_001"
    assert row["player_a"] == "Lin Gaoyuan"
    assert row["odds_a"] == 1.95
    assert row["status"] == "pending"

def test_parse_odds_response_multiple_bookmakers():
    """Test parsing odds from multiple bookmakers."""
    mock_odds = {
        "id": "london_2026_002",
        "home_team": "Fan Zhendong",
        "away_team": "Tomokazu Harimoto",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Fan Zhendong", "price": 1.80},
                            {"name": "Tomokazu Harimoto", "price": 2.05},
                        ]
                    }
                ]
            },
            {
                "key": "betfair",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Fan Zhendong", "price": 1.75},
                            {"name": "Tomokazu Harimoto", "price": 2.10},
                        ]
                    }
                ]
            }
        ]
    }

    odds_a, odds_b = parse_odds_response(mock_odds)

    # Should return odds from first bookmaker
    assert odds_a == 1.80
    assert odds_b == 2.05

def test_build_trade_row_all_fields():
    """Test that build_trade_row includes all required fields."""
    row = build_trade_row(
        match_id="match_999",
        player_a="Player A",
        player_b="Player B",
        odds_a=2.10,
        odds_b=1.75,
    )

    # Verify required fields
    assert "match_id" in row
    assert "fetch_ts" in row
    assert "player_a" in row
    assert "player_b" in row
    assert "odds_a" in row
    assert "odds_b" in row
    assert "implied_prob_a" in row
    assert "implied_prob_b" in row
    assert "model_prob_a" in row
    assert "model_prob_b" in row
    assert "selected_side" in row
    assert "edge" in row
    assert "stake" in row
    assert "result" in row
    assert "pnl" in row
    assert "status" in row
    assert "event" in row

@patch('mvp.fetch_odds.requests.get')
def test_fetch_odds_from_api(mock_get):
    """Test fetching odds from API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "events": [
            {
                "id": "london_001",
                "home_team": "Player A",
                "away_team": "Player B",
                "competition": "London 2026",
                "bookmakers": [],
            }
        ]
    }
    mock_get.return_value = mock_response

    from mvp.fetch_odds import fetch_odds_from_api
    events = fetch_odds_from_api()

    assert len(events) >= 1
    assert events[0]["competition"] == "London 2026"

def test_parse_odds_response_no_bookmakers():
    """Test parsing odds when no bookmakers available."""
    event = {
        "id": "london_002",
        "home_team": "Player A",
        "away_team": "Player B",
        "bookmakers": [],
    }

    odds_a, odds_b = parse_odds_response(event)

    assert odds_a is None
    assert odds_b is None

def test_parse_odds_response_no_markets():
    """Test parsing odds when bookmaker has no markets."""
    event = {
        "id": "london_003",
        "home_team": "Player A",
        "away_team": "Player B",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [],
            }
        ],
    }

    odds_a, odds_b = parse_odds_response(event)

    assert odds_a is None
    assert odds_b is None

def test_parse_odds_response_incomplete_outcomes():
    """Test parsing odds with incomplete outcomes."""
    event = {
        "id": "london_004",
        "home_team": "Player A",
        "away_team": "Player B",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Player A", "price": 1.95},
                        ]
                    }
                ]
            }
        ],
    }

    odds_a, odds_b = parse_odds_response(event)

    assert odds_a is None
    assert odds_b is None

def test_build_trade_row_has_event():
    """Test that build_trade_row includes event field from ODDS_LEAGUE."""
    from mvp.config import ODDS_LEAGUE
    row = build_trade_row(
        match_id="match_001",
        player_a="Player A",
        player_b="Player B",
        odds_a=1.95,
        odds_b=1.85,
    )
    assert row["event"] == ODDS_LEAGUE
