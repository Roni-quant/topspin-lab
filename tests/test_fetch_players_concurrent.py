"""Tests for concurrent player fetching in fetch_players."""

import threading
from unittest.mock import patch

import pandas as pd

from pipeline.http import RateLimiter


def test_fetch_players_uses_thread_pool(tmp_path):
    """Verify that fetching multiple players uses concurrent workers."""
    from pipeline import fetch_players

    call_thread_ids = []
    main_thread = threading.current_thread().ident

    def mock_fetch_player(player_id, headers):
        call_thread_ids.append(threading.current_thread().ident)
        return {"player_id": player_id, "player_name": f"Player {player_id}",
                "birth_year": None, "association": None, "hand": None,
                "grip": None, "style": None}

    limiter = RateLimiter(min_interval=0.0)
    ids = list(range(1, 11))  # 10 players
    headers = {"Cookie": "test"}
    existing_df = pd.DataFrame(columns=["player_id", "player_name",
                                         "birth_year", "association",
                                         "hand", "grip", "style"])

    with patch.object(fetch_players, "fetch_player", side_effect=mock_fetch_player):
        records, failed = fetch_players._fetch_players_concurrent(
            ids, headers, limiter, existing_df, tmp_path / "players.parquet"
        )

    assert len(records) == 10
    assert failed == 0
    # At least some calls should be on worker threads (not main thread)
    worker_calls = [tid for tid in call_thread_ids if tid != main_thread]
    assert len(worker_calls) > 0


def test_fetch_players_handles_failures(tmp_path):
    """Failed player fetches should be counted, not crash the pool."""
    from pipeline import fetch_players

    def mock_fetch_player(player_id, headers):
        if player_id % 2 == 0:
            return None  # simulate failure
        return {"player_id": player_id, "player_name": f"P{player_id}",
                "birth_year": None, "association": None, "hand": None,
                "grip": None, "style": None}

    limiter = RateLimiter(min_interval=0.0)
    ids = list(range(1, 7))  # 6 players, 3 will fail
    headers = {"Cookie": "test"}
    existing_df = pd.DataFrame(columns=["player_id", "player_name",
                                         "birth_year", "association",
                                         "hand", "grip", "style"])

    with patch.object(fetch_players, "fetch_player", side_effect=mock_fetch_player):
        records, failed = fetch_players._fetch_players_concurrent(
            ids, headers, limiter, existing_df, tmp_path / "players.parquet"
        )

    assert len(records) == 3
    assert failed == 3


def test_fetch_players_checkpoint_saves(tmp_path):
    """Checkpoint saves should write partial results to disk."""
    from pipeline import fetch_players

    output_path = tmp_path / "players.parquet"

    def mock_fetch_player(player_id, headers):
        return {"player_id": player_id, "player_name": f"P{player_id}",
                "birth_year": None, "association": None, "hand": None,
                "grip": None, "style": None}

    limiter = RateLimiter(min_interval=0.0)
    # 600 players to trigger at least 1 checkpoint (every 500)
    ids = list(range(1, 601))
    headers = {"Cookie": "test"}
    existing_df = pd.DataFrame(columns=["player_id", "player_name",
                                         "birth_year", "association",
                                         "hand", "grip", "style"])

    with patch.object(fetch_players, "fetch_player", side_effect=mock_fetch_player):
        with patch.object(fetch_players, "validate_schema", side_effect=lambda df, _: df):
            records, failed = fetch_players._fetch_players_concurrent(
                ids, headers, limiter, existing_df, output_path
            )

    assert len(records) == 600
    # Checkpoint file should exist (written at i=500)
    assert output_path.exists()
