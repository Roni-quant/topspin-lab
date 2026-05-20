"""Tests for concurrent match fetching in fetch_matches."""

import logging
import threading
from datetime import date
from unittest.mock import patch

import pandas as pd

from pipeline.http import RateLimiter


def test_fetch_event_matches_uses_limiter():
    """fetch_event_matches should call limiter.wait() before each page request."""
    from pipeline import fetch_matches

    wait_count = 0

    class CountingLimiter(RateLimiter):
        def wait(self):
            nonlocal wait_count
            wait_count += 1

    limiter = CountingLimiter(min_interval=0.0)
    logger = logging.getLogger("test")

    # Mock fetch_json to return one page of results then empty
    call_count = 0

    def mock_fetch_json(url, headers):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [[{
                "vw_matches___event_raw": "MS",
                "vw_matches___id_raw": "1",
                "__pk_val": "1",
                "vw_matches___player_a_id_raw": "100",
                "vw_matches___name_a_raw": "Player A",
                "vw_matches___player_x_id_raw": "200",
                "vw_matches___name_x_raw": "Player B",
                "vw_matches___res_raw": "3:1",
                "vw_matches___games_raw": "11:5 11:7 5:11 11:8",
            }]]
        return [[]]  # empty page, stops pagination

    with patch("pipeline.fetch_matches.fetch_json", side_effect=mock_fetch_json):
        matches = fetch_matches.fetch_event_matches(
            event_id=1, tournament_id=1, event_name="Test",
            event_date=date(2024, 1, 1), event_category="MS",
            headers={"Cookie": "test"}, logger=logger, limiter=limiter,
        )

    assert len(matches) >= 1
    assert wait_count >= 1  # limiter.wait() was called


def test_concurrent_event_fetching():
    """Multiple events should be fetched concurrently."""
    from pipeline import fetch_matches

    call_thread_ids = []
    main_thread = threading.current_thread().ident

    def mock_fetch_event_matches(event_id, tournament_id, event_name,
                                  event_date, event_category, headers,
                                  logger, limiter):
        call_thread_ids.append(threading.current_thread().ident)
        return [{"match_key": f"key_{event_id}", "event_id": event_id,
                 "match_date": event_date, "drop_reason": None}]

    limiter = RateLimiter(min_interval=0.0)
    logger = logging.getLogger("test")

    rows = [
        {"event_id": i, "tournament_id": i, "event_name": f"E{i}",
         "event_date": pd.Timestamp(f"2024-01-{i+1:02d}"),
         "event_category": "MS"}
        for i in range(1, 6)
    ]

    with patch.object(fetch_matches, "fetch_event_matches",
                      side_effect=mock_fetch_event_matches):
        matches, processed = fetch_matches._fetch_events_concurrent(
            rows, {"Cookie": "test"}, limiter, logger,
        )

    assert processed == 5
    # At least some calls should be on worker threads
    worker_calls = [tid for tid in call_thread_ids if tid != main_thread]
    assert len(worker_calls) > 0
