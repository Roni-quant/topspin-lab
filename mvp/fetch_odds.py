# mvp/fetch_odds.py
import requests
import json
from datetime import datetime
from pathlib import Path
import logging

from mvp.config import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, ODDS_LEAGUE,
    DATA_DIR, TRADES_CSV, PLAYER_MAPPING, TradeStatus
)
from mvp.utils import append_to_csv

logger = logging.getLogger(__name__)

def fetch_odds_from_api() -> list:
    """
    Query The Odds API for London 2026 World Team Championships matches.

    Retrieves table tennis match odds from The Odds API filtered for the
    London 2026 tournament (April 28 - May 10). Uses US bookmakers and
    head-to-head markets.

    Returns:
        List[Dict]: List of event dictionaries from The Odds API, filtered
                   for London 2026 tournament. Empty list if no matches found.

    Raises:
        requests.HTTPError: If API request fails (bad status code)
        requests.JSONDecodeError: If response is not valid JSON
        KeyError: If ODDS_API_KEY or ODDS_API_BASE config is missing

    Example:
        >>> events = fetch_odds_from_api()
        >>> print(f"Found {len(events)} London 2026 matches")
        >>> for event in events:
        ...     print(event["id"], event["home_team"], "vs", event["away_team"])
    """
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",  # Use US bookmakers
        "markets": "h2h",  # Head-to-head markets
    }

    logger.info(f"Fetching odds from {url}")
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    events = data.get("events", [])

    # Filter for London 2026 (by league/tournament name or date range)
    # This logic depends on The Odds API response structure
    london_events = [
        e for e in events
        if "london" in e.get("competition", "").lower()
        or "world team" in e.get("competition", "").lower()
    ]

    logger.info(f"Found {len(london_events)} London 2026 matches")
    return london_events

def parse_odds_response(event: dict) -> tuple:
    """
    Extract decimal odds for both players from The Odds API response.

    Parses head-to-head odds from the first available bookmaker in the event.
    Returns (None, None) if odds are incomplete.

    Args:
        event (Dict): Single event dictionary from The Odds API, containing
                     'id', 'bookmakers' with 'markets', 'outcomes', 'price'

    Returns:
        Tuple[float, float]: (odds_a, odds_b) as decimal odds, or (None, None)
                            if odds cannot be parsed

    Example:
        >>> event = {
        ...     "id": "evt_123",
        ...     "bookmakers": [{
        ...         "markets": [{
        ...             "outcomes": [
        ...                 {"price": 1.95},
        ...                 {"price": 1.85}
        ...             ]
        ...         }]
        ...     }]
        ... }
        >>> odds_a, odds_b = parse_odds_response(event)
        >>> print(f"Player A: {odds_a}, Player B: {odds_b}")
        Player A: 1.95, Player B: 1.85
    """
    bookmakers = event.get("bookmakers", [])

    if not bookmakers:
        logger.warning(f"No bookmakers for {event['id']}")
        return None, None

    # Use first available bookmaker
    bookie = bookmakers[0]
    markets = bookie.get("markets", [])

    if not markets:
        logger.warning(f"No markets for {event['id']}")
        return None, None

    h2h = markets[0]  # h2h market
    outcomes = h2h.get("outcomes", [])

    if len(outcomes) < 2:
        logger.warning(f"Incomplete outcomes for {event['id']}")
        return None, None

    odds_a = outcomes[0]["price"]
    odds_b = outcomes[1]["price"]

    return float(odds_a), float(odds_b)

def build_trade_row(match_id: str, player_a: str, player_b: str,
                    odds_a: float, odds_b: float) -> dict:
    """
    Build a complete trade row dictionary for insertion into trades.csv.

    Initializes all fields with appropriate values or empty strings. Model
    predictions and outcomes are filled in by downstream scripts
    (predict_and_log.py, settle_results.py).

    Args:
        match_id (str): The Odds API event ID
        player_a (str): Player A name
        player_b (str): Player B name
        odds_a (float): Decimal odds for player A
        odds_b (float): Decimal odds for player B

    Returns:
        Dict: Trade dictionary with 17 keys ready to write to CSV

    Example:
        >>> row = build_trade_row("evt_123", "Fan Zhendong", "Tomokazu Harimoto", 1.95, 1.85)
        >>> print(row["match_id"])
        evt_123
        >>> print(row["status"])
        pending
    """
    from mvp.utils import compute_implied_probability

    return {
        "match_id": match_id,
        "fetch_ts": datetime.utcnow().isoformat() + "Z",
        "player_a": player_a,
        "player_b": player_b,
        "odds_a": odds_a,
        "odds_b": odds_b,
        "model_prob_a": "",  # Filled by predict_and_log.py
        "model_prob_b": "",
        "implied_prob_a": compute_implied_probability(odds_a),
        "implied_prob_b": compute_implied_probability(odds_b),
        "selected_side": "",
        "edge": "",
        "stake": "",
        "result": "",
        "pnl": "",
        "status": TradeStatus.PENDING.value,
        "event": ODDS_LEAGUE,
    }

def main():
    """
    Fetch London 2026 match odds and append pending trades to trades.csv.

    Entry point for odds fetching script. Queries The Odds API for London 2026
    table tennis matches, parses odds, and appends each match as a pending
    trade to trades.csv for later prediction and settlement.

    Returns:
        None (writes to trades.csv)

    Raises:
        requests.HTTPError: If API request fails
        IOError: If trades.csv cannot be written

    Example:
        >>> python mvp/fetch_odds.py
        # Outputs:
        # INFO: Fetching odds from https://api.the-odds-api.com/v4/sports/...
        # INFO: Found 12 London 2026 matches
        # INFO: Appended to mvp/data/trades.csv: evt_123
        # INFO: Odds fetch complete
    """
    logging.basicConfig(level=logging.INFO)

    events = fetch_odds_from_api()

    for event in events:
        match_id = event["id"]
        player_a = event["home_team"]  # Adjust to actual field name
        player_b = event["away_team"]

        odds_a, odds_b = parse_odds_response(event)

        if odds_a is None or odds_b is None:
            logger.warning(f"Skipping {match_id} (incomplete odds)")
            continue

        row = build_trade_row(match_id, player_a, player_b, odds_a, odds_b)
        append_to_csv(TRADES_CSV, row)

    logger.info("Odds fetch complete")

if __name__ == "__main__":
    main()
