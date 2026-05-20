# mvp/settle_results.py
import csv
import requests
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from mvp.config import TRADES_CSV, ODDS_API_KEY, ODDS_API_BASE, TradeStatus, BetResult, TRADES_CSV_FIELDNAMES
from mvp.utils import write_trades_csv

logger = logging.getLogger(__name__)

def fetch_match_result(match_id: str) -> str:
    """
    Fetch match result from The Odds API for a specific match.

    Queries The Odds API to retrieve match outcome (win/loss/void). Only returns
    result if match status is "completed". Returns None for in-progress or
    cancelled matches.

    Args:
        match_id (str): The Odds API event ID

    Returns:
        str: "win", "loss", or None if match not yet completed

    Raises:
        requests.HTTPError: If API request fails
        requests.JSONDecodeError: If response is not valid JSON

    Note:
        Parsing depends on The Odds API response structure for table tennis.
        May need adjustment based on actual API response format.

    Example:
        >>> result = fetch_match_result("evt_123")
        >>> if result:
        ...     print(f"Match completed: {result}")
        >>> else:
        ...     print("Match still in progress")
    """
    url = f"{ODDS_API_BASE}/sports/table-tennis/events/{match_id}"
    params = {"apiKey": ODDS_API_KEY}

    response = requests.get(url, params=params)
    response.raise_for_status()

    event = response.json()

    # Extract winner from response
    # Structure depends on The Odds API response
    status = event.get("status")
    scores = event.get("scores", {})

    if status != "completed":
        return None

    # TODO: Parse winner from scores based on API structure
    # For MVP, implement once API response structure is known
    raise NotImplementedError(
        "Result parsing not implemented. Phase 0 task: inspect actual Odds API response structure "
        "for table tennis to determine how to extract winner from 'scores' field."
    )

def compute_pnl(stake: float, odds: float, result: str) -> float:
    """
    Compute profit/loss for a settled bet.

    Uses standard sports betting math:
    - Win: profit = stake * (odds - 1)
    - Loss: loss = -stake
    - Void/No Result: pnl = 0.0

    Args:
        stake (float): Bet amount in USD
        odds (float): Decimal odds at time of bet
        result (str): "win", "loss", or "void"

    Returns:
        float: Profit/loss in USD (can be positive or negative)

    Example:
        >>> win_pnl = compute_pnl(10.0, 2.0, "win")
        >>> print(win_pnl)
        10.0  # $10 stake at 2.0 odds = $10 profit
        >>> loss_pnl = compute_pnl(10.0, 2.0, "loss")
        >>> print(loss_pnl)
        -10.0  # Lost entire stake
    """
    if result == BetResult.WIN.value:
        return stake * (odds - 1)
    elif result == BetResult.LOSS.value:
        return -stake
    else:
        return 0.0

def update_settled_trade(row: dict, result: str, pnl: float) -> dict:
    """
    Update a trade row with match result and computed profit/loss.

    Changes status from "placed" to "settled" and fills in result and pnl fields.

    Args:
        row (Dict): Trade row from trades.csv in "placed" status
        result (str): "win", "loss", or "void"
        pnl (float): Computed profit/loss in USD

    Returns:
        Dict: Updated trade row with result and pnl fields populated

    Example:
        >>> row = {"match_id": "evt_123", "status": "placed", "pnl": ""}
        >>> settled = update_settled_trade(row, "win", 10.0)
        >>> print(settled["status"])
        settled
        >>> print(settled["pnl"])
        10.0
    """
    row.update({
        "result": result,
        "pnl": pnl,
        "status": TradeStatus.SETTLED.value,
    })
    return row

def _settle_single_trade(trade: dict) -> dict:
    """
    Process a single trade: fetch result, compute P&L, and update trade row.

    Helper function for parallel settlement processing. Designed to be called
    from ThreadPoolExecutor to check multiple matches concurrently.

    Args:
        trade (Dict): Trade row from trades.csv in "placed" status

    Returns:
        Dict: Updated trade row with result and pnl if match is completed,
              original trade row unchanged if match still in progress

    Example:
        >>> trade = {"match_id": "evt_123", "status": "placed", ...}
        >>> settled = _settle_single_trade(trade)
        >>> print(settled["status"])
        settled  # if match was completed
    """
    match_id = trade["match_id"]

    # Try to fetch result
    try:
        result = fetch_match_result(match_id)
    except Exception as e:
        logger.warning(f"{match_id}: Error fetching result: {e}")
        return trade

    if result is None:
        logger.debug(f"{match_id}: Not yet completed, skipping")
        return trade

    # Compute P&L
    stake = float(trade["stake"]) if trade["stake"] else 0.0
    odds = float(trade["odds_a"]) if trade["selected_side"] == "player_a" \
           else float(trade["odds_b"])

    pnl = compute_pnl(stake, odds, result)

    # Update trade
    settled = update_settled_trade(trade.copy(), result, pnl)
    logger.info(f"{match_id}: {result} {pnl:+.2f}")

    return settled

def main():
    """
    Fetch completed match results and settle all outstanding bets.

    Entry point for settlement script. Polls The Odds API for results of all
    placed but unsettled bets, computes P&L for each, and updates trades.csv
    with results and status = "settled".

    Returns:
        None (writes updated trades.csv)

    Raises:
        FileNotFoundError: If TRADES_CSV doesn't exist
        requests.HTTPError: If API requests fail
        ValueError: If CSV is corrupted or missing required columns

    Example:
        >>> python mvp/settle_results.py
        # Outputs:
        # INFO: Checking 5 placed trades for results
        # INFO: evt_123: win +10.00
        # INFO: Updated trades.csv with results
    """
    logging.basicConfig(level=logging.INFO)

    # Read all trades
    trades = []
    with open(TRADES_CSV, "r") as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    # Filter for placed but unsettled trades
    placed = [
        t for t in trades
        if t["status"] == TradeStatus.PLACED.value and t["result"] == ""
    ]

    logger.info(f"Checking {len(placed)} placed trades for results")

    # Update trades with results using parallel processing
    settled_count = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {executor.submit(_settle_single_trade, trade): trade for trade in placed}

        # Process completed tasks and update trade list
        for future in as_completed(futures):
            settled_trade = future.result()
            original_trade = futures[future]

            # Update the trade in the main list
            trade_idx = trades.index(original_trade)
            trades[trade_idx] = settled_trade

            if settled_trade["status"] == TradeStatus.SETTLED.value:
                settled_count += 1

    logger.info(f"Settled {settled_count} / {len(placed)} trades")

    # Rewrite CSV using utility function
    write_trades_csv(TRADES_CSV, trades)

    logger.info("Updated trades.csv with results")

if __name__ == "__main__":
    main()
