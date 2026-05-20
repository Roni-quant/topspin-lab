# mvp/analyze.py
import csv
import json
import logging
from pathlib import Path

from mvp.config import TRADES_CSV, STARTING_CAPITAL, TradeStatus, BetResult

logger = logging.getLogger(__name__)

def compute_metrics(trades: list) -> dict:
    """
    Compute performance metrics from complete trades list.

    Aggregates statistics across all trades: bets placed, settled, won, lost,
    total profit/loss, ROI, win rate, and average edge. Returns zeros for all
    metrics if no bets have settled yet.

    Args:
        trades (List[Dict]): List of trade dictionaries from CSV, with fields:
                            status, result, pnl, stake, edge

    Returns:
        Dict: Performance metrics dictionary with keys:
            - bets_placed (int): Count of placed bets (status in "placed"/"settled")
            - bets_settled (int): Count of completed bets (status = "settled")
            - bets_won (int): Count of winning bets (result = "win")
            - bets_lost (int): Count of losing bets (result = "loss")
            - win_rate_pct (float): (bets_won / bets_settled) * 100
            - total_pnl_usd (float): Sum of all pnl values
            - roi_pct (float): (total_pnl / STARTING_CAPITAL) * 100
            - average_edge_pct (float): Mean edge on placed bets * 100

    Example:
        >>> trades = [
        ...     {"status": "settled", "result": "win", "pnl": "10.0", "stake": "10", "edge": "0.04"},
        ...     {"status": "settled", "result": "loss", "pnl": "-10.0", "stake": "10", "edge": "0.02"},
        ... ]
        >>> metrics = compute_metrics(trades)
        >>> print(f"ROI: {metrics['roi_pct']}%")
        >>> print(f"Win Rate: {metrics['win_rate_pct']}%")
    """
    placed = [t for t in trades if t["status"] in (TradeStatus.PLACED.value, TradeStatus.SETTLED.value)]
    settled = [t for t in trades if t["status"] == TradeStatus.SETTLED.value]

    bets_placed = len(placed)
    bets_settled = len(settled)

    if bets_settled == 0:
        return {
            "bets_placed": bets_placed,
            "bets_settled": 0,
            "bets_won": 0,
            "bets_lost": 0,
            "win_rate_pct": 0.0,
            "total_pnl_usd": 0.0,
            "roi_pct": 0.0,
            "average_edge_pct": 0.0,
        }

    won = [t for t in settled if t["result"] == BetResult.WIN.value]
    lost = [t for t in settled if t["result"] == BetResult.LOSS.value]

    total_pnl = sum(float(t["pnl"]) for t in settled if t["pnl"])
    roi = (total_pnl / STARTING_CAPITAL) * 100

    placed_with_stakes = [t for t in placed if t["stake"]]
    avg_edge = (
        sum(float(t["edge"]) for t in placed_with_stakes if t["edge"])
        / len(placed_with_stakes)
        * 100
    ) if placed_with_stakes else 0.0

    return {
        "bets_placed": bets_placed,
        "bets_settled": bets_settled,
        "bets_won": len(won),
        "bets_lost": len(lost),
        "win_rate_pct": (len(won) / bets_settled * 100) if bets_settled > 0 else 0.0,
        "total_pnl_usd": round(total_pnl, 2),
        "roi_pct": round(roi, 2),
        "average_edge_pct": round(avg_edge, 2),
    }

def main():
    """
    Compute tournament performance metrics and generate summary report.

    Entry point for analysis script. Reads trades.csv, computes performance
    metrics, generates summary.json with tournament metadata and metrics,
    and prints formatted console report.

    Returns:
        None (writes summary.json, prints to stdout)

    Raises:
        FileNotFoundError: If TRADES_CSV doesn't exist
        ValueError: If CSV is corrupted or missing required columns

    Output Files:
        - mvp/data/summary.json: Machine-readable metrics with tournament metadata

    Example:
        >>> python mvp/analyze.py
        # Outputs:
        # ==================================================
        # LONDON 2026 MODEL VALIDATION RESULTS
        # ==================================================
        # Bets Placed:     12
        # Bets Settled:    10
        # Bets Won:        6
        # Bets Lost:       4
        # Win Rate:        60.0%
        # Total P&L:       $15.00
        # ROI:             +1.50%
        # Avg Edge:        3.45%
        # ==================================================
    """
    logging.basicConfig(level=logging.INFO)

    # Read trades
    trades = []
    with open(TRADES_CSV, "r") as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    # Compute metrics
    metrics = compute_metrics(trades)

    # Add metadata
    summary = {
        "tournament": "London 2026",
        "period": "2026-04-28 to 2026-05-10",
        "metrics": metrics,
    }

    # Output summary.json
    summary_path = TRADES_CSV.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary: {metrics}")
    logger.info(f"Saved to {summary_path}")

    # Print to console
    print("\n" + "="*50)
    print("LONDON 2026 MODEL VALIDATION RESULTS")
    print("="*50)
    print(f"Bets Placed:     {metrics['bets_placed']}")
    print(f"Bets Settled:    {metrics['bets_settled']}")
    print(f"Bets Won:        {metrics['bets_won']}")
    print(f"Bets Lost:       {metrics['bets_lost']}")
    print(f"Win Rate:        {metrics['win_rate_pct']:.1f}%")
    print(f"\nTotal P&L:       ${metrics['total_pnl_usd']:+.2f}")
    print(f"ROI:             {metrics['roi_pct']:+.2f}%")
    print(f"Avg Edge:        {metrics['average_edge_pct']:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
