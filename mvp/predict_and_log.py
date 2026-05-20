# mvp/predict_and_log.py
import csv
import logging
from pathlib import Path

from mvp.config import TRADES_CSV, MIN_EDGE_THRESHOLD, MODEL_PATH, TradeStatus, STAKE
from mvp.models import load_model, predict
from mvp.utils import compute_edge

logger = logging.getLogger(__name__)

def load_features(player_id: str) -> dict:
    """
    Load match features for a player from historical Elo and performance data.

    Computes required features from matches_with_elo.parquet and
    matches_clean.parquet. Features include Elo difference, recent win rates,
    match frequency, and momentum indicators.

    Args:
        player_id (str): Internal player ID (from player mapping)

    Returns:
        Dict: Feature dictionary with keys:
            - elo_difference (float): Player's Elo - Opponent's Elo
            - recent_win_rate (float): Win rate last 30 days [0.0-1.0]
            - matches_last_30_days (float): Number of matches in last 30 days
            - opponent_recent_form (float): Opponent win rate [0.0-1.0]
            - momentum (float): Win rate momentum indicator

    Note:
        For MVP, returns placeholder values. Production version should integrate
        with existing feature generation pipeline from parent project.

    Example:
        >>> features = load_features("fan_z_chn")
        >>> print(features["elo_difference"])
        50.0
    """
    raise NotImplementedError(
        "load_features() requires integration with parent project's feature pipeline. "
        "Phase 0 task: connect to parent project's feature generation (matches_with_elo.parquet, "
        "matches_clean.parquet) or provide pre-computed feature cache."
    )

def compute_edge_and_select_side(model_prob_a: float, model_prob_b: float,
                                  implied_prob_a: float,
                                  implied_prob_b: float) -> tuple:
    """
    Determine which side offers better edge and compute the edge value.

    Compares edge on both sides and selects the side with higher expected value.
    Edge = model_prob - implied_prob. Positive edge indicates profitable opportunity.

    Args:
        model_prob_a (float): Model's predicted P(player A wins) [0.0-1.0]
        model_prob_b (float): Model's predicted P(player B wins) [0.0-1.0]
        implied_prob_a (float): Bookmaker's implied P(player A wins) [0.0-1.0]
        implied_prob_b (float): Bookmaker's implied P(player B wins) [0.0-1.0]

    Returns:
        Tuple[str, float]: (selected_side, edge) where:
            - selected_side is "player_a" or "player_b" (whichever has higher edge)
            - edge is the EV edge as decimal [e.g., 0.04 for +4%]

    Example:
        >>> side, edge = compute_edge_and_select_side(0.55, 0.45, 0.51, 0.49)
        >>> print(f"Bet on {side}, edge={edge:.2%}")
        Bet on player_a, edge=4.00%
    """
    edge_a = compute_edge(model_prob_a, implied_prob_a)
    edge_b = compute_edge(model_prob_b, implied_prob_b)

    if edge_a > edge_b:
        return "player_a", edge_a
    else:
        return "player_b", edge_b

def should_place_bet(edge: float) -> bool:
    """
    Determine if computed edge meets minimum threshold for placing a bet.

    Bets are only placed when edge exceeds MIN_EDGE_THRESHOLD configured value
    (default 3% / 0.03).

    Args:
        edge (float): Computed EV edge as decimal [e.g., 0.04 for +4%]

    Returns:
        bool: True if edge > MIN_EDGE_THRESHOLD, False otherwise

    Example:
        >>> should_place_bet(0.04)  # 4% edge, threshold is 3%
        True
        >>> should_place_bet(0.02)  # 2% edge, threshold is 3%
        False
    """
    return edge > MIN_EDGE_THRESHOLD

def update_trade_row(row: dict, model_prob_a: float, model_prob_b: float,
                     selected_side: str, edge: float, stake: float) -> dict:
    """
    Update a pending trade row with model predictions and betting decision.

    Fills in model_prob_a, model_prob_b, selected_side, edge, stake, and status.
    Status changes from "pending" to "placed" only if a bet was placed
    (selected_side is not empty and stake > 0).

    Args:
        row (Dict): Original trade row from trades.csv
        model_prob_a (float): Model's predicted P(player A wins) [0.0-1.0]
        model_prob_b (float): Model's predicted P(player B wins) [0.0-1.0]
        selected_side (str): "player_a", "player_b", or "" (no bet)
        edge (float): Computed edge value as decimal
        stake (float): Bet amount in USD, 0.0 if no bet placed

    Returns:
        Dict: Updated trade row with all prediction fields populated

    Example:
        >>> row = {"match_id": "evt_123", "odds_a": 1.95, "status": "pending"}
        >>> updated = update_trade_row(row, 0.55, 0.45, "player_a", 0.04, 10.0)
        >>> print(updated["status"])
        placed
        >>> print(updated["stake"])
        10.0
    """
    row.update({
        "model_prob_a": model_prob_a,
        "model_prob_b": model_prob_b,
        "selected_side": selected_side,
        "edge": edge,
        "stake": stake if selected_side else "",
        "status": TradeStatus.PLACED.value if selected_side else TradeStatus.PENDING.value,
    })
    return row

def main():
    """
    Read pending trades from trades.csv, run model predictions, and decide bets.

    Entry point for prediction script. Processes all pending trades (those without
    model probabilities), runs Random Forest predictions, computes edges, and
    places bets where edge > MIN_EDGE_THRESHOLD. Updates trades.csv with results.

    Returns:
        None (writes updated trades.csv)

    Raises:
        FileNotFoundError: If MODEL_PATH or TRADES_CSV doesn't exist
        ValueError: If CSV is corrupted or missing required columns

    Example:
        >>> python mvp/predict_and_log.py
        # Outputs:
        # INFO: Loaded model from models/random_forest_v2.pkl
        # INFO: Found 5 pending trades
        # INFO: evt_123: Placing bet on player_a, edge=4.23%
        # INFO: Updated trades.csv
    """
    logging.basicConfig(level=logging.INFO)

    model = load_model(MODEL_PATH)

    # Read all trades
    trades = []
    with open(TRADES_CSV, "r") as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    # Filter for pending trades (missing predictions)
    pending = [t for t in trades if t["model_prob_a"] == ""]

    logger.info(f"Found {len(pending)} pending trades")

    # Predict for each pending trade
    updated_rows = []
    for trade in pending:
        # TODO: Load actual features for this matchup
        # For MVP, use placeholder
        features_a = load_features(trade["player_a"])

        # Run prediction
        model_prob_a = predict(model, features_a)
        model_prob_b = 1.0 - model_prob_a

        # Compute edge and select side
        implied_prob_a = float(trade["implied_prob_a"])
        implied_prob_b = float(trade["implied_prob_b"])

        side, edge = compute_edge_and_select_side(
            model_prob_a, model_prob_b,
            implied_prob_a, implied_prob_b
        )

        # Decide to bet or skip
        stake = 0.0
        if should_place_bet(edge):
            from mvp.config import STAKE
            stake = STAKE
            logger.info(f"{trade['match_id']}: Placing bet on {side}, edge={edge:.2%}")
        else:
            logger.info(f"{trade['match_id']}: Skipping (edge={edge:.2%} < threshold)")

        # Update row
        updated = update_trade_row(trade, model_prob_a, model_prob_b,
                                   side if stake > 0 else "", edge,
                                   stake)
        updated_rows.append(updated)

    # Rewrite CSV with updated rows
    if updated_rows:
        fieldnames = [
            "match_id", "fetch_ts", "player_a", "player_b",
            "odds_a", "odds_b",
            "model_prob_a", "model_prob_b",
            "implied_prob_a", "implied_prob_b",
            "selected_side", "edge", "stake",
            "result", "pnl", "status",
        ]

        with open(TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Write all trades (updated + unchanged)
            writer.writerows(trades)

        logger.info("Updated trades.csv")

if __name__ == "__main__":
    main()
