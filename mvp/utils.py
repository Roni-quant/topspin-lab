# mvp/utils.py
import csv
from pathlib import Path
from typing import Dict, List, Optional
import logging
from .config import TRADES_CSV_FIELDNAMES

logger = logging.getLogger(__name__)

def load_csv(path: Path) -> List[Dict]:
    """
    Load trades.csv file into memory as list of dictionaries.

    Args:
        path (Path): Path to trades.csv file

    Returns:
        List[Dict]: List of trade dictionaries, empty list if file doesn't exist.
                   Each dict has keys: match_id, fetch_ts, player_a, player_b,
                   odds_a, odds_b, model_prob_a, model_prob_b, implied_prob_a,
                   implied_prob_b, selected_side, edge, stake, result, pnl, status

    Example:
        >>> from pathlib import Path
        >>> trades = load_csv(Path("mvp/data/trades.csv"))
        >>> print(f"Loaded {len(trades)} trades")
        >>> for trade in trades:
        ...     print(trade["match_id"], trade["status"])
    """
    if not path.exists():
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

def append_to_csv(path: Path, row: Dict) -> None:
    """
    Append a single trade row to trades.csv, creating file if necessary.

    Args:
        path (Path): Path to trades.csv file
        row (Dict): Trade dictionary with required keys:
                   match_id, fetch_ts, player_a, player_b, odds_a, odds_b,
                   model_prob_a, model_prob_b, implied_prob_a, implied_prob_b,
                   selected_side, edge, stake, result, pnl, status

    Returns:
        None

    Raises:
        IOError: If file cannot be written to

    Example:
        >>> row = {
        ...     "match_id": "evt_123",
        ...     "fetch_ts": "2026-04-28T14:30:00Z",
        ...     "player_a": "Fan Zhendong",
        ...     "player_b": "Tomokazu Harimoto",
        ...     "odds_a": 1.95,
        ...     "odds_b": 1.85,
        ...     "status": "pending",
        ... }
        >>> append_to_csv(Path("mvp/data/trades.csv"), row)
    """
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Appended to {path}: {row['match_id']}")

def map_player_name(ittf_name: str, mapping: Dict) -> Optional[str]:
    """
    Map ITTF roster name to internal player ID using provided mapping.

    Args:
        ittf_name (str): Player name as listed in ITTF roster
        mapping (Dict): Dictionary mapping ITTF names to player IDs

    Returns:
        Optional[str]: Internal player ID if found, None otherwise

    Example:
        >>> mapping = {"Fan Zhendong": "fan_z_chn", "Harimoto Tomokazu": "har_t_jpn"}
        >>> player_id = map_player_name("Fan Zhendong", mapping)
        >>> print(player_id)
        fan_z_chn
    """
    return mapping.get(ittf_name)

def compute_implied_probability(odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Args:
        odds (float): Decimal odds (e.g., 1.95, 2.0)

    Returns:
        float: Implied probability [0.0-1.0], calculated as 1.0 / odds

    Example:
        >>> prob = compute_implied_probability(2.0)
        >>> print(prob)
        0.5
        >>> prob = compute_implied_probability(1.95)
        >>> print(round(prob, 4))
        0.5128
    """
    return 1.0 / odds

def compute_edge(model_prob: float, implied_prob: float) -> float:
    """
    Compute expected value edge between model prediction and bookmaker odds.

    Edge = model_prob - implied_prob. Positive edge means model is more optimistic
    than the market, suggesting a profitable opportunity.

    Args:
        model_prob (float): Model's predicted probability [0.0-1.0]
        implied_prob (float): Bookmaker implied probability [0.0-1.0]

    Returns:
        float: Expected value edge, negative if model is less bullish than market

    Example:
        >>> edge = compute_edge(0.55, 0.51)  # Model says 55%, market says 51%
        >>> print(edge)
        0.04  # 4% edge
        >>> if edge > 0.03:
        ...     print("Place bet")
    """
    return model_prob - implied_prob

def read_trades_csv(path: Path) -> List[Dict]:
    """
    Read all trades from trades.csv into memory.

    Convenience wrapper around load_csv() for explicit naming.

    Args:
        path (Path): Path to trades.csv

    Returns:
        List[Dict]: List of trade dictionaries, empty list if file doesn't exist
    """
    return load_csv(path)

def write_trades_csv(path: Path, rows: List[Dict]) -> None:
    """
    Write multiple trade rows to trades.csv, overwriting existing file.

    Args:
        path (Path): Path to trades.csv
        rows (List[Dict]): List of trade dictionaries to write

    Raises:
        IOError: If file cannot be written to
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} trades to {path}")

def safe_float(value, default: float = None) -> Optional[float]:
    """
    Safely convert value to float, returning default if conversion fails.

    Args:
        value: Value to convert
        default (float): Default value if conversion fails (default: None)

    Returns:
        Optional[float]: Converted float or default value

    Example:
        >>> safe_float("1.95")
        1.95
        >>> safe_float("invalid", default=0.0)
        0.0
        >>> safe_float(None)
        None
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default
