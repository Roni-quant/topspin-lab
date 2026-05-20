# mvp/config.py
import os
from pathlib import Path
from enum import Enum
from .london_2026_roster import PLAYER_MAPPING

# File paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TRADES_CSV = DATA_DIR / "trades.csv"

# CSV Schema
TRADES_CSV_FIELDNAMES = [
    "match_id", "fetch_ts", "player_a", "player_b",
    "odds_a", "odds_b",
    "model_prob_a", "model_prob_b",
    "implied_prob_a", "implied_prob_b",
    "selected_side", "edge", "stake",
    "result", "pnl", "status", "event",
]

# Trade Status Enum
class TradeStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    SETTLED = "settled"

# Bet Result Enum
class BetResult(Enum):
    WIN = "win"
    LOSS = "loss"
    VOID = "void"

# API Configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "your_key_here")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "table-tennis"  # or "tennis" if table-tennis not available
ODDS_LEAGUE = "london-2026"
ODDS_API_REGION = "us"
ODDS_API_MARKET = "h2h"

# Betting Rules
MIN_EDGE_THRESHOLD = 0.03  # 3% minimum edge
STAKE = 10.00  # $10 per bet
STARTING_CAPITAL = 1000.00
DEFAULT_EVENT = "unknown"

# Features — single source of truth is the training pipeline
from pipeline.train_models_v2 import FEATURE_COLS_ENHANCED as FEATURE_NAMES

# Parent project data directory (Parquet files)
PARENT_DATA_DIR = PROJECT_ROOT.parent / "data"

# Model
MODEL_PATH = PROJECT_ROOT.parent / "models" / "random_forest_v2.pkl"

# Player Mapping (London 2026 roster → your player IDs)
# Imported from london_2026_roster.py (Task 2)
# Contains ~160 validated entries mapping ITTF official names to player_ids
# NOTE: Full 500-player roster requires complete ITTF tournament registration data
__all__ = [
    "PLAYER_MAPPING", "ODDS_API_KEY", "FEATURE_NAMES", "STAKE", "MIN_EDGE_THRESHOLD",
    "TRADES_CSV_FIELDNAMES", "TradeStatus", "BetResult", "DEFAULT_EVENT",
]
