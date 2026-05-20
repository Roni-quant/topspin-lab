import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MERGED_OUTPUT = PROJECT_ROOT / "data" / "raw_matches.parquet"
LOG_DIR = PROJECT_ROOT / "logs"

# ITTF API
ITTF_BASE_URL = "https://results.ittf.link/"
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Rate limiting (seconds between requests)
RATE_LIMIT_EVENTS = 0.5
RATE_LIMIT_PLAYERS = 0.1

# Concurrent workers for fetch stages
CONCURRENCY_PLAYERS = 5
CONCURRENCY_MATCHES = 3

# Scraping scope
START_YEAR = 2004
CURRENT_YEAR_REFRESH_MONTHS = 3  # re-fetch events from last N months for current year

# Merge thresholds
DUPLICATE_RATIO_THRESHOLD = 0.01  # fail merge if duplicates exceed 1%


def get_headers() -> dict:
    """Build request headers with ITTF session cookie."""
    cookie = os.getenv("ITTF_COOKIE_HEADER")
    if not cookie:
        # Try auto-login if credentials are available
        from pipeline.auth import refresh_cookie

        cookie = refresh_cookie()
    if not cookie:
        raise RuntimeError(
            "ITTF_COOKIE_HEADER not set and auto-login unavailable. "
            "Set ITTF_USERNAME + ITTF_PASSWORD in .env for auto-login, "
            "or copy your session cookie as ITTF_COOKIE_HEADER. "
            "See .env.example for instructions."
        )
    return {"Cookie": cookie}


def ensure_dirs() -> None:
    """Create required output directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
