"""Stage 3 — Player Info Scraper.

Collects unique player IDs from all matches_{year}.parquet files,
fetches each player's profile from the ITTF API, and writes
data/raw/players.parquet.

Usage:
    python -m pipeline.fetch_players
"""

import logging
import re

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.config import (
    CONCURRENCY_PLAYERS,
    DATA_DIR,
    ITTF_BASE_URL,
    RATE_LIMIT_PLAYERS,
    get_headers,
)
from pipeline.http import RateLimiter, fetch_json
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import PLAYER_DTYPES, validate_schema

# --- Paths ---

PLAYERS_OUTPUT = DATA_DIR / "players.parquet"

# Module-level logger — initialised in main() via setup_stage_logger.
logger: logging.Logger = logging.getLogger("fetch_players")


# --- Collect player IDs from match files ---


def collect_player_ids() -> set[int]:
    """Scan all matches_{year}.parquet files and return unique player IDs."""
    player_ids: set[int] = set()
    match_files = sorted(DATA_DIR.glob("matches_*.parquet"))

    if not match_files:
        logger.warning("No matches_*.parquet files found in data/raw/")
        return player_ids

    for path in match_files:
        logger.info("Scanning %s for player IDs", path.name)
        df = pd.read_parquet(path, columns=["player_a_id", "player_b_id"])
        player_ids.update(df["player_a_id"].dropna().astype(int))
        player_ids.update(df["player_b_id"].dropna().astype(int))

    logger.info(
        "Found %d unique player IDs across %d files",
        len(player_ids),
        len(match_files),
    )
    return player_ids


# --- Load existing players for resumability ---


def load_existing_players() -> pd.DataFrame:
    """Load existing players.parquet if it exists, else return empty DataFrame."""
    if PLAYERS_OUTPUT.exists():
        df = pd.read_parquet(PLAYERS_OUTPUT)
        logger.info("Loaded %d existing players from %s", len(df), PLAYERS_OUTPUT.name)
        return df
    return pd.DataFrame(columns=list(PLAYER_DTYPES.keys()))


# --- Fetch a single player profile ---


def _build_player_url(player_id: int) -> str:
    """Build the API URL for a player profile lookup."""
    return (
        f"{ITTF_BASE_URL}"
        f"index.php?option=com_fabrik&view=list&listid=33"
        f"&resetfilters=1&format=json&limit33=100&limitstart33=0"
        f"&vw_profiles___player_id_raw[value]={player_id}"
    )


def _extract_notranslate(html: str, label: str) -> str | None:
    """Extract a notranslate span value following a label in profile HTML."""
    pattern = rf"{label}:\s*<span class='notranslate'>([^<]+)</span>"
    m = re.search(pattern, html, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _parse_player(player_id: int, data) -> dict | None:
    """Extract player fields from the API JSON response.

    Returns a dict with player fields, or None if the response is malformed
    or contains no data for the player.
    """
    try:
        # Handle [[...records...]] format from Fabrik API
        if isinstance(data, list):
            rows = data[0] if data and isinstance(data[0], list) else data
        elif isinstance(data, dict):
            rows = data.get("data", [])
        else:
            return None

        if not rows:
            return None

        row = rows[0]

        # player_name from name_raw field (format: "WANG Hao (#109961)")
        name_raw = row.get("vw_profiles___name_raw", "")
        if isinstance(name_raw, str):
            # Strip the (#ID) suffix
            player_name = re.sub(r"\s*\(#\d+\)\s*$", "", name_raw).strip()
        else:
            player_name = ""
        if not player_name:
            return None

        # Parse the profile HTML blob for structured fields
        profile_html = row.get("vw_profiles___profile_raw", "") or ""

        # birth_year
        birth_year = None
        by_str = _extract_notranslate(profile_html, "Birth Year")
        if by_str:
            try:
                birth_year = int(by_str)
            except (ValueError, TypeError):
                pass

        # association from assoc_raw
        association_raw = row.get("vw_profiles___assoc_raw", None)
        association = None
        if isinstance(association_raw, str) and association_raw.strip():
            association = association_raw.strip()

        # hand — from profile HTML "Right-Hand" or "Left-Hand"
        hand = None
        hand_match = re.search(r"Style:\s*<span class='notranslate'>(Right|Left)-Hand</span>", profile_html)
        if hand_match:
            hand = hand_match.group(1).lower()

        # grip — from profile HTML "(Penhold)" or "(Shakehand)"
        grip = None
        grip_match = re.search(r"\((?:<span class='notranslate'>)?(Penhold|Shakehand)(?:</span>)?\)", profile_html, re.IGNORECASE)
        if grip_match:
            grip = grip_match.group(1).lower()

        # style — from profile HTML "Attack", "Defense", "Allround"
        style = None
        style_match = re.search(r"<span class='notranslate'>(Attack|Defense|Allround)</span>", profile_html)
        if style_match:
            style = style_match.group(1).lower()

        return {
            "player_id": player_id,
            "player_name": player_name,
            "birth_year": birth_year,
            "association": association,
            "hand": hand,
            "grip": grip,
            "style": style,
        }

    except Exception as exc:
        logger.warning("Malformed response for player %d: %s", player_id, exc)
        return None


def fetch_player(player_id: int, headers: dict) -> dict | None:
    """Fetch a single player profile using the shared HTTP client.

    Returns a parsed player dict, or None on failure.
    """
    url = _build_player_url(player_id)
    data = fetch_json(url, headers)
    if data is not None:
        return _parse_player(player_id, data)
    return None


# --- Concurrent fetch helpers ---


def _fetch_one(player_id: int, headers: dict, limiter: RateLimiter) -> tuple[int, dict | None]:
    """Fetch a single player profile with rate limiting. Runs in a worker thread."""
    limiter.wait()
    return player_id, fetch_player(player_id, headers)


def _fetch_players_concurrent(
    ids_to_fetch: list[int],
    headers: dict,
    limiter: RateLimiter,
    existing_df: pd.DataFrame,
    output_path,
) -> tuple[list[dict], int]:
    """Fetch player profiles concurrently using a thread pool.

    Returns (new_records, failed_count).
    """
    new_records: list[dict] = []
    failed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY_PLAYERS) as pool:
            futures = {
                pool.submit(_fetch_one, pid, headers, limiter): pid
                for pid in ids_to_fetch
            }
            try:
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        player_id, record = future.result()
                    except Exception as exc:
                        logger.warning("Worker failed: %s", exc)
                        failed_count += 1
                        if i % 100 == 0 or i == len(ids_to_fetch):
                            logger.info("Progress: %d/%d players fetched", i, len(ids_to_fetch))
                        continue
                    if record is not None:
                        new_records.append(record)
                    else:
                        failed_count += 1

                    if i % 100 == 0 or i == len(ids_to_fetch):
                        logger.info("Progress: %d/%d players fetched", i, len(ids_to_fetch))

                    # Checkpoint save every 500
                    if i % 500 == 0 and new_records:
                        new_df = validate_schema(pd.DataFrame(new_records), PLAYER_DTYPES)
                        combined = pd.concat([existing_df, new_df], ignore_index=True)
                        combined.to_parquet(output_path, index=False)
                        logger.info("Checkpoint: saved %d players", len(combined))

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt — shutting down and saving progress")
                pool.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        pass  # already handled above

    return new_records, failed_count


# --- Certification report ---


def print_certification(
    total_requested: int,
    successfully_fetched: int,
    failed_count: int,
    df: pd.DataFrame,
) -> None:
    """Print and log the Stage 3 certification report."""
    nullable_fields = ["birth_year", "association", "hand", "grip", "style"]

    lines = [
        "",
        "=" * 60,
        "Stage 3 Certification — Player Info Scraper",
        "=" * 60,
        f"  Unique players requested : {total_requested}",
        f"  Successfully fetched     : {successfully_fetched}",
        f"  Missing/failed profiles  : {failed_count}",
        "",
        "  Null rates by field:",
    ]

    null_rates = {}
    for field in nullable_fields:
        if field in df.columns and len(df) > 0:
            null_count = df[field].isna().sum()
            rate = null_count / len(df)
        else:
            rate = 1.0
        null_rates[field] = rate
        lines.append(f"    {field:<20s}: {rate:.1%}")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    log_structured(
        logger,
        logging.INFO,
        "Stage 3 certification",
        total_requested=total_requested,
        successfully_fetched=successfully_fetched,
        failed_count=failed_count,
        null_rates=null_rates,
    )


# --- Main ---


def main() -> None:
    global logger
    logger = setup_stage_logger("fetch_players")
    logger.info("Stage 3 — Player Info Scraper starting")

    # Collect all unique player IDs from match files
    all_player_ids = collect_player_ids()
    if not all_player_ids:
        logger.warning("No player IDs found. Nothing to fetch.")
        print_certification(0, 0, 0, pd.DataFrame(columns=list(PLAYER_DTYPES.keys())))
        return

    # Load existing players for resumability
    existing_df = load_existing_players()
    existing_ids = set()
    if len(existing_df) > 0 and "player_id" in existing_df.columns:
        existing_ids = set(existing_df["player_id"].dropna().astype(int).tolist())

    ids_to_fetch = sorted(all_player_ids - existing_ids)
    logger.info(
        "%d unique players, %d already fetched, %d to fetch",
        len(all_player_ids),
        len(existing_ids),
        len(ids_to_fetch),
    )

    # Fetch player profiles concurrently
    headers = get_headers()
    limiter = RateLimiter(RATE_LIMIT_PLAYERS)
    new_records, failed_count = _fetch_players_concurrent(
        ids_to_fetch, headers, limiter, existing_df, PLAYERS_OUTPUT,
    )

    logger.info("Fetched %d new profiles, %d failed", len(new_records), failed_count)

    # Combine existing + new
    if new_records:
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = existing_df

    # Validate schema and write output
    if len(combined_df) > 0:
        combined_df = validate_schema(combined_df, PLAYER_DTYPES)
        combined_df = combined_df.sort_values("player_id").reset_index(drop=True)
        combined_df.to_parquet(PLAYERS_OUTPUT, index=False)
        logger.info("Wrote %d players to %s", len(combined_df), PLAYERS_OUTPUT.name)
    else:
        logger.warning("No player data to write")

    # Certification
    total_requested = len(ids_to_fetch)
    successfully_fetched = len(new_records)
    print_certification(total_requested, successfully_fetched, failed_count, combined_df)


if __name__ == "__main__":
    main()
