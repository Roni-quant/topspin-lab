"""Stage 2 — Match Scraper.

Reads events_index.parquet, fetches all matches for each event from the
ITTF results API, parses results, and writes yearly parquet files to
data/raw/matches_{year}.parquet.

Usage:
    python -m pipeline.fetch_matches
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.config import (
    CONCURRENCY_MATCHES,
    CURRENT_YEAR_REFRESH_MONTHS,
    DATA_DIR,
    ITTF_BASE_URL,
    MERGED_OUTPUT,
    RATE_LIMIT_EVENTS,
    get_headers,
)
from pipeline.http import RateLimiter, fetch_json
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import (
    ACCEPTED_RESULT_REGEX,
    GAME_SCORE_REGEX,
    MATCH_DTYPES,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_SIZE = 100
_EVENTS_INDEX_PATH = DATA_DIR / "events_index.parquet"

_WALKOVER_RE = re.compile(r"W/?O", re.IGNORECASE)
_RETIREMENT_RE = re.compile(r"RET", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Match-key computation
# ---------------------------------------------------------------------------


def _compute_match_key(
    event_id: int,
    match_date: str,
    player_a_id: int,
    player_b_id: int,
    result: str,
    games: str,
) -> str:
    """Deterministic SHA-256 hash for deduplication."""
    payload = f"{event_id}|{match_date}|{player_a_id}|{player_b_id}|{result}|{games}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse_result(
    result_raw: str | None,
    games_raw: str | None,
    player_a_id: int | None,
    player_b_id: int | None,
) -> tuple[int | None, str | None]:
    """Return (winner_id, drop_reason) for a single match.

    winner_id is set only when drop_reason is None.
    """
    # Missing player IDs
    if player_a_id is None or player_b_id is None:
        return None, "missing_player_id"
    if pd.isna(player_a_id) or pd.isna(player_b_id):
        return None, "missing_player_id"

    # Missing result
    if result_raw is None or (isinstance(result_raw, str) and result_raw.strip() == ""):
        return None, "missing_result"
    if pd.isna(result_raw):
        return None, "missing_result"

    result_str = str(result_raw).strip()

    # Walkover
    if _WALKOVER_RE.search(result_str):
        return None, "walkover"

    # Retirement
    if _RETIREMENT_RE.search(result_str):
        return None, "retirement"

    # Non-standard result format
    if not ACCEPTED_RESULT_REGEX.match(result_str):
        return None, "non_standard_result"

    # Parse scores
    parts = result_str.split(":")
    try:
        score_a = int(parts[0].strip())
        score_b = int(parts[1].strip())
    except (ValueError, IndexError):
        return None, "non_standard_result"

    # Determine winner
    if score_a > score_b:
        winner = int(player_a_id)
    elif score_b > score_a:
        winner = int(player_b_id)
    else:
        # Tied result — shouldn't happen in table tennis, treat as non-standard
        return None, "non_standard_result"

    # Optional: validate games string if present
    if games_raw is not None and not pd.isna(games_raw):
        games_str = str(games_raw).strip()
        if games_str:
            # Each individual game should look like "N:M"
            individual_games = games_str.split()
            for g in individual_games:
                if not GAME_SCORE_REGEX.match(g.strip()):
                    return None, "malformed_games"

    return winner, None


# ---------------------------------------------------------------------------
# Fetching matches for a single event
# ---------------------------------------------------------------------------


def _safe_int(value) -> int | None:
    """Convert a value to int, returning None on failure."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value) -> str | None:
    """Convert a value to str, returning None for empty/NaN."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def fetch_event_matches(
    event_id: int,
    tournament_id: int,
    event_name: str,
    event_date: date,
    event_category: str,
    headers: dict,
    logger: logging.Logger,
    limiter=None,
) -> list[dict]:
    """Fetch all match records for a single event via paginated API calls.

    Returns a list of dicts, one per match (including dropped matches).
    """
    matches: list[dict] = []
    offset = 0
    match_date_str = event_date.isoformat()

    while True:
        if limiter is not None:
            limiter.wait()

        url = (
            f"{ITTF_BASE_URL}index.php?"
            f"option=com_fabrik&view=list&listid=31&Itemid=250"
            f"&resetfilters=1&format=json"
            f"&vw_matches___tournament_id_raw[value][]={tournament_id}"
            f"&limit31={_PAGE_SIZE}&limitstart31={offset}"
        )

        data = fetch_json(url, headers)
        if data is None:
            logger.warning(
                "Failed to fetch matches for event %d (%s) at offset %d — skipping remainder",
                event_id, event_name, offset,
            )
            break

        # The ITTF Fabrik API returns [[...records...]] (list of lists)
        if isinstance(data, list):
            rows = data[0] if data and isinstance(data[0], list) else data
        elif isinstance(data, dict):
            rows = data.get("data", data.get("rows", []))
            if isinstance(rows, dict):
                rows = list(rows.values())
        else:
            rows = []

        if not isinstance(rows, list) or len(rows) == 0:
            break

        for row in rows:
            # Filter by category (MS/WS) at match level
            match_category = _safe_str(row.get("vw_matches___event_raw"))
            if match_category not in ("MS", "WS"):
                continue

            # Extract raw fields — API uses player_a / player_x naming
            source_match_id = _safe_int(
                row.get("vw_matches___id_raw")
                or row.get("__pk_val")
            )
            player_a_id = _safe_int(row.get("vw_matches___player_a_id_raw"))
            player_a_name = _safe_str(row.get("vw_matches___name_a_raw"))
            # API uses "player_x" which we rename to "player_b"
            player_b_id = _safe_int(row.get("vw_matches___player_x_id_raw"))
            player_b_name = _safe_str(row.get("vw_matches___name_x_raw"))
            result_raw = _safe_str(row.get("vw_matches___res_raw"))
            games_raw = _safe_str(row.get("vw_matches___games_raw"))

            # Normalize result format: "0 - 3" → "0:3"
            if result_raw:
                result_raw = re.sub(r"\s*-\s*", ":", result_raw)

            # Derive winner and drop_reason
            winner_id, drop_reason = _parse_result(
                result_raw, games_raw, player_a_id, player_b_id
            )

            # Compute deterministic match key
            match_key = _compute_match_key(
                event_id,
                match_date_str,
                player_a_id or 0,
                player_b_id or 0,
                result_raw or "",
                games_raw or "",
            )

            matches.append(
                {
                    "match_key": match_key,
                    "source_match_id": source_match_id,
                    "match_date": event_date,
                    "event_id": event_id,
                    "event_name": event_name,
                    "player_a_id": player_a_id,
                    "player_a_name": player_a_name or "",
                    "player_b_id": player_b_id,
                    "player_b_name": player_b_name or "",
                    "winner_id": winner_id,
                    "result": result_raw or "",
                    "games": games_raw or "",
                    "category": match_category,
                    "drop_reason": drop_reason,
                }
            )

        total = data.get("total") if isinstance(data, dict) else None
        fetched_so_far = offset + len(rows)

        # Stop when we've received fewer than a full page or reached total
        if len(rows) < _PAGE_SIZE:
            break
        if total is not None:
            try:
                if fetched_so_far >= int(total):
                    break
            except (ValueError, TypeError):
                pass

        offset += _PAGE_SIZE

    return matches


# ---------------------------------------------------------------------------
# Resumability helpers
# ---------------------------------------------------------------------------


_MERGED_CACHE: pd.DataFrame | None = None


def _load_merged_subset(year: int, logger: logging.Logger) -> pd.DataFrame:
    """Load the year-subset of the merged raw_matches.parquet if present.

    Used as a resumability fallback when yearly per-year files were never
    written but the merged artifact exists.
    """
    global _MERGED_CACHE
    if _MERGED_CACHE is None:
        if not MERGED_OUTPUT.exists():
            _MERGED_CACHE = pd.DataFrame(columns=list(MATCH_DTYPES.keys()))
        else:
            try:
                df = pd.read_parquet(MERGED_OUTPUT)
                df["match_date"] = pd.to_datetime(df["match_date"])
                _MERGED_CACHE = df
                logger.info(
                    "Resumability: loaded %d rows from merged %s",
                    len(df), MERGED_OUTPUT.name,
                )
            except Exception as exc:
                logger.warning("Could not read %s: %s — treating as empty", MERGED_OUTPUT, exc)
                _MERGED_CACHE = pd.DataFrame(columns=list(MATCH_DTYPES.keys()))

    if _MERGED_CACHE.empty:
        return _MERGED_CACHE
    return _MERGED_CACHE[_MERGED_CACHE["match_date"].dt.year == year]


def _load_existing_matches(year: int, logger: logging.Logger) -> pd.DataFrame:
    """Load existing matches for a given year.

    Tries the per-year parquet first; falls back to the merged raw_matches.parquet
    (year-subset) so resumability also works when only the merged artifact is
    present (e.g., after Stage 4 merge but yearly files were cleaned up).
    """
    path = DATA_DIR / f"matches_{year}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read %s: %s — treating as empty", path, exc)

    merged_subset = _load_merged_subset(year, logger)
    if not merged_subset.empty:
        return merged_subset
    return pd.DataFrame(columns=list(MATCH_DTYPES.keys()))


def _events_needing_refresh(events_df: pd.DataFrame) -> set[int]:
    """Return event IDs from the current year that fall within the refresh window."""
    today = date.today()
    if events_df.empty:
        return set()

    cutoff = today - timedelta(days=CURRENT_YEAR_REFRESH_MONTHS * 30)
    current_year = today.year

    mask = (
        (events_df["event_date"].dt.year == current_year)
        & (events_df["event_date"].dt.date >= cutoff)
    )
    return set(events_df.loc[mask, "event_id"].dropna().unique())


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _write_yearly_parquet(
    year: int,
    new_matches: list[dict],
    existing_df: pd.DataFrame,
    logger: logging.Logger,
) -> int:
    """Merge new matches into the yearly parquet, dedup by match_key.

    Returns the number of duplicate match_keys that were replaced.
    """
    new_df = pd.DataFrame(new_matches)
    if new_df.empty:
        return 0

    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    # Dedup: keep last occurrence (i.e., the freshly fetched version)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="match_key", keep="last")
    dup_count = before_dedup - len(combined)

    # Validate and cast schema
    combined = validate_schema(combined, MATCH_DTYPES)

    path = DATA_DIR / f"matches_{year}.parquet"
    combined.to_parquet(path, index=False)
    logger.info("Wrote %d matches to %s (%d duplicates removed)", len(combined), path, dup_count)

    return dup_count


# ---------------------------------------------------------------------------
# Concurrent fetch helpers
# ---------------------------------------------------------------------------


def _fetch_one_event(
    event_row: dict,
    headers: dict,
    limiter: RateLimiter,
    logger: logging.Logger,
) -> tuple[int, int, list[dict]]:
    """Fetch matches for a single event. Runs in a worker thread."""
    eid = int(event_row["event_id"])
    tid = int(event_row.get("tournament_id", eid))
    ename = str(event_row["event_name"])
    edate = event_row["event_date"]
    if hasattr(edate, "date"):
        edate = edate.date()
    ecat = str(event_row.get("event_category", ""))
    matches = fetch_event_matches(eid, tid, ename, edate, ecat, headers, logger, limiter)
    return eid, edate.year, matches


def _fetch_events_concurrent(
    event_rows: list[dict],
    headers: dict,
    limiter: RateLimiter,
    logger: logging.Logger,
) -> tuple[dict[int, list[dict]], int]:
    """Fetch matches for multiple events concurrently.

    Returns (matches_by_year dict, events_processed count).
    """
    matches_by_year: dict[int, list[dict]] = defaultdict(list)
    events_processed = 0

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY_MATCHES) as pool:
            futures = {
                pool.submit(_fetch_one_event, row, headers, limiter, logger): row
                for row in event_rows
            }
            try:
                for future in as_completed(futures):
                    try:
                        eid, year, event_matches = future.result()
                    except Exception as exc:
                        logger.warning("Event worker failed: %s", exc)
                        events_processed += 1
                        continue
                    if event_matches:
                        matches_by_year[year].extend(event_matches)
                        logger.info(
                            "  -> event %d: %d matches (%d dropped)",
                            eid, len(event_matches),
                            sum(1 for m in event_matches if m.get("drop_reason") is not None),
                        )
                    else:
                        logger.info("  -> event %d: 0 matches returned", eid)
                    events_processed += 1
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt — shutting down and saving progress")
                pool.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        pass

    return dict(matches_by_year), events_processed


# ---------------------------------------------------------------------------
# Certification report
# ---------------------------------------------------------------------------


def _print_certification(
    matches_by_year: dict[int, list[dict]],
    total_duplicates: int,
    logger: logging.Logger,
) -> None:
    """Print and log the Stage 2 certification summary."""
    all_matches = [m for ms in matches_by_year.values() for m in ms]

    if not all_matches:
        msg = "Stage 2 certification: No matches fetched."
        logger.info(msg)
        print(msg)
        return

    df = pd.DataFrame(all_matches)

    # Matches fetched by year
    df["_year"] = pd.to_datetime(df["match_date"]).dt.year
    by_year = df.groupby("_year").size().to_dict()

    # Dropped matches by reason code
    dropped = df[df["drop_reason"].notna()]
    by_reason = dropped.groupby("drop_reason").size().to_dict() if not dropped.empty else {}

    # Duplicate match_key count
    dup_keys = df["match_key"].duplicated().sum()

    # Percent missing player IDs
    missing_pid = df[
        df["player_a_id"].isna() | df["player_b_id"].isna()
    ].shape[0]
    pct_missing_pid = (missing_pid / len(df) * 100) if len(df) > 0 else 0.0

    # Percent non-standard results
    non_standard = df[df["drop_reason"] == "non_standard_result"].shape[0]
    pct_non_standard = (non_standard / len(df) * 100) if len(df) > 0 else 0.0

    # Category distribution
    cat_dist = df.groupby("category").size().to_dict()

    # Min/max match date
    dates = pd.to_datetime(df["match_date"])
    min_date = dates.min()
    max_date = dates.max()

    report_lines = [
        "",
        "=" * 60,
        "  STAGE 2 CERTIFICATION — Match Scraper",
        "=" * 60,
        "",
        "  Matches fetched by year:",
    ]
    for y in sorted(by_year):
        report_lines.append(f"    {y}: {by_year[y]}")

    report_lines.append("")
    report_lines.append("  Dropped matches by reason code:")
    if by_reason:
        for reason in sorted(by_reason):
            report_lines.append(f"    {reason}: {by_reason[reason]}")
    else:
        report_lines.append("    (none)")

    report_lines += [
        "",
        f"  Duplicate match_key count:     {dup_keys + total_duplicates}",
        f"  Percent missing player IDs:    {pct_missing_pid:.2f}%",
        f"  Percent non-standard results:  {pct_non_standard:.2f}%",
        "",
        "  Category distribution:",
    ]
    for cat in sorted(cat_dist):
        report_lines.append(f"    {cat}: {cat_dist[cat]}")

    report_lines += [
        "",
        f"  Min match date:  {min_date.date() if pd.notna(min_date) else 'N/A'}",
        f"  Max match date:  {max_date.date() if pd.notna(max_date) else 'N/A'}",
        "",
        "=" * 60,
    ]

    report = "\n".join(report_lines)
    print(report)
    log_structured(
        logger,
        logging.INFO,
        "Stage 2 certification",
        stage="fetch_matches",
        entity_type="certification",
        status="complete",
        reason=None,
        matches_by_year=by_year,
        dropped_by_reason=by_reason,
        duplicate_match_keys=dup_keys + total_duplicates,
        pct_missing_player_ids=round(pct_missing_pid, 2),
        pct_non_standard_results=round(pct_non_standard, 2),
        category_distribution=cat_dist,
        min_match_date=str(min_date.date()) if pd.notna(min_date) else None,
        max_match_date=str(max_date.date()) if pd.notna(max_date) else None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Stage 2: fetch matches for all indexed events."""
    logger = setup_stage_logger("fetch_matches")

    # Load events index
    if not _EVENTS_INDEX_PATH.exists():
        logger.error(
            "events_index.parquet not found at %s. Run Stage 1 first.", _EVENTS_INDEX_PATH
        )
        raise SystemExit(1)

    events_df = pd.read_parquet(_EVENTS_INDEX_PATH)
    logger.info("Loaded %d events from events_index.parquet", len(events_df))

    if events_df.empty:
        logger.info("No events to process. Exiting.")
        return

    # Ensure event_date is datetime
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])

    headers = get_headers()

    # Determine which events need a current-year refresh
    refresh_ids = _events_needing_refresh(events_df)
    if refresh_ids:
        logger.info(
            "Current-year refresh: %d events from last %d months will be re-fetched",
            len(refresh_ids), CURRENT_YEAR_REFRESH_MONTHS,
        )

    # Group events by year for resumability checks
    events_df["_year"] = events_df["event_date"].dt.year

    total_duplicates = 0
    events_processed = 0
    events_skipped = 0

    # Collect new matches grouped by year for efficient writing
    matches_by_year: dict[int, list[dict]] = defaultdict(list)

    # Cache existing DataFrames per year (loaded once, reused for write)
    existing_dfs: dict[int, pd.DataFrame] = {}

    limiter = RateLimiter(RATE_LIMIT_EVENTS)

    for year, year_events in events_df.groupby("_year"):
        year = int(year)

        # Load existing matches once per year
        existing_df = _load_existing_matches(year, logger)
        existing_dfs[year] = existing_df

        # Compute already-scraped set BEFORE submitting futures
        if existing_df.empty or "event_id" not in existing_df.columns:
            already_scraped: set[int] = set()
        else:
            already_scraped = set(existing_df["event_id"].dropna().unique())

        # Filter to events that need fetching
        rows_to_fetch = []
        for _, event_row in year_events.iterrows():
            eid = int(event_row["event_id"])
            if eid in already_scraped and eid not in refresh_ids:
                events_skipped += 1
                continue
            rows_to_fetch.append(event_row.to_dict())

        if not rows_to_fetch:
            continue

        logger.info("Fetching %d events for year %d", len(rows_to_fetch), year)

        year_matches, year_processed = _fetch_events_concurrent(
            rows_to_fetch, headers, limiter, logger,
        )
        events_processed += year_processed

        # Merge into matches_by_year
        for yr, ms in year_matches.items():
            matches_by_year[yr].extend(ms)

        # Checkpoint: write yearly parquet after each year completes
        if year in matches_by_year and matches_by_year[year]:
            dups = _write_yearly_parquet(
                year, matches_by_year[year],
                existing_dfs.get(year, pd.DataFrame(columns=list(MATCH_DTYPES.keys()))),
                logger,
            )
            total_duplicates += dups

    logger.info(
        "Stage 2 complete: %d events processed, %d skipped (already scraped)",
        events_processed, events_skipped,
    )

    # Certification report
    _print_certification(matches_by_year, total_duplicates, logger)


if __name__ == "__main__":
    main()
