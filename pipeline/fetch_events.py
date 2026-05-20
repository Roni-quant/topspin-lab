"""Stage 1 — Event Index Scraper.

Paginates through the ITTF events endpoint, collects tournament metadata,
filters to MS/WS categories, and writes data/raw/events_index.parquet.
"""

import logging
import re
import time

import pandas as pd

from pipeline.config import (
    DATA_DIR,
    ITTF_BASE_URL,
    RATE_LIMIT_EVENTS,
    ensure_dirs,
    get_headers,
)
from pipeline.http import fetch_json
from pipeline.log import log_structured, setup_stage_logger
from pipeline.schema import EVENT_DTYPES, validate_schema

# ---------------------------------------------------------------------------
# Category parsing
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS = [
    (re.compile(r"\bMS\b|Men'?s\s+Singles", re.IGNORECASE), "MS"),
    (re.compile(r"\bWS\b|Women'?s\s+Singles", re.IGNORECASE), "WS"),
]


def _parse_category(event_name: str) -> str | None:
    """Return 'MS' or 'WS' if detectable in *event_name*, else None."""
    for pattern, cat in _CATEGORY_PATTERNS:
        if pattern.search(event_name):
            return cat
    return None


# ---------------------------------------------------------------------------
# Event extraction from a single API record
# ---------------------------------------------------------------------------


def _extract_event(logger: logging.Logger, record: dict) -> dict | None:
    """Parse a single API record into an event dict, or return None."""
    try:
        event_id = int(record.get("__pk_val") or record.get("vw_tournaments___id_raw"))
    except (TypeError, ValueError, KeyError):
        log_structured(
            logger,
            logging.WARNING,
            f"Malformed record — cannot extract event_id: {record!r:.200}",
            status="skipped",
            reason="malformed_record",
            entity_type="event",
        )
        return None

    # tournament_id is the foreign key used by match/player endpoints
    try:
        tournament_id = int(record.get("vw_tournaments___tournament_id_raw") or event_id)
    except (TypeError, ValueError):
        tournament_id = event_id

    event_name = (
        record.get("vw_tournaments___tournament_raw", "")
        or record.get("tour_name", "")
        or ""
    )
    if not event_name:
        log_structured(
            logger,
            logging.WARNING,
            f"Event {event_id} has no name — skipping",
            status="skipped",
            reason="missing_name",
            entity_type="event",
            entity_id=event_id,
        )
        return None

    # Parse category (optional at tournament level — MS/WS filtering happens at match level)
    category = _parse_category(event_name)

    # Parse date
    date_raw = (
        record.get("vw_tournaments___tour_end_raw", "")
        or record.get("tour_end_raw", "")
        or record.get("tour_end", "")
    )
    try:
        event_date = pd.to_datetime(date_raw)
    except (ValueError, TypeError):
        log_structured(
            logger,
            logging.WARNING,
            f"Event {event_id} — cannot parse date {date_raw!r}, skipping",
            status="skipped",
            reason="malformed_date",
            entity_type="event",
            entity_id=event_id,
        )
        return None

    return {
        "event_id": event_id,
        "tournament_id": tournament_id,
        "event_name": event_name,
        "event_date": event_date,
        "event_category": category,
    }


# ---------------------------------------------------------------------------
# Pagination loop
# ---------------------------------------------------------------------------


def fetch_all_events(logger: logging.Logger) -> tuple[pd.DataFrame, int]:
    """Paginate through the ITTF events API.

    Returns a tuple of (DataFrame of events, count of skipped records).
    """
    headers = get_headers()
    all_events: list[dict] = []
    skipped_count = 0
    offset = 0
    page_size = 100

    while True:
        log_structured(logger, logging.INFO, f"Fetching events at offset {offset}", status="fetching")

        url = (
            f"{ITTF_BASE_URL}index.php?option=com_fabrik&view=list"
            f"&listid=27&Itemid=268&format=json&limit27=100&limitstart27={offset}"
        )
        data = fetch_json(url, headers)

        if data is None:
            log_structured(logger, logging.ERROR, f"Failed to fetch page at offset {offset}")
            break

        # The ITTF Fabrik API returns [[...records...]] (list of lists)
        if isinstance(data, list):
            rows = data[0] if data and isinstance(data[0], list) else data
        elif isinstance(data, dict):
            rows = data.get("rows", [])
            if isinstance(rows, dict):
                rows = list(rows.values())
        else:
            rows = []

        if not rows:
            log_structured(logger, logging.INFO, "No more rows returned — pagination complete")
            break

        for row in rows:
            # rows may be nested: each row can be a dict of column->value
            # or wrapped in another layer
            if isinstance(row, dict) and "data" in row:
                row = row["data"]

            event = _extract_event(logger, row)
            if event is not None:
                all_events.append(event)
            else:
                skipped_count += 1

        total = data.get("total", None) if isinstance(data, dict) else None
        fetched_so_far = offset + len(rows)

        log_structured(
            logger,
            logging.INFO,
            f"Page done: {len(rows)} rows, {fetched_so_far} fetched so far"
            + (f" / {total} total" if total is not None else ""),
        )

        # Stop conditions: fewer results than page size, or we have them all
        if len(rows) < page_size:
            break
        if total is not None:
            try:
                if fetched_so_far >= int(total):
                    break
            except (ValueError, TypeError):
                pass

        offset += page_size
        time.sleep(RATE_LIMIT_EVENTS)

    log_structured(
        logger,
        logging.INFO,
        f"Fetching complete: {len(all_events)} events collected, "
        f"{skipped_count} skipped",
        status="complete",
    )

    if not all_events:
        return pd.DataFrame(columns=list(EVENT_DTYPES.keys())), skipped_count

    df = pd.DataFrame(all_events)
    df = validate_schema(df, EVENT_DTYPES)
    df = df.drop_duplicates(subset=["event_id"])
    df = df.sort_values("event_date").reset_index(drop=True)
    return df, skipped_count


# ---------------------------------------------------------------------------
# Resumability — merge with existing file
# ---------------------------------------------------------------------------

OUTPUT_PATH = DATA_DIR / "events_index.parquet"


def _merge_with_existing(
    logger: logging.Logger, df_new: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """If events_index.parquet exists, merge in only genuinely new events.

    Returns (merged_df, changed) where *changed* is True when new events
    were actually added (or the file didn't previously exist).
    """
    if not OUTPUT_PATH.exists():
        return df_new, True

    df_existing = pd.read_parquet(OUTPUT_PATH)
    log_structured(
        logger,
        logging.INFO,
        f"Existing file has {len(df_existing)} events; "
        f"new fetch has {len(df_new)} events",
    )

    existing_ids = set(df_existing["event_id"])
    df_additions = df_new[~df_new["event_id"].isin(existing_ids)]

    if df_additions.empty:
        log_structured(logger, logging.INFO, "No new events to add")
        return df_existing, False

    log_structured(logger, logging.INFO, f"Adding {len(df_additions)} new events")
    df_merged = pd.concat([df_existing, df_additions], ignore_index=True)
    df_merged = validate_schema(df_merged, EVENT_DTYPES)
    df_merged = df_merged.drop_duplicates(subset=["event_id"])
    df_merged = df_merged.sort_values("event_date").reset_index(drop=True)
    return df_merged, True


# ---------------------------------------------------------------------------
# Certification report
# ---------------------------------------------------------------------------


def _print_certification(
    logger: logging.Logger, df: pd.DataFrame, skipped: int = 0
) -> None:
    """Print and log the Stage 1 certification summary."""
    lines = [
        "",
        "=" * 60,
        "  STAGE 1 CERTIFICATION — Event Index",
        "=" * 60,
        f"  Total events:          {len(df)}",
    ]

    # Events by year
    if not df.empty:
        year_dist = df["event_date"].dt.year.value_counts().sort_index()
        lines.append("  Events by year:")
        for year, count in year_dist.items():
            lines.append(f"    {year}: {count}")

        # Category split
        cat_dist = df["event_category"].value_counts()
        lines.append("  Events by category:")
        for cat, count in cat_dist.items():
            lines.append(f"    {cat}: {count}")

    lines.append(f"  Skipped (unclassified): {skipped}")

    # Sample: oldest 5 and newest 5
    if len(df) >= 10:
        oldest = df.head(5)[["event_id", "event_name", "event_date"]].to_string(
            index=False
        )
        newest = df.tail(5)[["event_id", "event_name", "event_date"]].to_string(
            index=False
        )
        lines.append("  Oldest 5 events:")
        for line in oldest.split("\n"):
            lines.append(f"    {line}")
        lines.append("  Newest 5 events:")
        for line in newest.split("\n"):
            lines.append(f"    {line}")
    elif not df.empty:
        sample = df[["event_id", "event_name", "event_date"]].to_string(index=False)
        lines.append("  All events:")
        for line in sample.split("\n"):
            lines.append(f"    {line}")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    log_structured(logger, logging.INFO, report, status="certification")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Stage 1 event index scraper."""
    logger = setup_stage_logger("fetch_events")
    ensure_dirs()

    log_structured(logger, logging.INFO, "Stage 1 — fetch_events starting", status="start")

    df_new, skipped = fetch_all_events(logger)

    df_final, changed = _merge_with_existing(logger, df_new)

    if changed:
        df_final.to_parquet(OUTPUT_PATH, index=False)
        log_structured(
            logger,
            logging.INFO,
            f"Saved {len(df_final)} events to {OUTPUT_PATH}",
            status="saved",
        )
    else:
        log_structured(
            logger,
            logging.INFO,
            f"No new events — skipping write ({len(df_final)} events unchanged)",
            status="skipped_write",
        )

    _print_certification(logger, df_final, skipped=skipped)
    log_structured(logger, logging.INFO, "Stage 1 — fetch_events complete", status="done")


if __name__ == "__main__":
    main()
