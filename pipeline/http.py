"""Shared HTTP fetch with retry, backoff, and auth-failure handling."""

import json
import logging
import threading
import time

import requests

from pipeline.config import MAX_RETRIES, REQUEST_TIMEOUT, RETRYABLE_STATUS_CODES

logger = logging.getLogger("pipeline.http")


class RateLimiter:
    """Thread-safe rate limiter using time-slot assignment.

    Threads acquire sequential time slots under a lock, then sleep outside
    the lock so network I/O can overlap.
    """

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            target = max(now, self._last_call + self._min_interval)
            self._last_call = target
        sleep_for = target - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)

_auth_lock = threading.Lock()


def _looks_like_json(resp: requests.Response) -> bool:
    """Check if response content-type or body looks like JSON."""
    ct = resp.headers.get("Content-Type", "")
    if "json" in ct:
        return True
    body = resp.text.strip()
    return body.startswith(("{", "["))


def _parse_json(resp: requests.Response, url: str) -> dict | None:
    """Try to parse JSON from response, logging on failure."""
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Malformed JSON from %s: %s", url, exc)
        return None


def _try_refresh_cookie(headers: dict) -> bool:
    """Attempt to refresh the session cookie. Returns True on success."""
    from pipeline.auth import refresh_cookie

    new_cookie = refresh_cookie()
    if new_cookie:
        headers["Cookie"] = new_cookie
        logger.info("Session cookie refreshed successfully")
        return True
    return False


def fetch_json(url: str, headers: dict) -> dict | None:
    """GET *url* and return parsed JSON, with retry and exponential backoff.

    - Retries on RETRYABLE_STATUS_CODES and timeouts (up to MAX_RETRIES).
    - Detects expired sessions (401/403 or 200 with HTML body) and auto-refreshes.
    - Returns None on non-retryable failures or after exhausting retries.
    """
    if "User-Agent" not in headers:
        headers["User-Agent"] = _USER_AGENT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "Timeout (attempt %d/%d) for %s — retrying in %ds",
                attempt, MAX_RETRIES, url, backoff,
            )
            time.sleep(backoff)
            continue
        except requests.exceptions.RequestException as exc:
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "Request error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
            continue

        # Detect session expiry: either explicit 401/403 or 200 with non-JSON body
        needs_refresh = (
            resp.status_code in (401, 403)
            or (resp.status_code == 200 and not _looks_like_json(resp))
        )

        if needs_refresh:
            stale_cookie = headers.get("Cookie")
            with _auth_lock:
                if headers.get("Cookie") != stale_cookie:
                    # Another thread already refreshed; retry with new cookie
                    continue
                if _try_refresh_cookie(headers):
                    continue  # retry with refreshed cookie
                # Refresh failed — credentials are broken
                logger.critical(
                    "Session expired (HTTP %d) and refresh failed. "
                    "Check credentials in .env and re-run.",
                    resp.status_code,
                )
                raise SystemExit(1)

        if resp.status_code in RETRYABLE_STATUS_CODES:
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "HTTP %d (attempt %d/%d) — retrying in %ds",
                resp.status_code, attempt, MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            continue

        if resp.status_code != 200:
            logger.warning("HTTP %d for %s — skipping", resp.status_code, url)
            return None

        return _parse_json(resp, url)

    logger.warning("Exhausted %d retries for %s", MAX_RETRIES, url)
    return None
