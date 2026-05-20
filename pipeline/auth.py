"""Automated ITTF session cookie management via Joomla login."""

import logging
import os
import re
import time

import requests
from bs4 import BeautifulSoup

from pipeline.config import ITTF_BASE_URL, REQUEST_TIMEOUT

logger = logging.getLogger("pipeline.auth")

_last_login_attempt: float = 0
_LOGIN_COOLDOWN: int = 60  # seconds

LOGIN_URL = "https://results.ittf.link/index.php?option=com_users&view=login"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)


def login(username: str, password: str) -> str | None:
    """Programmatic Joomla login. Returns cookie header string or None."""
    global _last_login_attempt
    _last_login_attempt = time.time()

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    # GET the login page to find the CSRF token
    try:
        page = session.get(LOGIN_URL, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        logger.error("Failed to reach login page")
        return None

    if page.status_code != 200:
        logger.error("Login page returned HTTP %d", page.status_code)
        return None

    # Parse CSRF token: Joomla uses a hidden input with a 32-char hex name and value="1"
    soup = BeautifulSoup(page.text, "html.parser")
    csrf_input = soup.find(
        "input",
        attrs={"type": "hidden", "value": "1", "name": re.compile(r"^[a-f0-9]{32}$")},
    )
    if csrf_input is None:
        logger.error("CSRF token not found on login page")
        return None

    csrf_name = csrf_input["name"]

    # POST login
    payload = {
        "username": username,
        "password": password,
        "task": "user.login",
        csrf_name: "1",
    }
    try:
        resp = session.post(LOGIN_URL, data=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        logger.error("Login POST failed")
        return None

    if resp.status_code >= 400:
        logger.error("Login failed: HTTP %d", resp.status_code)
        return None

    # Check that we got session cookies
    if not session.cookies:
        logger.error("Login failed: no session cookies received")
        return None

    cookie_header = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
    logger.info("Login successful")
    return cookie_header


def refresh_cookie() -> str | None:
    """Read credentials from env, login, update os.environ in-memory."""
    username = os.environ.get("ITTF_USERNAME")
    password = os.environ.get("ITTF_PASSWORD")

    if not username or not password:
        return None

    # Cooldown check
    if time.time() - _last_login_attempt < _LOGIN_COOLDOWN:
        logger.warning(
            "Login cooldown active (last attempt <60s ago) — skipping refresh"
        )
        return None

    cookie = login(username, password)
    if cookie:
        os.environ["ITTF_COOKIE_HEADER"] = cookie
    return cookie


def validate_cookie(cookie: str) -> bool:
    """Lightweight check — GET events endpoint with limit27=1."""
    try:
        resp = requests.get(
            ITTF_BASE_URL + "index.php?option=com_fabrik&format=json"
            "&task=plugin.cron.cronRun&listid=27&limit27=1",
            headers={"Cookie": cookie, "User-Agent": _USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        return False
    return resp.status_code == 200
