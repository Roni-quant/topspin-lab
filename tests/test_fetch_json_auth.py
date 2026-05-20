"""Tests for thread-safe auth refresh in fetch_json."""

import threading
import time
from unittest.mock import MagicMock, patch

from pipeline.http import fetch_json


def _make_response(status_code=200, json_data=None, content_type="application/json"):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"Content-Type": content_type}
    resp.json.return_value = json_data or {"data": []}
    resp.text = '{"data": []}'
    return resp


def _make_html_response():
    """Create a mock response that looks like HTML (triggers auth refresh)."""
    resp = _make_response(200, content_type="text/html")
    resp.text = "<html>login page</html>"
    resp.json.side_effect = ValueError("not json")
    return resp


def test_auth_refresh_double_check_only_one_thread_refreshes():
    """When two threads both detect stale cookies, only one should call refresh."""
    shared_headers = {"Cookie": "stale_cookie"}
    refresh_call_count = 0
    refresh_lock = threading.Lock()

    def mock_refresh(headers):
        nonlocal refresh_call_count
        with refresh_lock:
            refresh_call_count += 1
        time.sleep(0.05)  # simulate login latency
        headers["Cookie"] = "fresh_cookie"
        return True

    def mock_get(*args, **kwargs):
        if shared_headers.get("Cookie") == "stale_cookie":
            return _make_html_response()
        return _make_response(200, json_data={"data": [1]})

    with patch("pipeline.http.requests.get", side_effect=mock_get):
        with patch("pipeline.http._try_refresh_cookie", side_effect=mock_refresh):
            results = [None, None]
            errors = [None, None]

            def worker(idx):
                try:
                    results[idx] = fetch_json("http://example.com", shared_headers)
                except SystemExit:
                    errors[idx] = "SystemExit"

            t1 = threading.Thread(target=worker, args=(0,))
            t2 = threading.Thread(target=worker, args=(1,))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

    # Only one thread should have called _try_refresh_cookie
    assert refresh_call_count == 1
    # Both threads should have gotten a result (no SystemExit)
    assert errors == [None, None]
    assert all(r is not None for r in results)


def test_auth_refresh_called_when_cookie_stale():
    """When cookie is genuinely stale, refresh should be called."""
    headers = {"Cookie": "stale_cookie"}
    call_count = 0

    def get_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return _make_html_response()
        return _make_response(200, json_data={"result": "ok"})

    with patch("pipeline.http.requests.get", side_effect=get_side_effect):
        with patch("pipeline.http._try_refresh_cookie", return_value=True) as mock_refresh:
            result = fetch_json("http://example.com", headers)
            mock_refresh.assert_called_once()
