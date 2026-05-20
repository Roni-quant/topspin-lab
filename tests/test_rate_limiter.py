"""Tests for the RateLimiter class."""

import threading
import time

from pipeline.http import RateLimiter


def test_rate_limiter_enforces_min_interval():
    """Two sequential waits should be spaced at least min_interval apart."""
    limiter = RateLimiter(min_interval=0.1)
    limiter.wait()
    t0 = time.monotonic()
    limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.09  # small tolerance for timer precision


def test_rate_limiter_allows_concurrent_slot_assignment():
    """Multiple threads should get sequential time slots without blocking each other
    during sleep. Total wall time for N threads should be ~(N-1)*interval, not N*interval + N*work."""
    limiter = RateLimiter(min_interval=0.05)
    n_threads = 5
    start_times = [0.0] * n_threads
    barrier = threading.Barrier(n_threads)

    def worker(idx):
        barrier.wait()  # all threads start together
        limiter.wait()
        start_times[idx] = time.monotonic()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 5 threads should complete within ~0.25s (5 * 0.05), not 5 * (0.05 + work)
    sorted_times = sorted(start_times)
    total_spread = sorted_times[-1] - sorted_times[0]
    assert total_spread >= 0.15  # at least (n-1) * interval
    assert total_spread < 1.0  # should not take unreasonably long


def test_rate_limiter_zero_interval():
    """A zero interval should not block."""
    limiter = RateLimiter(min_interval=0.0)
    t0 = time.monotonic()
    for _ in range(10):
        limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
