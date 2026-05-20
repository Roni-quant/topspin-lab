# mvp/tests/test_analyze.py
import pytest
from mvp.analyze import compute_metrics

def test_compute_metrics():
    """Test metrics calculation."""
    trades = [
        {
            "status": "settled",
            "result": "win",
            "pnl": "9.50",
            "edge": "0.05",
            "stake": "10.0",
            "event": "london-2026",
        },
        {
            "status": "settled",
            "result": "loss",
            "pnl": "-10.0",
            "edge": "0.04",
            "stake": "10.0",
            "event": "london-2026",
        },
        {
            "status": "placed",
            "result": "",
            "pnl": "",
            "edge": "0.06",
            "stake": "10.0",
            "event": "london-2026",
        },
    ]

    metrics = compute_metrics(trades)

    assert metrics["bets_placed"] == 3
    assert metrics["bets_settled"] == 2
    assert metrics["bets_won"] == 1
    assert metrics["bets_lost"] == 1
    assert metrics["win_rate_pct"] == pytest.approx(50.0)
    assert metrics["total_pnl_usd"] == pytest.approx(-0.50)

def test_compute_metrics_empty():
    """Test metrics calculation with no settled trades."""
    trades = [
        {
            "status": "placed",
            "result": "",
            "pnl": "",
            "edge": "0.06",
            "stake": "10.0",
            "event": "london-2026",
        },
    ]

    metrics = compute_metrics(trades)

    assert metrics["bets_placed"] == 1
    assert metrics["bets_settled"] == 0
    assert metrics["bets_won"] == 0
    assert metrics["bets_lost"] == 0
    assert metrics["win_rate_pct"] == 0.0
    assert metrics["total_pnl_usd"] == 0.0

def test_compute_metrics_all_wins():
    """Test metrics calculation with all winning trades."""
    trades = [
        {
            "status": "settled",
            "result": "win",
            "pnl": "15.0",
            "edge": "0.10",
            "stake": "10.0",
            "event": "london-2026",
        },
        {
            "status": "settled",
            "result": "win",
            "pnl": "20.0",
            "edge": "0.08",
            "stake": "10.0",
            "event": "london-2026",
        },
    ]

    metrics = compute_metrics(trades)

    assert metrics["bets_placed"] == 2
    assert metrics["bets_settled"] == 2
    assert metrics["bets_won"] == 2
    assert metrics["bets_lost"] == 0
    assert metrics["win_rate_pct"] == pytest.approx(100.0)
    assert metrics["total_pnl_usd"] == pytest.approx(35.0)

def test_compute_metrics_roi_and_edges():
    """Test ROI and average edge calculations."""
    trades = [
        {
            "status": "settled",
            "result": "win",
            "pnl": "25.0",
            "edge": "0.08",
            "stake": "10.0",
            "event": "london-2026",
        },
        {
            "status": "placed",
            "result": "",
            "pnl": "",
            "edge": "0.10",
            "stake": "10.0",
            "event": "london-2026",
        },
        {
            "status": "pending",
            "result": "",
            "pnl": "",
            "edge": "",
            "stake": "",
            "event": "london-2026",
        },
    ]

    metrics = compute_metrics(trades)

    assert metrics["bets_placed"] == 2  # placed and pending with stakes
    assert metrics["bets_settled"] == 1
    assert metrics["total_pnl_usd"] == 25.0
    assert metrics["roi_pct"] == pytest.approx(2.5)  # 25 / 1000 starting capital
    assert metrics["average_edge_pct"] == pytest.approx(9.0)  # (0.08 + 0.10) / 2 * 100
