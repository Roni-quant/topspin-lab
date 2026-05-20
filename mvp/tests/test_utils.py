# mvp/tests/test_utils.py
import pytest
from pathlib import Path
import tempfile
import csv
from mvp.utils import (
    compute_implied_probability,
    compute_edge,
    load_csv,
    append_to_csv,
    map_player_name,
)

def test_compute_implied_probability():
    """Test converting odds to implied probability."""
    # 2.0 odds = 50% implied probability
    assert compute_implied_probability(2.0) == pytest.approx(0.5)

    # 1.5 odds = 66.67% implied probability
    assert compute_implied_probability(1.5) == pytest.approx(1/1.5, rel=1e-3)

    # 3.0 odds = 33.33% implied probability
    assert compute_implied_probability(3.0) == pytest.approx(1/3.0, rel=1e-3)

def test_compute_edge():
    """Test computing edge between model and implied probability."""
    # Model 65%, implied 50% = 15% edge
    assert compute_edge(0.65, 0.50) == pytest.approx(0.15)

    # Model 40%, implied 50% = -10% edge (no value)
    assert compute_edge(0.40, 0.50) == pytest.approx(-0.10)

    # Model 55%, implied 55% = 0% edge (fair)
    assert compute_edge(0.55, 0.55) == pytest.approx(0.0)

def test_map_player_name():
    """Test mapping player names to IDs."""
    mapping = {
        "Fan Zhendong": "player_1",
        "Hugo Calderano": "player_2",
        "Tomokazu Harimoto": "player_3",
    }

    assert map_player_name("Fan Zhendong", mapping) == "player_1"
    assert map_player_name("Hugo Calderano", mapping) == "player_2"
    assert map_player_name("Unknown Player", mapping) is None

def test_load_csv_file_not_exists():
    """Test loading from non-existent file."""
    result = load_csv(Path("/nonexistent/path/trades.csv"))
    assert result == []

def test_load_csv_with_data():
    """Test loading CSV with data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"

        # Create a test CSV file
        fieldnames = ["match_id", "player_a", "player_b", "status"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"match_id": "m1", "player_a": "p1", "player_b": "p2", "status": "pending"})
            writer.writerow({"match_id": "m2", "player_a": "p3", "player_b": "p4", "status": "placed"})

        # Load and verify
        result = load_csv(csv_path)
        assert len(result) == 2
        assert result[0]["match_id"] == "m1"
        assert result[1]["status"] == "placed"

def test_append_to_csv_new_file():
    """Test appending to a new CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"

        row = {
            "match_id": "m1",
            "fetch_ts": "2026-04-06T10:00:00Z",
            "player_a": "p1",
            "player_b": "p2",
            "odds_a": "1.95",
            "odds_b": "1.90",
            "model_prob_a": "0.55",
            "model_prob_b": "0.45",
            "implied_prob_a": "0.51",
            "implied_prob_b": "0.53",
            "selected_side": "",
            "edge": "0.04",
            "stake": "",
            "result": "",
            "pnl": "",
            "status": "pending",
            "event": "london-2026",
        }

        # Append to new file
        append_to_csv(csv_path, row)

        # Verify file was created with header and row
        assert csv_path.exists()
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["match_id"] == "m1"

def test_append_to_csv_existing_file():
    """Test appending to an existing CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"

        row1 = {
            "match_id": "m1",
            "fetch_ts": "2026-04-06T10:00:00Z",
            "player_a": "p1",
            "player_b": "p2",
            "odds_a": "1.95",
            "odds_b": "1.90",
            "model_prob_a": "0.55",
            "model_prob_b": "0.45",
            "implied_prob_a": "0.51",
            "implied_prob_b": "0.53",
            "selected_side": "",
            "edge": "0.04",
            "stake": "",
            "result": "",
            "pnl": "",
            "status": "pending",
            "event": "london-2026",
        }

        row2 = {
            "match_id": "m2",
            "fetch_ts": "2026-04-06T11:00:00Z",
            "player_a": "p3",
            "player_b": "p4",
            "odds_a": "1.85",
            "odds_b": "2.00",
            "model_prob_a": "0.60",
            "model_prob_b": "0.40",
            "implied_prob_a": "0.54",
            "implied_prob_b": "0.50",
            "selected_side": "",
            "edge": "0.06",
            "stake": "",
            "result": "",
            "pnl": "",
            "status": "pending",
            "event": "london-2026",
        }

        # Append two rows
        append_to_csv(csv_path, row1)
        append_to_csv(csv_path, row2)

        # Verify both rows are in file
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["match_id"] == "m1"
            assert rows[1]["match_id"] == "m2"
