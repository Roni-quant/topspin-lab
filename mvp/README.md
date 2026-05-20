# London 2026 MVP: Edge Validation

Lightweight trading strategy validation framework for the Elo model on the 2026 ITTF World Team Table Tennis Championships.

**Question:** Does the model have a real edge vs professional bookmakers?

**Answer:** Run this MVP and analyze the results. If ROI ≥ 0% over 30+ bets, the edge is real.

## Quick Start

### Prerequisites
- Python 3.10+
- BetsAPI token: https://betsapi.com (~$10/mo, table tennis odds)
- Trained Random Forest model at `models/random_forest_v2.pkl`

### 1-Minute Setup

```bash
# Set API token (from betsapi.com)
export BETSAPI_TOKEN="your_token_here"

# Verify model exists
ls models/random_forest_v2.pkl

# Run tests to verify installation
pytest mvp/tests/ -v
```

All green? You're ready.

## Daily Workflow (April 28 – May 10)

### Morning (7am UTC)
```bash
cd mvp
python fetch_odds.py        # Fetch London 2026 match odds
python predict_and_log.py   # Run predictions, place bets where edge > 3%
```

### Afternoon/Evening (as matches complete)
```bash
python settle_results.py    # Fetch results, calculate P&L
python analyze.py           # Print summary and save summary.json
```

### Check Results
```bash
streamlit run mvp/dashboard.py   # Visual dashboard
cat data/summary.json            # Machine-readable metrics
cat data/trades.csv              # Human-readable trade log
```

## Files

| File | Purpose |
|------|---------|
| `fetch_odds.py` | Query BetsAPI for match odds |
| `predict_and_log.py` | Run RF model, place bets when edge > 3% |
| `settle_results.py` | Fetch results, compute P&L |
| `analyze.py` | Compute ROI, win rate, edge metrics |
| `dashboard.py` | Streamlit performance dashboard |
| `data/trades.csv` | Complete trade log (all predictions & outcomes) |
| `data/summary.json` | Final metrics JSON |

## Success Criteria

MVP succeeds if **all three** are true:
- **Bets placed:** ≥30
- **ROI:** ≥0% (positive or break-even)
- **Avg edge:** >0% (selected edges are positive on average)

If all three: Model has a real edge. Consider next steps (real betting, model refinement, etc.)

## Configuration

Edit `config.py` to customize:
- `BETSAPI_TOKEN` - Your BetsAPI token (from betsapi.com)
- `MIN_EDGE_THRESHOLD` - Minimum edge to place bet (default 3%)
- `STAKE` - Bet amount per trade (default $10)
- `MODEL_PATH` - Path to trained RF model

## Troubleshooting

See `TROUBLESHOOTING.md` for common issues.

## Architecture

```
fetch_odds.py
    ↓ (odds → CSV)
predict_and_log.py
    ↓ (predictions + edge → CSV)
settle_results.py
    ↓ (results + P&L → CSV)
analyze.py
    ↓ (summary.json + console output)
```

Single CSV table (`trades.csv`) flows through all scripts. Human-readable, easy to inspect.

## Testing

```bash
# Run all tests
pytest mvp/tests/ -v

# Check coverage
pytest mvp/tests/ --cov=mvp --cov-report=term-missing
```

Expected: 51+ tests passing.

## Next Steps (After Tournament)

- Analyze `summary.json` for ROI and edge
- If successful: consider real betting or model improvements
- If unsuccessful: investigate edge calculation or feature engineering

---

**Created:** April 2026
**Last Updated:** April 12, 2026
