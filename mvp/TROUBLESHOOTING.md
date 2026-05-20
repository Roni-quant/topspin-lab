# Troubleshooting Guide

Common issues and solutions.

## API Issues

### "No module named requests"
```bash
pip install requests
```

### "API returns 401 Unauthorized"
- Verify ODDS_API_KEY is set: `echo $ODDS_API_KEY`
- Verify it's a valid key from https://the-odds-api.com
- If key is valid, API might be down - check https://status.the-odds-api.com

### "No London 2026 matches found"
- Verify tournament is happening (April 28 – May 10)
- Check API filtering logic in `fetch_odds.py` - may need adjustment for actual API structure
- Fallback: Manually add match odds to trades.csv

## Model Issues

### "FileNotFoundError: Model not found"
- Verify MODEL_PATH in `config.py` points to actual model
- Run: `ls models/random_forest_v2.pkl`
- If missing, retrain model or use dummy model

### "ValueError: Feature names mismatch"
- Check FEATURE_NAMES in `config.py` match training features
- Verify order: elo_difference, recent_win_rate, matches_last_30_days, opponent_recent_form, momentum

## CSV Issues

### "No pending trades found"
- Check `trades.csv` is in `mvp/data/`
- Verify fetch_odds.py ran and created entries
- Run: `head mvp/data/trades.csv`

### "CSV is corrupted / strange formatting"
- Backup: `cp mvp/data/trades.csv mvp/data/trades_broken.csv`
- Delete: `rm mvp/data/trades.csv`
- Restart: Re-run fetch_odds.py (creates fresh CSV)

## Metrics Issues

### "ROI is negative even though most bets won"
- This is possible if losing bets have larger stakes than winning bets
- Check individual P&L in trades.csv
- May indicate edge calculation is wrong or odds were heavily skewed

### "Average edge is 0% even though bets were placed"
- Verify edge calculation in predict_and_log.py: edge = model_prob - implied_prob
- Check that model_prob_a, model_prob_b are populated in trades.csv
- If empty, predict_and_log.py didn't run

## General

### "How do I reset and start over?"
```bash
# Backup current trades
cp mvp/data/trades.csv mvp/data/trades_backup_$(date +%s).csv

# Delete trade history
rm mvp/data/trades.csv mvp/data/summary.json

# Restart from fetch
python mvp/fetch_odds.py
```

### "Can I run scripts manually for specific dates?"
- Modify ODDS_API query in `fetch_odds.py` to filter by date
- Then run predict/settle/analyze on that date's trades

### "What if The Odds API is down?"
- Manually enter odds into trades.csv for key matches
- Or wait for API to recover
- Keep a backup of odds from other sources

---

**Still stuck?** Check script logs or reach out for help.
