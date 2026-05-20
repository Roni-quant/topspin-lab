# Deployment Guide

Step-by-step instructions to deploy and run the MVP during the London 2026 World Team Championships (April 28 – May 10).

## Pre-Tournament Setup (By April 27)

### 1. Verify Environment

```bash
cd /path/to/elo-sports-lab
python --version       # Should be 3.10+
pip list | grep -i sklearn pytest requests
```

### 2. Set API Key

```bash
export ODDS_API_KEY="your_actual_key"
echo $ODDS_API_KEY     # Verify it's set
```

### 3. Verify Model

```bash
ls -lh models/random_forest_v2.pkl
python -c "from mvp.models import load_model; from pathlib import Path; model = load_model(Path('models/random_forest_v2.pkl')); print('✓ Model loads OK')"
```

### 4. Test API Connectivity

```bash
python -c "
from mvp.config import ODDS_API_KEY, ODDS_API_BASE
import requests
url = f'{ODDS_API_BASE}/sports'
r = requests.get(url, params={'apiKey': ODDS_API_KEY})
print(f'✓ API responds: {r.status_code}')
"
```

### 5. Run Tests

```bash
pytest mvp/tests/ -v
# Expected: 34+ passed
```

## Daily Operations (April 28 – May 10)

### Morning Routine (7am UTC)

```bash
# 1. Fetch odds
python mvp/fetch_odds.py

# 2. Run predictions
python mvp/predict_and_log.py

# 3. Check how many bets were placed
grep "placed" mvp/data/trades.csv | wc -l

# 4. Log output
python mvp/analyze.py > /tmp/mvp_$(date +%Y%m%d_%H%M%S).log
```

### Throughout the Day

```bash
# As matches complete, fetch results
python mvp/settle_results.py

# Check running metrics
python mvp/analyze.py
```

### Evening Check

```bash
# Review the day's summary
cat mvp/data/summary.json | jq .metrics
```

## Monitoring

Check these files in order of priority:

1. **mvp/data/summary.json** - Current metrics
2. **mvp/data/trades.csv** - Individual trades (open in Excel)
3. Logs - Any errors or warnings from scripts

## Rollback / Troubleshooting

If something breaks:

1. **Stop everything**
2. **Backup trades.csv**: `cp mvp/data/trades.csv mvp/data/trades_backup_$(date +%s).csv`
3. **Check logs**: Look for error messages
4. **See TROUBLESHOOTING.md**
5. **Restart**: Once fixed, re-run scripts

## Post-Tournament (May 11–12)

```bash
# Generate final report
python mvp/analyze.py

# Inspect trades.csv
wc -l mvp/data/trades.csv
grep "placed" mvp/data/trades.csv | wc -l
grep "settled" mvp/data/trades.csv | wc -l

# Save results
cp mvp/data/summary.json ./london_2026_results_$(date +%Y%m%d).json
```

---

**Ready to deploy?** Run pre-tournament checks above, then start daily operations on April 28.
