# Braze Canvas Forecasting Tool

A comprehensive Python tool for **time-series forecasting** of Braze Canvas data. This tool predicts when Canvas sends will decay to zero (the "quiet date") using simple linear regression.

### Installation

1. **Clone and setup environment:**

```bash
make install
```

### Step 1: Retrieve Historical Data

**⚠️ API Key Required** - You need a valid Braze REST API key with canvas read permissions.

```bash
make ingest-historical # examine this command then add your api key
BRAZE_REST_KEY=your-braze-rest-key pyenv exec python -m src.ingest_historical --days 90 --filter-prefix "transactional" #  e.g.
```

This command will:
- Fetch historical Canvas statistics for all 100+ Canvases
- Create a hierarchical data structure: `data/canvas-id/step-id/channel.jsonl`
- Process thousands of event records with actual engagement metrics
- Handle API rate limiting and chunking automatically

### Step 2: Process with Linear Regression

```bash
# Run linear regression analysis on the collected data
make forecast
```
