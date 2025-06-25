# Braze Canvas Forecasting Tool

A comprehensive Python tool for **time-series forecasting** of Braze Canvas data. This tool predicts when Canvas sends will decay to zero (the "quiet date") using simple linear regression.

### Installation

```bash
make install
```

### Fetch

**⚠️ API Key Required** - You need a valid Braze REST API key with canvas read permissions.

```bash
make ingest-historical # examine this command then add your api key
BRAZE_REST_KEY=your-braze-rest-key pyenv exec python -m src.ingest_historical --days 90 --filter-prefix "transactional"
```

This command will:
- Fetch historical Canvas statistics for all 100+ Canvases
- Create a hierarchical data structure: `data/canvas-id/step-id/channel.jsonl`
- Process thousands of event records with actual engagement metrics
- Handle API rate limiting and chunking automatically

### Forecast

```bash
# Run linear regression analysis on the collected data
make forecast
```
