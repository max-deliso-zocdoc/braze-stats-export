# Braze Canvas Forecasting Tool

A comprehensive Python tool for **time-series forecasting** of Braze Canvas data. This tool predicts when Canvas sends will decay to zero (the "quiet date") using advanced machine learning models including polynomial regression, power law, ARIMA, and ensemble methods.

## Features

🚀 **Core Functionality**
- Export complete canvas list with metadata
- Fetch detailed canvas configurations including steps, variants, and workflows
- Comprehensive request logging with performance metrics
- Data persistence in structured JSON format

📊 **Analytics & Insights**
- Canvas tag frequency analysis
- Recent activity tracking (last 30 days)
- Channel distribution analysis
- Schedule type breakdowns
- Workflow complexity metrics (steps per canvas)
- Canvas status distribution (enabled/disabled/archived)

🔍 **Detailed Analysis**
- Step type distribution across workflows
- Canvas variant analysis
- Performance monitoring with response times
- Success rate tracking

🔮 **Enhanced Time-Series Forecasting**
- Historical Canvas send data ingestion with append-only JSONL storage
- **Advanced Models:** Polynomial, Power Law, ARIMA, Linear, Exponential
- **Ensemble Methods:** Weighted model combination for higher confidence
- **Cross-Validation:** Time series splitting for reliable validation
- **Bootstrap Confidence Intervals:** 95% prediction intervals
- **Parallel Processing:** Multi-threaded Canvas analysis
- **Canvas Filtering:** Focus on specific Canvas types (e.g., "Transactional")
- Confidence scoring with enhanced calculation methods
- Trend analysis (declining, stable, growing) with strength measurement
- Comprehensive forecast reports with urgent Canvas identification

## Quick Start

### Prerequisites
- Python 3.11.4+
- Valid Braze REST API key with canvas read permissions
- pyenv (recommended for Python version management)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd braze-stats-export
make setup
```

2. **Activate the virtual environment:**
```bash
export PYENV_VERSION=braze-extractor-env
```

3. **Run enhanced forecasting:**
```bash
BRAZE_REST_KEY=your-braze-rest-key make ingest-historical-extended
make forecast-enhanced
```

Or run directly:
```bash
BRAZE_REST_KEY=your-braze-rest-key python -m src.ingest_historical --days 30
python -m src.forecast_enhanced --verbose
```

## Historical Canvas Analysis Workflow

For comprehensive historical analysis and quiet date prediction, use this two-step process:

### Step 1: Retrieve Historical Data

**⚠️ API Key Required** - You need a valid Braze REST API key with canvas read permissions.

```bash
# Retrieve 60 days of historical Canvas data (recommended for robust analysis)
BRAZE_REST_KEY=your-braze-rest-key python src/ingest_historical.py --days 60 --verbose

# Or specify a custom date range
BRAZE_REST_KEY=your-braze-rest-key python src/ingest_historical.py --start-date 2023-11-01 --end-date 2023-12-31 --verbose
```

This command will:
- Fetch historical Canvas statistics for all 100+ Canvases
- Create a hierarchical data structure: `data/canvas-id/step-id/channel.jsonl`
- Process thousands of event records with actual engagement metrics
- Handle API rate limiting and chunking automatically

### Step 2: Process with Linear Regression

```bash
# Run linear regression analysis on the collected data
python src/forecast_quiet_dates.py --verbose

# Optional: Adjust the quiet threshold (default: 5 daily sends)
python src/forecast_quiet_dates.py --quiet-threshold 10 --verbose
```

This command will:
- Analyze Canvas trends (declining, stable, growing)
- Generate quiet date predictions using linear and exponential decay models
- Provide confidence scores for each prediction
- Create comprehensive forecast reports

**Expected Results:**
- Process 100 Canvas directories with step-based data
- Identify Canvases with predictable quiet dates
- Generate confidence-scored predictions
- Export detailed JSON reports with urgent Canvas identification

## Time-Series Forecasting

### Generate Forecasts

The tool provides a comprehensive forecasting system that analyzes historical Canvas data and predicts when Canvas sends will decay to approximately zero (the "quiet date").

**Note:** Daily ingestion has been removed. Use `ingest_historical.py` to collect historical data first.

**Generate forecasts from existing data:**
```bash
make forecast
```

**Create sample data for testing:**
```bash
make sample-data
```

### Forecast Output

The tool generates comprehensive reports with:

```
🔮 CANVAS QUIET DATE FORECAST REPORT
======================================================================

📊 OVERVIEW:
  • Total Canvases Analyzed: 25
  • Predictable Quiet Dates: 18
  • Unpredictable: 7
  • Going Quiet Soon (≤30 days): 3
  • Going Quiet Later (>30 days): 15

📈 CURRENT TRENDS:
  📉 Declining: 12
  📊 Stable: 8
  📈 Growing: 3
  ❓ Insufficient Data: 2

🎯 PREDICTION CONFIDENCE:
  • High (≥70%): 8
  • Medium (40-70%): 7
  • Low (<40%): 3

⚠️  URGENT: CANVASES GOING QUIET SOON
   Canvas ID                         Quiet Date    Days  Confidence  Trend
   ----------------------------------------------------------------
   canvas-lifecycle-welcome          2024-01-15      8    85.2%      declining
   canvas-retention-day7             2024-01-22     15    72.1%      declining
   canvas-promo-holiday              2024-01-25     18    91.5%      declining
```

### Forecasting Methods

The tool uses multiple regression models and automatically selects the best fit:

1. **Linear Decay Model**: `y = ax + b`
   - Best for steady declining trends
   - Predicts when sends will reach the quiet threshold

2. **Exponential Decay Model**: `y = a * e^(-bx) + c`
   - Better for accelerating decay patterns
   - Handles rapid drop-offs more accurately

**Confidence Scoring:**
- Based on R-squared values from regression models
- Adjusted for data quality (more data = higher confidence)
- Reduced for growing trends (less reliable for prediction)

## Output Files

The tool generates several files:

**Canvas Analysis:**
- **`canvas_list.json`** - Complete list of all canvases with basic metadata
- **`canvas_details.json`** - Detailed configurations for sample canvases
- **`request_log.json`** - Complete log of all API requests with performance data
- **`braze_export.log`** - Application logs with timestamps

**Time-Series Forecasting:**
- **`data/canvas-id/step-id/*.jsonl`** - Historical time-series data for each Canvas step and channel (append-only)
- **`forecast_report_YYYYMMDD_HHMMSS.json`** - Comprehensive forecast analysis
- **`braze_historical_ingest.log`** - Historical data ingestion logs
- **`braze_forecast.log`** - Forecasting process logs

## Sample Output

```
==================================================
CANVAS STATISTICS
==================================================
Total Canvases: 100
Total Unique Tags: 28
Canvases Updated (Last 30 days): 0
Canvases Without Tags: 3

Most Common Tags:
  Lifecycle: 49
  Approved: 28
  QA: 23
  Well Guide: 21
  Ad Hoc: 16

🔗 API PERFORMANCE:
  • Total API Requests: 6
  • Successful Requests: 6
  • Success Rate: 100.0%
  • Average Response Time: 273.02ms

🔍 DETAILED ANALYSIS:
  • Canvases Analyzed: 5
  • Channel Distribution:
    - Email: 4
    - Webhook: 1
  • Complexity Metrics:
    - Avg Steps per Canvas: 6.4
    - Steps Range: 2 - 12
```

## Configuration

### Environment Variables
- `BRAZE_REST_KEY` - Your Braze REST API key (required)

### Forecasting Settings
- `BRAZE_ENDPOINT` - Braze cluster endpoint (default: iad-02)
- `--quiet-threshold` - Daily sends below this are considered "quiet" (default: 5)
- `--filter-prefix` - Only analyze canvases starting with this prefix
- `--workers` - Number of parallel workers for processing (default: 4)

## Data Models

The tool uses structured DTOs to represent canvas data:

### Canvas List Response
```python
@dataclass
class Canvas:
    id: str
    name: str
    tags: List[str]
    last_edited: str
```

### Canvas Details Response
```python
@dataclass
class CanvasDetails:
    canvas_id: str
    name: str
    created_at: str
    updated_at: str
    enabled: bool
    archived: bool
    draft: bool
    schedule_type: str
    channels: List[str]
    variants: List[CanvasVariant]
    steps: List[CanvasStep]
    # ... additional fields
```

## API Endpoints Used

**Canvas Analysis:**
- `GET /canvas/list` - Retrieve list of all canvases
- `GET /canvas/details` - Get detailed canvas configuration

**Time-Series Forecasting:**
- `GET /canvas/list` - Retrieve all Canvas IDs for data ingestion
- `GET /canvas/data_series` - Get daily Canvas statistics (entries, sends, delivered, opens, conversions)

## Development

### Available Make Commands

**Setup & Development:**
```bash
make setup      # Complete setup (venv + dependencies)
make run        # Run enhanced forecasting (default)
make test       # Run unit tests
make test-cov   # Run tests with coverage report
make lint       # Run flake8 linting
make fmt        # Format code with black
make clean      # Clean up virtual environment
```

**Time-Series Forecasting:**
```bash
make forecast                    # Generate basic forecasts from existing data
make forecast-enhanced           # Generate advanced forecasts with multiple models
make forecast-transactional      # Forecast only transactional canvases
make forecast-filter PREFIX=name # Forecast canvases starting with prefix
make ingest-historical           # Ingest 3 days of historical data
make ingest-historical-extended  # Ingest 30 days for better predictions
```

### Architecture

The tool follows a modular architecture with clear separation of concerns:

```
src/
├── models/          # Data models and DTOs
│   ├── canvas.py    # Canvas-related models
│   └── request_log.py # Request logging model
├── api/             # API client layer
│   └── client.py    # Braze API client
├── storage/         # Data persistence layer
│   └── data_storage.py # JSON file operations
├── analytics/       # Analytics and reporting
│   └── statistics.py # Statistics engine
├── forecasting/     # Time-series forecasting (NEW)
│   └── linear_decay.py # Regression models for quiet date prediction
├── ingest_historical.py # Historical data ingestion script
├── forecast_quiet_dates.py # Basic forecasting pipeline
└── forecast_enhanced.py # Enhanced forecasting with advanced models

tests/               # Comprehensive unit tests
├── test_models.py   # Model tests
├── test_api.py      # API client tests
├── test_storage.py  # Storage tests
├── test_analytics.py # Analytics tests
└── test_forecasting.py # Forecasting tests (NEW)
```

**Key Components:**
- **Models Package** - Type-safe data models with validation
- **API Package** - HTTP client with comprehensive logging
- **Storage Package** - JSON persistence with error handling
- **Analytics Package** - Statistics and reporting engine
- **Forecasting Package** - Time-series analysis with linear/exponential decay models
- **Comprehensive Test Suite** - 80+ tests with extensive coverage including forecasting

## Limitations

- Canvas details are fetched for a sample (default: 5) to avoid rate limits
- Designed for the iad-02 Braze cluster (configurable)
- Requires read-only canvas permissions

## Troubleshooting

**API Authentication Errors:**
- Verify your `BRAZE_REST_KEY` is valid and has canvas read permissions
- Check that you're using the correct Braze cluster endpoint

**Rate Limiting:**
- The tool fetches canvas details for only 5 canvases by default
- Increase delays between requests if needed

**Missing Data:**
- Some canvas fields may be empty if not configured in Braze
- Check the application logs (`braze_export.log`) for detailed error information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `make lint` and ensure code passes
5. Submit a pull request

## License

[Add your license information here]