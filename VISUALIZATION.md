# Canvas Forecast Visualization

This module provides comprehensive visualization tools for Canvas quiet date forecasting, based on the pandas vectorized data processing pipeline.

## Quick Start

### Using Makefile Targets

```bash
# Create overview plot for first 9 canvases (displays plot)
make visualization

# Create overview plot and save to plots/ directory
make visualization-save
```

### Using Command Line

```bash
# Create overview plot for first 9 canvases
python -m src.visualization.main --overview --max-canvases 9

# Create individual plots for all canvases and save to plots/
python -m src.visualization.main --output plots/

# Create plot for specific canvas
python -m src.visualization.main --canvas-id your-canvas-id

# Create plots with different metric
python -m src.visualization.main --metric total_opens --output plots/
```

## Features

- **Individual Canvas Plots**: Detailed forecast plots with confidence intervals
- **Multi-Canvas Overview**: Subplot grid showing multiple canvases
- **Flexible Metrics**: Analyze any metric (total_sent, total_opens, total_clicks, etc.)
- **Statistical Information**: R² values, slopes, and confidence intervals
- **Export Capabilities**: Save plots as high-resolution PNG files
- **Command Line Interface**: Full CLI with argument parsing

## Module Structure

```
src/visualization/
├── __init__.py              # Module exports
├── main.py                  # Command line interface
└── canvas_forecast.py       # Core visualization functions
```

## API Reference

### Command Line Options

- `--data-dir`: Directory containing canvas data (default: data)
- `--output`: Output directory for saving plots
- `--canvas-id`: Specific canvas ID to visualize
- `--metric`: Metric to analyze (default: total_sent)
- `--quiet-threshold`: Daily sends below this are considered "quiet" (default: 5)
- `--horizon-days`: Number of days to predict into the future (default: 180)
- `--max-canvases`: Maximum number of canvases to process
- `--overview`: Create multi-canvas overview plot
- `--no-display`: Don't display plots (only save if --output is specified)
- `--verbose`: Enable verbose logging

### Available Metrics

- `total_sent`: Total messages sent
- `total_opens`: Total opens
- `total_unique_opens`: Total unique opens
- `total_clicks`: Total clicks
- `total_unique_clicks`: Total unique clicks
- `total_delivered`: Total delivered
- `total_bounces`: Total bounces
- `total_unsubscribes`: Total unsubscribes
- `active_steps`: Number of active steps
- `active_channels`: Number of active channels

## Programmatic Usage

### Basic Usage

```python
from pathlib import Path
from src.forecasting.linear_decay import StepBasedForecaster
from src.visualization import plot_canvas_forecast

# Load canvas data
forecaster = StepBasedForecaster(quiet_threshold=5)
metrics = forecaster.load_canvas_metrics("your-canvas-id", Path("data"))

# Create forecast plot
quiet_date = plot_canvas_forecast(
    metrics=metrics,
    metric_col="total_sent",
    quiet_threshold=5,
    horizon_days=180
)
```

### Multiple Canvases

```python
from src.visualization import plot_multiple_canvases

# Load data for multiple canvases
canvas_data = {}
for canvas_id in ["canvas1", "canvas2", "canvas3"]:
    metrics = forecaster.load_canvas_metrics(canvas_id, Path("data"))
    if metrics:
        canvas_data[canvas_id] = metrics

# Create overview plot
plot_multiple_canvases(
    canvas_data=canvas_data,
    metric_col="total_sent",
    quiet_threshold=5,
    max_canvases=6
)
```

## Plot Elements

Each forecast plot includes:

1. **Actual Data Points**: Blue scatter plot of historical data
2. **Linear Regression**: Red line showing the fitted trend
3. **Confidence Interval**: Light red shaded area (95% CI on the mean)
4. **Quiet Threshold**: Orange dashed line showing the quiet threshold
5. **Predicted Quiet Date**: Green dotted vertical line (if applicable)
6. **Statistics**: Title shows R² value and slope

## Example Workflows

### 1. Quick Overview

```bash
# See overview of first 9 canvases
make visualization
```

### 2. Save All Plots

```bash
# Create plots directory and save all individual plots
python -m src.visualization.main --output plots/
```

### 3. Analyze Specific Canvas

```bash
# Focus on one canvas
python -m src.visualization.main --canvas-id your-canvas-id --metric total_opens
```

### 4. Batch Processing

```bash
# Process only first 5 canvases with different metric
python -m src.visualization.main --max-canvases 5 --metric total_clicks --output plots/
```

## Integration with Existing Code

The visualization module works seamlessly with your existing forecasting pipeline:

```python
# Use with existing forecast results
from src.forecast_quiet_dates import main as run_forecast

# Run forecasting
report = run_forecast()

# Create visualizations for predictable canvases
predictable_canvases = {
    result.canvas_id: forecaster.load_canvas_metrics(result.canvas_id, Path("data"))
    for result in report["all_canvases"]
    if result["quiet_date"] is not None
}

plot_multiple_canvases(predictable_canvases, metric_col="total_sent")
```

## Dependencies

The visualization module requires:
- `matplotlib~=3.7.0`
- `statsmodels~=0.14.0`
- `pandas~=2.0.0`
- `numpy~=1.24.0`

These are automatically installed when you run `pip install -r requirements.txt`.

## Performance Notes

- The visualization uses the same vectorized pandas pipeline as the forecasting
- Plots are generated efficiently using matplotlib
- Large numbers of canvases can be processed in batches
- Memory usage scales linearly with the number of data points

## Troubleshooting

**No plots displayed**: Ensure you have a display environment or set `--no-display` and use `--output`

**Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

**Empty plots**: Check that your canvas data contains sufficient historical data (at least 7 days recommended)

**Memory issues**: Reduce `--max-canvases` or process canvases in smaller batches

**Module not found**: Ensure you're running from the project root directory