#!/usr/bin/env python3
"""Main script for Canvas forecast visualization."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from ..forecasting.linear_decay import StepBasedForecaster
from .canvas_forecast import (create_forecast_report_plots,
                              plot_canvas_forecast, plot_canvas_forecast_all_models, plot_multiple_canvases,
                              validate_canvas_data)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_canvas_name_mapping(data_dir: Path) -> dict:
    """Load the canvas name mapping from the index file if available."""
    index_path = data_dir / "canvas_index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            return json.load(f)
    return {}


def create_multi_model_forecast_plots(
    canvas_data: dict,
    output_dir: Path,
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    horizon_days: int = 180,
    name_map: Optional[dict] = None,
    show_plot: bool = False,
) -> None:
    """
    Create multi-model forecast plots for all canvases.

    Args:
        canvas_data: Dictionary mapping canvas_id to CanvasMetrics list
        output_dir: Directory to save plots
        metric_col: Metric column to analyze
        quiet_threshold: Daily sends below this are considered "quiet"
        horizon_days: Number of days to predict into the future
        name_map: Optional mapping of canvas_id to canvas_name
        show_plot: Whether to display plots
    """
    output_dir.mkdir(exist_ok=True)

    logging.info(f"Creating multi-model forecast plots for {len(canvas_data)} canvases...")

    successful_plots = 0
    failed_plots = 0

    for canvas_id, metrics in canvas_data.items():
        try:
            # Get canvas name for filename
            canvas_name = name_map.get(canvas_id, canvas_id) if name_map else canvas_id
            safe_name = "".join(c for c in canvas_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')

            # Create filename
            filename = f"multi_model_forecast_{canvas_id}_{safe_name}.png"
            save_path = output_dir / filename

            logging.info(f"Generating multi-model forecast for {canvas_name} ({canvas_id})")

            # Generate multi-model forecast plot
            multi_result = plot_canvas_forecast_all_models(
                metrics=metrics,
                metric_col=metric_col,
                quiet_threshold=quiet_threshold,
                horizon_days=horizon_days,
                save_path=save_path,
                show_plot=show_plot,
            )

            if multi_result and multi_result.forecasts:
                logging.info(f"  Generated {len(multi_result.forecasts)} forecasts")
                if multi_result.best_forecast and multi_result.best_forecast.quiet_date:
                    logging.info(
                        f"  Best forecast: {multi_result.best_forecast.quiet_date} "
                        f"({multi_result.best_forecast.model_type}, "
                        f"R²={multi_result.best_forecast.r_squared:.2f})"
                    )
                successful_plots += 1
            else:
                logging.warning(f"  No successful forecasts for {canvas_name}")
                failed_plots += 1

        except Exception as e:
            logging.error(f"Error generating multi-model forecast for {canvas_id}: {e}")
            failed_plots += 1

    logging.info("Multi-model forecast plots completed:")
    logging.info(f"  Successful: {successful_plots}")
    logging.info(f"  Failed: {failed_plots}")
    logging.info(f"  Total: {len(canvas_data)}")
    logging.info(f"  Plots saved to: {output_dir}")


def load_canvas_data(
    data_dir: Path,
    max_canvases: Optional[int] = None,
    metric_col: str = "total_sent",
    filter_prefix: Optional[str] = None,
) -> dict:
    """Load canvas data from the data directory, optionally filtering by name prefix."""
    forecaster = StepBasedForecaster(quiet_threshold=5)
    canvas_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    name_map = get_canvas_name_mapping(data_dir)

    if not canvas_dirs:
        logging.error("No canvas directories found in %s", data_dir)
        return {}

    # Filter by prefix if specified
    if filter_prefix:
        filter_prefix = filter_prefix.lower()
        filtered = []
        for d in canvas_dirs:
            canvas_id = d.name
            canvas_name = name_map.get(canvas_id, "")
            if canvas_name.lower().startswith(filter_prefix):
                filtered.append(d)
        canvas_dirs = filtered
        logging.info(
            "Filtered to %d canvases with prefix '%s'", len(canvas_dirs), filter_prefix
        )

    if max_canvases:
        canvas_dirs = canvas_dirs[:max_canvases]

    canvas_data = {}
    skipped_count = 0

    for canvas_dir in canvas_dirs:
        canvas_id = canvas_dir.name
        try:
            metrics = forecaster.load_canvas_metrics(canvas_id, data_dir)
            if metrics and validate_canvas_data(metrics, metric_col):
                canvas_data[canvas_id] = metrics
                logging.info("Loaded %d days of data for %s", len(metrics), canvas_id)
            else:
                skipped_count += 1
                logging.debug("Skipped %s: insufficient or invalid data", canvas_id)
        except Exception as e:
            skipped_count += 1
            logging.warning("Error loading %s: %s", canvas_id, e)

    if skipped_count > 0:
        logging.info(
            "Skipped %d canvases due to insufficient or invalid data", skipped_count
        )

    return canvas_data


def main() -> None:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Canvas Forecast Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create individual plots for all canvases (single model)
  python -m src.visualization.main --output plots/

  # Create multi-model forecast plots for all canvases
  python -m src.visualization.main --multi-model --output plots/

  # Create overview plot for first 9 canvases
  python -m src.visualization.main --overview --max-canvases 9

  # Create plot for specific canvas
  python -m src.visualization.main --canvas-id your-canvas-id

  # Create plots with different metric
  python -m src.visualization.main --metric total_opens --output plots/
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing canvas data (default: data)",
    )

    parser.add_argument("--output", type=Path, help="Output directory for saving plots")

    parser.add_argument("--canvas-id", type=str, help="Specific canvas ID to visualize")

    parser.add_argument(
        "--metric",
        type=str,
        default="total_sent",
        choices=[
            "total_sent",
            "total_opens",
            "total_unique_opens",
            "total_clicks",
            "total_unique_clicks",
            "total_delivered",
            "total_bounces",
            "total_unsubscribes",
            "active_steps",
            "active_channels",
        ],
        help="Metric to analyze (default: total_sent)",
    )

    parser.add_argument(
        "--quiet-threshold",
        type=int,
        default=5,
        help="Daily sends below this are considered 'quiet' (default: 5)",
    )

    parser.add_argument(
        "--horizon-days",
        type=int,
        default=180,
        help="Number of days to predict into the future (default: 180)",
    )

    parser.add_argument(
        "--max-canvases", type=int, help="Maximum number of canvases to process"
    )

    parser.add_argument(
        "--overview", action="store_true", help="Create multi-canvas overview plot"
    )

    parser.add_argument(
        "--multi-model", action="store_true", help="Use multi-model forecasting (shows all prediction vectors)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plots (only save if --output is specified)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--filter-prefix",
        type=str,
        help="Only analyze canvases whose names start with this prefix (case-insensitive)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check data directory
    if not args.data_dir.exists():
        logging.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    # Load canvas data
    if args.canvas_id:
        # Single canvas mode
        forecaster = StepBasedForecaster(quiet_threshold=args.quiet_threshold)
        try:
            metrics = forecaster.load_canvas_metrics(args.canvas_id, args.data_dir)
            if not metrics:
                logging.error("No data found for canvas: %s", args.canvas_id)
                sys.exit(1)

            canvas_data = {args.canvas_id: metrics}
            logging.info("Loaded %d days of data for %s", len(metrics), args.canvas_id)
        except Exception as e:
            logging.error("Error loading canvas %s: %s", args.canvas_id, e)
            sys.exit(1)
    else:
        # Multiple canvas mode
        canvas_data = load_canvas_data(
            args.data_dir, args.max_canvases, args.metric, args.filter_prefix
        )
        if not canvas_data:
            logging.error("No canvas data found")
            sys.exit(1)

    # Create output directory if needed
    if args.output:
        args.output.mkdir(exist_ok=True)
        logging.info("Output directory: %s", args.output)

    try:
        if args.overview:
            # Create overview plot
            logging.info("Creating multi-canvas overview plot...")
            save_path = args.output / "overview.png" if args.output else None
            plot_multiple_canvases(
                canvas_data=canvas_data,
                metric_col=args.metric,
                quiet_threshold=args.quiet_threshold,
                max_canvases=args.max_canvases or 9,
                save_path=save_path,
                show_plot=not args.no_display,
            )
        elif args.output and args.multi_model:
            # Create multi-model forecast plots for all canvases
            logging.info("Creating multi-model forecast plots...")
            name_map = get_canvas_name_mapping(args.data_dir)
            create_multi_model_forecast_plots(
                canvas_data=canvas_data,
                output_dir=args.output,
                metric_col=args.metric,
                quiet_threshold=args.quiet_threshold,
                horizon_days=args.horizon_days,
                name_map=name_map,
                show_plot=not args.no_display,
            )
        elif args.output:
            # Create individual plots for all canvases (single model)
            logging.info(
                "Creating individual plots for %d canvases...", len(canvas_data)
            )
            name_map = get_canvas_name_mapping(args.data_dir)
            create_forecast_report_plots(
                canvas_data=canvas_data,
                output_dir=args.output,
                metric_col=args.metric,
                quiet_threshold=args.quiet_threshold,
                name_map=name_map,
                show_plot=not args.no_display,
            )
        else:
            # Show individual plot for first canvas
            canvas_id = list(canvas_data.keys())[0]
            metrics = canvas_data[canvas_id]
            logging.info("Creating plot for %s", canvas_id)

            if args.multi_model:
                # Use multi-model forecasting
                multi_result = plot_canvas_forecast_all_models(
                    metrics=metrics,
                    metric_col=args.metric,
                    quiet_threshold=args.quiet_threshold,
                    horizon_days=args.horizon_days,
                    show_plot=not args.no_display,
                )

                if multi_result and multi_result.best_forecast:
                    quiet_date = multi_result.best_forecast.quiet_date
                    if quiet_date:
                        logging.info("Predicted quiet date: %s", quiet_date)
                    else:
                        logging.info("No quiet date predicted")
                else:
                    logging.info("No successful forecasts")
            else:
                # Use single model forecasting
                quiet_date = plot_canvas_forecast(
                    metrics=metrics,
                    metric_col=args.metric,
                    quiet_threshold=args.quiet_threshold,
                    horizon_days=args.horizon_days,
                    show_plot=not args.no_display,
                )

                if quiet_date:
                    logging.info("Predicted quiet date: %s", quiet_date)
                else:
                    logging.info("No quiet date predicted")

        logging.info("Visualization completed successfully!")

    except Exception as e:
        logging.error("Error during visualization: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
