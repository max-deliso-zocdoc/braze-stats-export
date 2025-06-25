"""Canvas Quiet Date Forecasting Tool.

This script analyzes existing Canvas data using regression analysis to predict
when Canvas sends will decay to approximately zero (the "quiet date").

This module is dedicated to forecasting only. For data ingestion, use ingest_historical.py.

Usage:
    # Generate forecasts from existing data
    python src/forecast_quiet_dates.py

    # Forecast only transactional canvases
    python src/forecast_quiet_dates.py --filter-prefix "transactional"
"""

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .forecasting.linear_decay import QuietDatePredictor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("braze_forecast.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def save_forecast_report(report: dict, filename: Optional[str] = None) -> None:
    """Save forecast report to JSON file."""
    if filename is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_report_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Forecast report saved to {filename}")


def print_forecast_summary(report: dict) -> None:
    """Print a summary of the forecast report to stdout."""
    summary = report["summary"]
    trends = report["trends"]

    print("\n" + "=" * 60)
    print("FORECAST SUMMARY")
    print("=" * 60)
    print(f"Total Canvases Analyzed: {summary['total_canvases']}")
    print(f"Predictable Canvases: {summary['predictable']}")
    print(f"Unpredictable Canvases: {summary['unpredictable']}")
    print(f"Prediction Rate: {summary['prediction_rate']:.1%}")

    if summary["predictable"] > 0:
        print(f"\nAverage Days to Quiet: {summary['avg_days_to_quiet']:.1f}")
        print(f"Average Confidence: {summary['avg_confidence']:.1%}")

    print("\nTrend Distribution:")
    for trend, count in trends.items():
        if count > 0:
            percentage = (count / summary["total_canvases"]) * 100
            print(f"  {trend.capitalize()}: {count} ({percentage:.1f}%)")

    if summary["predictable"] > 0:
        print("\nConfidence Distribution:")
        confidence_dist = report["confidence_distribution"]
        for confidence_range, count in confidence_dist.items():
            if count > 0:
                percentage = (count / summary["predictable"]) * 100
                print(f"  {confidence_range}: {count} ({percentage:.1f}%)")

    print("=" * 60)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Canvas Quiet Date Forecasting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate forecasts from existing data
  python src/forecast_quiet_dates.py

  # Forecast only transactional canvases
  python src/forecast_quiet_dates.py --filter-prefix "transactional"

  # Forecast canvases starting with specific prefix
  python src/forecast_quiet_dates.py --filter-prefix "welcome" --verbose
        """,
    )

    parser.add_argument(
        "--quiet-threshold",
        type=int,
        default=5,
        help='Daily sends below this threshold are considered "quiet" (default: 5)',
    )

    parser.add_argument("--output", help="Output filename for forecast report JSON")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--filter-prefix",
        type=str,
        help="Only analyze canvases whose names start with this prefix (case-insensitive)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Forecasting (this module is dedicated to forecasting only)
        logger.info("Starting quiet date forecasting...")

        data_dir = Path("data")
        if not data_dir.exists():
            logger.error(
                "Data directory does not exist. Run data ingestion first."
            )
            sys.exit(1)

        # Check for hierarchical Canvas directories (new format only)
        canvas_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

        if not canvas_dirs:
            logger.error(
                "No Canvas directories found. Run historical data ingestion first to create step-based data structure"
            )
            sys.exit(1)

        logger.info(f"Found {len(canvas_dirs)} Canvas directories with step-based data")

        # Run forecasting
        predictor = QuietDatePredictor(
            data_dir=data_dir, quiet_threshold=args.quiet_threshold
        )

        # Apply canvas name filter if specified
        if args.filter_prefix:
            predictor.canvas_name_filter = args.filter_prefix.lower()

        report = predictor.generate_forecast_report()

        # Save report
        save_forecast_report(report, args.output)

        # Display summary
        print_forecast_summary(report)

        logger.info("Forecasting completed successfully!")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
