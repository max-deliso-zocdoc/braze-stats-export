"""Canvas Quiet Date Forecasting Tool.

This script analyzes existing Canvas data using regression analysis to predict
when Canvas sends will decay to approximately zero (the "quiet date").

Note: Daily ingestion functionality has been removed. Use ingest_historical.py
for data collection.

Usage:
    # Generate forecasts from existing data
    python src/forecast_quiet_dates.py

    # Create sample data for testing
    python src/forecast_quiet_dates.py --create-sample-data
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Final, List, Dict, Any
import time

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential


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


def save_forecast_report(report: dict, filename: str = None) -> None:
    """Save forecast report to JSON file."""
    if filename is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_report_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Forecast report saved to {filename}")


def print_forecast_summary(report: dict) -> None:
    """Print a human-readable summary of the forecast report."""
    summary = report["summary"]
    trends = report["trends"]
    confidence = report["confidence_distribution"]
    all_canvases = report.get("all_canvases", [])

    print("\n" + "=" * 70)
    print("🔮 CANVAS QUIET DATE FORECAST REPORT")
    print("=" * 70)

    print(f"\n📊 OVERVIEW:")
    print(f"  • Total Canvases Analyzed: {summary['total_canvases']}")
    print(f"  • Predictable Quiet Dates: {summary['predictable']}")
    print(f"  • Unpredictable: {summary['unpredictable']}")
    print(f"  • Going Quiet Soon (≤30 days): {summary['going_quiet_soon']}")
    print(f"  • Going Quiet Later (>30 days): {summary['going_quiet_later']}")

    print(f"\n📈 CURRENT TRENDS:")
    for trend, count in trends.items():
        trend_emoji = {
            "declining": "📉",
            "stable": "📊",
            "growing": "📈",
            "insufficient_data": "❓",
        }.get(trend, "❔")
        print(f"  {trend_emoji} {trend.replace('_', ' ').title()}: {count}")

    print(f"\n🎯 PREDICTION CONFIDENCE:")
    print(f"  • High (≥70%): {confidence['high']}")
    print(f"  • Medium (40-70%): {confidence['medium']}")
    print(f"  • Low (<40%): {confidence['low']}")

    # Filter canvases with predicted quiet dates and sort by quiet date (ascending)
    predictable_canvases = [
        canvas for canvas in all_canvases if canvas.get("quiet_date")
    ]
    predictable_canvases.sort(key=lambda x: x.get("quiet_date", "9999-12-31"))

    if predictable_canvases:
        print(f"\n📅 PREDICTED QUIET DATES (Sorted by Quiet Date)")
        print(
            "   Canvas Name                                                  Quiet Date     Days  Confidence  Trend"
        )
        print("   " + "-" * 105)
        for canvas in predictable_canvases:
            canvas_name = canvas.get("canvas_name", canvas["canvas_id"])[:60]
            quiet_date = (
                canvas["quiet_date"][:10] if canvas["quiet_date"] else "Unknown"
            )
            days = str(canvas["days_to_quiet"]) if canvas["days_to_quiet"] else "N/A"
            confidence = f"{canvas['confidence']:.1%}"
            trend = canvas["trend"][:10]

            print(
                f"   {canvas_name:60} {quiet_date:10}  {days:>4}  {confidence:9}  {trend:10}"
            )

    print("\n" + "=" * 70)


def create_sample_data() -> None:
    """Create sample time-series data for testing purposes."""
    from datetime import date, timedelta
    import random

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    logger.info("Creating sample data for testing...")

    # Create sample data for 3 Canvas IDs
    sample_canvases = [
        {"id": "sample-declining-canvas", "initial_sends": 1000, "trend": "declining"},
        {"id": "sample-stable-canvas", "initial_sends": 500, "trend": "stable"},
        {"id": "sample-growing-canvas", "initial_sends": 200, "trend": "growing"},
    ]

    base_date = date.today() - timedelta(days=30)

    for canvas in sample_canvases:
        canvas_id = canvas["id"]
        initial_sends = canvas["initial_sends"]
        trend = canvas["trend"]

        jsonl_path = data_dir / f"{canvas_id}.jsonl"

        with jsonl_path.open("w") as f:
            for i in range(30):  # 30 days of data
                current_date = base_date + timedelta(days=i)

                if trend == "declining":
                    sends = max(
                        0, int(initial_sends * (1 - i * 0.05) + random.randint(-50, 50))
                    )
                elif trend == "stable":
                    sends = max(0, int(initial_sends + random.randint(-100, 100)))
                else:  # growing
                    sends = max(
                        0, int(initial_sends * (1 + i * 0.02) + random.randint(-30, 30))
                    )

                entries = int(sends * 1.05 + random.randint(0, 20))
                delivered = int(sends * 0.95 + random.randint(-10, 10))
                opens = int(sends * 0.15 + random.randint(-5, 5))
                conversions = int(sends * 0.02 + random.randint(0, 2))

                record = {
                    "date": current_date.isoformat(),
                    "entries": max(0, entries),
                    "sends": max(0, sends),
                    "delivered": max(0, delivered),
                    "opens": max(0, opens),
                    "conversions": max(0, conversions),
                }

                f.write(json.dumps(record) + "\n")

        logger.info(f"Created sample data for {canvas_id} ({trend} trend)")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Canvas Quiet Date Forecasting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate forecasts from existing data
  python src/forecast_quiet_dates.py

  # Create sample data for testing
  python src/forecast_quiet_dates.py --create-sample-data

  # Forecast only transactional canvases
  python src/forecast_quiet_dates.py --filter-prefix "transactional"

  # Forecast canvases starting with specific prefix
  python src/forecast_quiet_dates.py --filter-prefix "welcome" --verbose
        """,
    )

    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="(Deprecated - daily ingestion removed) Run forecasting on existing data",
    )

    parser.add_argument(
        "--quiet-threshold",
        type=int,
        default=5,
        help='Daily sends below this threshold are considered "quiet" (default: 5)',
    )

    parser.add_argument("--output", help="Output filename for forecast report JSON")

    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample time-series data for testing",
    )

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
        # Create sample data if requested
        if args.create_sample_data:
            create_sample_data()
            print("✅ Sample data created successfully!")
            return

        # Forecasting (daily ingestion has been removed)
        logger.info("🔮 Starting quiet date forecasting...")

        data_dir = Path("data")
        if not data_dir.exists():
            logger.error(
                "Data directory does not exist. Run data ingestion first or use --create-sample-data"
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

        logger.info("✅ Forecasting completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
