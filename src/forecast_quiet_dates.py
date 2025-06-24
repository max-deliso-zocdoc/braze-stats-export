"""Canvas Quiet Date Forecasting Tool.

This script combines daily data ingestion and regression analysis to predict
when Canvas sends will decay to approximately zero (the "quiet date").

Usage:
    # Ingest data and generate forecasts
    BRAZE_REST_KEY=your-key python src/forecast_quiet_dates.py

    # Only run forecasting (skip ingestion)
    python src/forecast_quiet_dates.py --forecast-only

    # Ingest specific date
    BRAZE_REST_KEY=your-key python src/forecast_quiet_dates.py --date 2023-12-15
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from forecasting import QuietDatePredictor
from ingest_daily import main as ingest_main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('braze_forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def save_forecast_report(report: dict, filename: str = None) -> None:
    """Save forecast report to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"forecast_report_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Forecast report saved to {filename}")


def print_forecast_summary(report: dict) -> None:
    """Print a human-readable summary of the forecast report."""
    summary = report['summary']
    trends = report['trends']
    confidence = report['confidence_distribution']
    urgent = report['urgent_canvases']

    print("\n" + "="*70)
    print("🔮 CANVAS QUIET DATE FORECAST REPORT")
    print("="*70)

    print(f"\n📊 OVERVIEW:")
    print(f"  • Total Canvases Analyzed: {summary['total_canvases']}")
    print(f"  • Predictable Quiet Dates: {summary['predictable']}")
    print(f"  • Unpredictable: {summary['unpredictable']}")
    print(f"  • Going Quiet Soon (≤30 days): {summary['going_quiet_soon']}")
    print(f"  • Going Quiet Later (>30 days): {summary['going_quiet_later']}")

    print(f"\n📈 CURRENT TRENDS:")
    for trend, count in trends.items():
        trend_emoji = {
            'declining': '📉',
            'stable': '📊',
            'growing': '📈',
            'insufficient_data': '❓'
        }.get(trend, '❔')
        print(f"  {trend_emoji} {trend.replace('_', ' ').title()}: {count}")

    print(f"\n🎯 PREDICTION CONFIDENCE:")
    print(f"  • High (≥70%): {confidence['high']}")
    print(f"  • Medium (40-70%): {confidence['medium']}")
    print(f"  • Low (<40%): {confidence['low']}")

    if urgent:
        print(f"\n⚠️  URGENT: CANVASES GOING QUIET SOON")
        print("   Canvas ID" + " "*25 + "Quiet Date    Days  Confidence  Trend")
        print("   " + "-"*65)
        for canvas in urgent:
            canvas_id = canvas['canvas_id'][:35]
            quiet_date = canvas['quiet_date'][:10] if canvas['quiet_date'] else 'Unknown'
            days = str(canvas['days_to_quiet']) if canvas['days_to_quiet'] else 'N/A'
            confidence = f"{canvas['confidence']:.1%}"
            trend = canvas['trend'][:10]

            print(f"   {canvas_id:35} {quiet_date:10} {days:4}  {confidence:9}  {trend}")

    print("\n" + "="*70)


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
        {"id": "sample-growing-canvas", "initial_sends": 200, "trend": "growing"}
    ]

    base_date = date.today() - timedelta(days=30)

    for canvas in sample_canvases:
        canvas_id = canvas["id"]
        initial_sends = canvas["initial_sends"]
        trend = canvas["trend"]

        jsonl_path = data_dir / f"{canvas_id}.jsonl"

        with jsonl_path.open('w') as f:
            for i in range(30):  # 30 days of data
                current_date = base_date + timedelta(days=i)

                if trend == "declining":
                    sends = max(0, int(initial_sends * (1 - i * 0.05) + random.randint(-50, 50)))
                elif trend == "stable":
                    sends = max(0, int(initial_sends + random.randint(-100, 100)))
                else:  # growing
                    sends = max(0, int(initial_sends * (1 + i * 0.02) + random.randint(-30, 30)))

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
                    "conversions": max(0, conversions)
                }

                f.write(json.dumps(record) + '\n')

        logger.info(f"Created sample data for {canvas_id} ({trend} trend)")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Canvas Quiet Date Forecasting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: ingest data and forecast
  BRAZE_REST_KEY=your-key python src/forecast_quiet_dates.py

  # Only run forecasting
  python src/forecast_quiet_dates.py --forecast-only

  # Ingest specific date
  BRAZE_REST_KEY=your-key python src/forecast_quiet_dates.py --date 2023-12-15

  # Create sample data for testing
  python src/forecast_quiet_dates.py --create-sample-data
        """
    )

    parser.add_argument(
        '--forecast-only',
        action='store_true',
        help='Skip data ingestion and only run forecasting'
    )

    parser.add_argument(
        '--date',
        help='Specific date to ingest (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--quiet-threshold',
        type=int,
        default=5,
        help='Daily sends below this threshold are considered "quiet" (default: 5)'
    )

    parser.add_argument(
        '--output',
        help='Output filename for forecast report JSON'
    )

    parser.add_argument(
        '--create-sample-data',
        action='store_true',
        help='Create sample time-series data for testing'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
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

        # Step 1: Data Ingestion (unless skipped)
        if not args.forecast_only:
            if not os.environ.get("BRAZE_REST_KEY"):
                logger.error("BRAZE_REST_KEY environment variable is required for data ingestion")
                sys.exit(1)

            logger.info("🔄 Starting data ingestion...")
            try:
                ingest_main(args.date)
                logger.info("✅ Data ingestion completed")
            except Exception as e:
                logger.error(f"❌ Data ingestion failed: {e}")
                # Continue with forecasting using existing data
                logger.info("Continuing with forecasting using existing data...")

        # Step 2: Forecasting
        logger.info("🔮 Starting quiet date forecasting...")

        data_dir = Path("data")
        if not data_dir.exists() or not list(data_dir.glob("*.jsonl")):
            logger.error("No time-series data found. Run data ingestion first or use --create-sample-data")
            sys.exit(1)

        # Run forecasting
        predictor = QuietDatePredictor(
            data_dir=data_dir,
            quiet_threshold=args.quiet_threshold
        )

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