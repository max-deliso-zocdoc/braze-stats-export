"""Enhanced Canvas Quiet Date Forecasting Tool.

This script uses advanced forecasting models including polynomial regression,
power law models, ARIMA, and ensemble methods to provide better confidence
estimation for Canvas quiet date predictions.

Usage:
    # Generate enhanced forecasts with all models
    python src/forecast_enhanced.py

    # Use with specific quiet threshold
    python src/forecast_enhanced.py --quiet-threshold 10

    # Save detailed report
    python src/forecast_enhanced.py --output enhanced_forecast.json
"""

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from .forecasting.enhanced_predictor import EnhancedQuietDatePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("braze_enhanced_forecast.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def save_enhanced_report(report: dict, filename: str = None) -> None:
    """Save enhanced forecast report to JSON file."""
    if filename is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_forecast_report_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Enhanced forecast report saved to {filename}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Canvas Quiet Date Forecasting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate enhanced forecasts with all models
  python src/forecast_enhanced.py

  # Use with specific quiet threshold
  python src/forecast_enhanced.py --quiet-threshold 10

  # Save detailed report
  python src/forecast_enhanced.py --output enhanced_forecast.json

  # Parallel processing with custom workers
  python src/forecast_enhanced.py --workers 8
        """,
    )

    parser.add_argument(
        "--quiet-threshold",
        type=int,
        default=5,
        help='Daily sends below this threshold are considered "quiet" (default: 5)',
    )

    parser.add_argument(
        "--output", help="Output filename for enhanced forecast report JSON"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing Canvas data (default: data)",
    )

    parser.add_argument(
        "--filter-prefix",
        type=str,
        help="Only analyze canvases whose names start with this prefix (case-insensitive)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Check data directory
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(
                f"Data directory {data_dir} does not exist. Run historical data ingestion first."
            )
            sys.exit(1)

        # Check for Canvas directories
        canvas_dirs = [
            d for d in data_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
        ]
        if not canvas_dirs:
            logger.error(
                "No Canvas directories found. Run historical data ingestion first."
            )
            sys.exit(1)

        logger.info(
            f"🚀 Starting enhanced quiet date forecasting on {len(canvas_dirs)} canvases..."
        )

        # Initialize enhanced predictor
        predictor = EnhancedQuietDatePredictor(
            data_dir=data_dir, quiet_threshold=args.quiet_threshold
        )

        # Apply canvas name filter if specified
        if args.filter_prefix:
            predictor.canvas_name_filter = args.filter_prefix.lower()

        # Generate enhanced forecast report
        report = predictor.generate_enhanced_forecast_report()

        # Save report
        save_enhanced_report(report, args.output)

        # Display enhanced summary
        predictor.print_enhanced_forecast_summary(report)

        # Print improvement summary
        print(f"\n🎯 CONFIDENCE IMPROVEMENTS:")
        model_perf = report["model_performance"]
        ensemble_avg = model_perf["average_confidence"]["ensemble_vs_single"][
            "ensemble_avg"
        ]
        single_avg = model_perf["average_confidence"]["ensemble_vs_single"][
            "single_avg"
        ]

        if ensemble_avg > 0 and single_avg > 0:
            improvement = ((ensemble_avg - single_avg) / single_avg) * 100
            print(
                f"  • Ensemble models show {improvement:.1f}% confidence improvement over single models"
            )

        high_conf_count = model_perf["confidence_distribution"]["high"]
        total_predictable = report["summary"]["predictable"]

        if total_predictable > 0:
            high_conf_rate = (high_conf_count / total_predictable) * 100
            print(
                f"  • {high_conf_rate:.1f}% of predictions have high confidence (≥70%)"
            )

        print(f"\n📈 RECOMMENDATIONS:")

        # Data recommendations
        error_count = report["summary"]["errors"]
        if error_count > 0:
            print(f"  • {error_count} canvases had errors - check data quality")

        # Model recommendations
        model_usage = model_perf["model_usage"]
        if model_usage.get("advanced", 0) > 0:
            print(
                f"  • Advanced models used for {model_usage.get('advanced', 0)} canvases"
            )

        # Confidence recommendations
        low_conf_count = model_perf["confidence_distribution"]["low"]
        if low_conf_count > 0:
            print(
                f"  • {low_conf_count} predictions have low confidence - consider collecting more data"
            )

        logger.info("✅ Enhanced forecasting completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
