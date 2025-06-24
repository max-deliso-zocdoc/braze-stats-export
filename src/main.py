#!/usr/bin/env python
"""Braze API Statistics Exporter.

This script hits the Braze REST API at the iad‑02 cluster, extracts canvas
information, and provides detailed statistics. It expects an environment
variable `BRAZE_REST_KEY` containing a valid Braze REST key.

Run with:
    BRAZE_REST_KEY=<<your‑key>> python src/main.py
"""

import os
import sys
import logging

import requests

from .api import BrazeAPIClient
from .storage import DataStorage
from .analytics import CanvasStatistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('braze_export.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry-point: fetch canvas data, analyze, and export."""

    api_key = os.getenv("BRAZE_REST_KEY")
    if not api_key:
        sys.exit("BRAZE_REST_KEY environment variable is not set.")

    client = BrazeAPIClient(api_key)
    storage = DataStorage()

    try:
        logger.info("Starting Braze canvas export...")

        # Fetch canvas list
        logger.info("Fetching canvas list...")
        canvas_list = client.get_canvas_list(limit=100)
        storage.save_canvas_list(canvas_list)

        # Generate and display statistics
        stats = CanvasStatistics.analyze_canvas_list(canvas_list)
        print("\n" + "="*50)
        print("CANVAS STATISTICS")
        print("="*50)
        print(f"Total Canvases: {stats['total_canvases']}")
        print(f"Total Unique Tags: {stats['total_tags']}")
        print(f"Canvases Updated (Last 30 days): {stats['recent_activity_count']}")
        print(f"Canvases Without Tags: {stats['canvases_without_tags']}")

        print("\nMost Common Tags:")
        for tag, count in stats['most_common_tags']:
            print(f"  {tag}: {count}")

        # Fetch details for a few sample canvases (to avoid hitting rate limits)
        sample_canvases = canvas_list.canvases[:5]  # First 5 canvases
        logger.info(f"Fetching details for {len(sample_canvases)} sample canvases...")

        canvas_details = []
        for canvas in sample_canvases:
            try:
                logger.info(f"Fetching details for canvas: {canvas.name}")
                details = client.get_canvas_details(canvas.id)
                canvas_details.append(details)
            except requests.HTTPError as err:
                logger.error(f"Failed to fetch details for canvas {canvas.name}: {err}")
                continue

        if canvas_details:
            storage.save_canvas_details(canvas_details)

        # Save request log
        client.save_request_log()

        # Generate and display detailed statistics
        details_stats = {"error": "No canvas details available"}
        if canvas_details:
            details_stats = CanvasStatistics.analyze_canvas_details(canvas_details)
            print("\n" + "="*50)
            print("CANVAS DETAILS STATISTICS")
            print("="*50)
            for key, value in details_stats.items():
                if isinstance(value, dict):
                    print(f"\n{key.replace('_', ' ').title()}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey.replace('_', ' ').title()}: {subvalue}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")

        # Generate and display summary report
        summary_report = CanvasStatistics.generate_summary_report(
            stats,
            details_stats,
            client.request_log
        )
        print("\n" + summary_report)

        print(f"\n✅ Export completed successfully!")
        print(f"📊 Canvas list saved to: canvas_list.json")
        if canvas_details:
            print(f"📋 Canvas details saved to: canvas_details.json ({len(canvas_details)} detailed)")
        print(f"📝 Request log saved to: request_log.json")

    except requests.HTTPError as err:
        logger.error(f"Braze API error: {err}")
        if hasattr(err, 'response') and err.response:
            logger.error(f"Response: {err.response.text}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
