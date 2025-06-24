"""Historical Canvas data ingestion script.

This script pulls historical Canvas statistics from the Braze API for a specified
date range and stores them in append-only JSONL files, one per Canvas ID.

Usage:
    # Ingest last 60 days
    BRAZE_REST_KEY=your-key python src/ingest_historical.py --days 60

    # Ingest specific date range
    BRAZE_REST_KEY=your-key python src/ingest_historical.py --start-date 2023-11-01 --end-date 2023-12-31
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

# Load environment variables
load_dotenv()

# Configuration
BRAZE_ENDPOINT: Final[str] = "https://rest.iad-02.braze.com"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('braze_historical_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_api_config():
    """Get API configuration, checking for required environment variables."""
    api_key = os.environ.get("BRAZE_REST_KEY")
    if not api_key:
        raise ValueError("BRAZE_REST_KEY environment variable is required")

    return {
        "api_key": api_key,
        "headers": {"Authorization": f"Bearer {api_key}"}
    }


def get_canvas_ids() -> List[str]:
    """Fetch all Canvas IDs from the Braze API with pagination support."""
    api_config = get_api_config()
    headers = api_config["headers"]

    ids = []
    url = f"{BRAZE_ENDPOINT}/canvas/list"
    params = {"limit": 100}

    logger.info("Fetching Canvas IDs...")

    while True:
        logger.debug(f"Requesting: {url} with params: {params}")
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        canvas_batch = data.get("canvases", [])
        ids.extend(c["id"] for c in canvas_batch)
        logger.debug(f"Retrieved {len(canvas_batch)} Canvas IDs (total: {len(ids)})")

        # Check for pagination
        next_page = data.get("next_page")
        if next_page is None:
            break
        url = f"{BRAZE_ENDPOINT}{next_page}"
        params = {}  # Clear params for paginated requests

    logger.info(f"Found {len(ids)} total Canvas IDs")
    return ids


def get_canvas_name_mapping() -> Dict[str, str]:
    """Get a mapping of Canvas ID to Canvas name for better logging."""
    api_config = get_api_config()
    headers = api_config["headers"]

    name_map = {}
    url = f"{BRAZE_ENDPOINT}/canvas/list"
    params = {"limit": 100}

    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for canvas in data.get("canvases", []):
            name_map[canvas["id"]] = canvas["name"]

        next_page = data.get("next_page")
        if next_page is None:
            break
        url = f"{BRAZE_ENDPOINT}{next_page}"
        params = {}

    return name_map


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3))
def get_canvas_data_chunk(canvas_id: str, chunk_start: str, chunk_end: str) -> List[Dict[str, Any]]:
    """
    Fetch a chunk of Canvas data series (≤13 days due to API limits).

    Args:
        canvas_id: Canvas ID
        chunk_start: Start date in YYYY-MM-DD format
        chunk_end: End date in YYYY-MM-DD format

    Returns:
        List of daily statistics dictionaries
    """
    api_config = get_api_config()
    headers = api_config["headers"]

    # Calculate the number of days to fetch
    start_dt = dt.datetime.strptime(chunk_start, '%Y-%m-%d').date()
    end_dt = dt.datetime.strptime(chunk_end, '%Y-%m-%d').date()
    length = (end_dt - start_dt).days + 1

    if length > 13:
        raise ValueError(f"Chunk too large: {length} days. Braze API limit is 13 days.")

    logger.debug(f"Fetching {length} days of data for Canvas {canvas_id} from {chunk_start} to {chunk_end}")

    try:
        resp = requests.get(
            f"{BRAZE_ENDPOINT}/canvas/data_series",
            headers=headers,
            params={
                "canvas_id": canvas_id,
                "length": length,
                "ending_at": chunk_end
            },
            timeout=30,
        )
        logger.debug(f"Response status: {resp.status_code} for Canvas {canvas_id}")
        resp.raise_for_status()
    except requests.HTTPError as e:
        error_msg = "Unknown error"
        status_code = "Unknown"
        try:
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                error_msg = e.response.text
        except AttributeError:
            pass
        logger.error(f"HTTP Error for Canvas {canvas_id}: {status_code} - {error_msg}")
        raise
    except requests.RequestException as e:
        logger.error(f"Request Error for Canvas {canvas_id}: {e}")
        raise

    api_data = resp.json()

    # Handle empty or malformed responses
    if "data" not in api_data or "stats" not in api_data["data"] or not api_data["data"]["stats"]:
        logger.debug(f"No data returned for Canvas {canvas_id} chunk {chunk_start} to {chunk_end}")
        return []

    # Process and normalize each day's data
    normalized_data = []
    stats_array = api_data["data"]["stats"]

    # The API returns data with explicit dates in the "time" field
    for stat_item in stats_array:
        time_str = stat_item.get("time", "")
        total_stats = stat_item.get("total_stats", {})

        # Parse the date from the time field
        try:
            stat_date = dt.datetime.strptime(time_str, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"Invalid date format in API response: {time_str}")
            continue

        normalized_record = {
            "date": stat_date.isoformat(),
            "entries": total_stats.get("entries", 0),
            "sends": total_stats.get("sends", 0),
            "delivered": total_stats.get("delivered", 0),
            "opens": total_stats.get("opens", 0),
            "conversions": total_stats.get("conversions", 0),
            "revenue": total_stats.get("revenue", 0.0)
        }

        normalized_data.append(normalized_record)

    logger.debug(f"Processed {len(normalized_data)} days of data for Canvas {canvas_id}")
    return normalized_data


def get_canvas_data_series(canvas_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch historical data series for a Canvas ID, breaking large ranges into chunks.

    Args:
        canvas_id: Canvas ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of daily statistics dictionaries
    """
    start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    total_days = (end_dt - start_dt).days + 1

    logger.debug(f"Fetching {total_days} days of data for Canvas {canvas_id}, breaking into chunks")

    all_data = []
    chunk_size = 13  # Maximum allowed by Braze API

    current_start = start_dt

    while current_start <= end_dt:
        # Calculate chunk end date (don't exceed the overall end date)
        chunk_end = min(current_start + dt.timedelta(days=chunk_size - 1), end_dt)

        chunk_start_str = current_start.isoformat()
        chunk_end_str = chunk_end.isoformat()

        try:
            chunk_data = get_canvas_data_chunk(canvas_id, chunk_start_str, chunk_end_str)
            all_data.extend(chunk_data)
            logger.debug(f"✓ Successfully fetched chunk {chunk_start_str} to {chunk_end_str} for Canvas {canvas_id}: {len(chunk_data)} records")

            # Small delay between chunks to be gentle on the API
            time.sleep(0.1)

        except Exception as e:
            logger.warning(f"✗ Failed to fetch chunk {chunk_start_str} to {chunk_end_str} for Canvas {canvas_id}: {type(e).__name__}: {e}")

        # Move to next chunk
        current_start = chunk_end + dt.timedelta(days=1)

    # Sort by date (oldest first) for consistent order
    all_data.sort(key=lambda x: x["date"])

    logger.debug(f"Fetched total of {len(all_data)} days of data for Canvas {canvas_id}")
    return all_data


def append_canvas_data(canvas_id: str, daily_records: List[Dict[str, Any]]) -> int:
    """
    Append historical records to a Canvas JSONL file, avoiding duplicates.

    Args:
        canvas_id: Canvas ID
        daily_records: List of daily statistics dictionaries

    Returns:
        Number of new records added
    """
    jsonl_path = DATA_DIR / f"{canvas_id}.jsonl"

    # Read existing dates to avoid duplicates
    existing_dates = set()
    if jsonl_path.exists():
        with jsonl_path.open('r') as f:
            for line in f:
                if line.strip():
                    try:
                        existing_record = json.loads(line)
                        existing_dates.add(existing_record.get("date"))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line in {jsonl_path}")
                        continue

    # Filter out records that already exist
    new_records = [record for record in daily_records if record["date"] not in existing_dates]

    if not new_records:
        logger.debug(f"No new records to add for Canvas {canvas_id}")
        return 0

    # Append new records
    with jsonl_path.open('a') as f:
        for record in new_records:
            f.write(json.dumps(record) + '\n')

    logger.debug(f"Added {len(new_records)} new records for Canvas {canvas_id}")
    return len(new_records)


def ingest_historical_data(start_date: str, end_date: str) -> None:
    """
    Main function to ingest historical data for all Canvas IDs.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    logger.info(f"Starting historical Canvas data ingestion from {start_date} to {end_date}")

    try:
        # Get Canvas IDs and names
        canvas_ids = get_canvas_ids()
        logger.info(f"Processing {len(canvas_ids)} Canvas IDs")

        # Get name mapping for better reporting
        name_mapping = get_canvas_name_mapping()

        success_count = 0
        partial_success_count = 0
        error_count = 0
        total_records_added = 0

        for i, canvas_id in enumerate(canvas_ids, 1):
            canvas_name = name_mapping.get(canvas_id, "Unknown")

            try:
                # Fetch historical data for this Canvas
                daily_records = get_canvas_data_series(canvas_id, start_date, end_date)

                if daily_records:
                    # Append to JSONL file
                    records_added = append_canvas_data(canvas_id, daily_records)
                    total_records_added += records_added

                    # Calculate expected vs actual records to detect partial failures
                    start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
                    end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
                    expected_days = (end_dt - start_dt).days + 1

                    if len(daily_records) == expected_days:
                        success_count += 1
                        logger.info(f"✓ ({i}/{len(canvas_ids)}) {canvas_name[:30]:30} | {records_added:3} new records | {len(daily_records):3}/{expected_days} days (complete)")
                    else:
                        partial_success_count += 1
                        logger.warning(f"◐ ({i}/{len(canvas_ids)}) {canvas_name[:30]:30} | {records_added:3} new records | {len(daily_records):3}/{expected_days} days (partial)")
                else:
                    error_count += 1
                    logger.warning(f"◯ ({i}/{len(canvas_ids)}) {canvas_name[:30]:30} | No data available")

            except Exception as exc:
                error_count += 1
                logger.error(f"⚠ ({i}/{len(canvas_ids)}) {canvas_name[:30]:30} | Unexpected error: {type(exc).__name__}: {exc}")

            finally:
                # Rate limiting and progress updates
                time.sleep(0.3)  # Small delay between Canvas requests

                # Progress update every 10 canvases
                if i % 10 == 0:
                    total_processed = success_count + partial_success_count + error_count
                    logger.info(f"Progress: {i}/{len(canvas_ids)} processed | ✓ {success_count} complete | ◐ {partial_success_count} partial | ✗ {error_count} failed")

        # Summary
        logger.info(f"Historical ingestion complete:")
        logger.info(f"  • Complete success: {success_count}")
        logger.info(f"  • Partial success: {partial_success_count}")
        logger.info(f"  • Failed: {error_count}")
        logger.info(f"  • Total new records added: {total_records_added}")
        logger.info(f"  • Date range: {start_date} to {end_date}")

    except Exception as exc:
        logger.error(f"Fatal error during historical ingestion: {exc}")
        raise


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Historical Canvas Data Ingestion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest last 60 days
  BRAZE_REST_KEY=your-key python src/ingest_historical.py --days 60

  # Ingest specific date range
  BRAZE_REST_KEY=your-key python src/ingest_historical.py --start-date 2023-11-01 --end-date 2023-12-31
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Number of days back from today to ingest (e.g., 60 for last 60 days)'
    )

    parser.add_argument(
        '--start-date',
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        help='End date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.days and (args.start_date or args.end_date):
        logger.error("Cannot specify both --days and --start-date/--end-date")
        sys.exit(1)

    if not args.days and not (args.start_date and args.end_date):
        logger.error("Must specify either --days or both --start-date and --end-date")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("BRAZE_REST_KEY"):
        logger.error("BRAZE_REST_KEY environment variable is required")
        sys.exit(1)

    # Calculate date range
    if args.days:
        end_date = dt.date.today() - dt.timedelta(days=1)  # Yesterday
        start_date = end_date - dt.timedelta(days=args.days - 1)
        start_date_str = start_date.isoformat()
        end_date_str = end_date.isoformat()
    else:
        start_date_str = args.start_date
        end_date_str = args.end_date

        # Validate date formats
        try:
            dt.datetime.strptime(start_date_str, '%Y-%m-%d')
            dt.datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            sys.exit(1)

    try:
        ingest_historical_data(start_date_str, end_date_str)
        logger.info("✅ Historical ingestion completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()