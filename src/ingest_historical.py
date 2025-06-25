"""Historical Canvas data ingestion script.

This script pulls historical Canvas statistics from the Braze API for a specified
date range and stores them in JSONL files, one per Canvas ID. If data for the same
date already exists, it will be replaced with the new data.

This module is dedicated to data ingestion only. For forecasting, use forecast_quiet_dates.py.

Usage:
    # Ingest last 60 days
    BRAZE_REST_KEY=your-key python src/ingest_historical.py --days 60

    # Ingest specific date range
    BRAZE_REST_KEY=your-key python src/ingest_historical.py --start-date 2023-11-01 --end-date 2023-12-31

    # Ingest only canvases with names starting with "Campaign"
    BRAZE_REST_KEY=your-key python src/ingest_historical.py --days 30 --filter-prefix Campaign
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Final, List, Dict, Any, Optional
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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("braze_historical_ingest.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_api_config():
    """Get API configuration, checking for required environment variables."""
    api_key = os.environ.get("BRAZE_REST_KEY")
    if not api_key:
        raise ValueError("BRAZE_REST_KEY environment variable is required")

    return {"api_key": api_key, "headers": {"Authorization": f"Bearer {api_key}"}}


def get_canvas_ids(filter_prefix: Optional[str] = None) -> List[str]:
    """Fetch all Canvas IDs from the Braze API with pagination support."""
    api_config = get_api_config()
    headers = api_config["headers"]

    ids: List[str] = []
    page = 0
    limit = 100

    logger.info("Fetching Canvas IDs (excluding archived, newest first)...")
    if filter_prefix:
        logger.info(f"Filtering canvases by name prefix: '{filter_prefix}'")

    while True:
        params = {
            "page": page,
            "limit": limit,
            "include_archived": "false",  # Exclude archived canvases
            "sort_direction": "desc",  # Newest first
        }

        logger.debug(f"Requesting page {page} with params: {params}")
        resp = requests.get(
            f"{BRAZE_ENDPOINT}/canvas/list", headers=headers, params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        canvas_batch = data.get("canvases", [])

        # Filter by prefix if specified
        if filter_prefix:
            filtered_batch = [
                c
                for c in canvas_batch
                if c.get("name", "").lower().startswith(filter_prefix.lower())
            ]
            ids.extend(c["id"] for c in filtered_batch)
            logger.debug(
                f"Retrieved {len(filtered_batch)} Canvas IDs from page {page} (filtered from {len(canvas_batch)})"
            )
        else:
            ids.extend(c["id"] for c in canvas_batch)
            logger.debug(
                f"Retrieved {len(canvas_batch)} Canvas IDs from page {page} (total: {len(ids)})"
            )

        # Check if we got fewer canvases than the limit, indicating last page
        if len(canvas_batch) < limit:
            break

        page += 1

    logger.info(f"Found {len(ids)} total active Canvas IDs (newest first)")
    if filter_prefix:
        logger.info(f"After filtering by prefix '{filter_prefix}': {len(ids)} canvases")
    return ids


def get_canvas_name_mapping(filter_prefix: Optional[str] = None) -> Dict[str, str]:
    """Get a mapping of Canvas ID to Canvas name for better logging."""
    api_config = get_api_config()
    headers = api_config["headers"]

    name_map: Dict[str, str] = {}
    page = 0
    limit = 100

    while True:
        params = {
            "page": page,
            "limit": limit,
            "include_archived": "false",  # Exclude archived canvases
            "sort_direction": "desc",  # Newest first
        }

        resp = requests.get(
            f"{BRAZE_ENDPOINT}/canvas/list", headers=headers, params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        canvas_batch = data.get("canvases", [])

        # Filter by prefix if specified
        if filter_prefix:
            for canvas in canvas_batch:
                if canvas.get("name", "").lower().startswith(filter_prefix.lower()):
                    name_map[canvas["id"]] = canvas["name"]
        else:
            for canvas in canvas_batch:
                name_map[canvas["id"]] = canvas["name"]

        # Check if we got fewer canvases than the limit, indicating last page
        if len(canvas_batch) < limit:
            break

        page += 1

    return name_map


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3))
def get_canvas_data_chunk(
    canvas_id: str, chunk_start: str, chunk_end: str
) -> List[Dict[str, Any]]:
    """
    Fetch a chunk of Canvas data series (≤13 days due to API limits) with step breakdown.

    Args:
        canvas_id: Canvas ID
        chunk_start: Start date in YYYY-MM-DD format
        chunk_end: End date in YYYY-MM-DD format

    Returns:
        List of daily statistics dictionaries with step-level breakdown
    """
    api_config = get_api_config()
    headers = api_config["headers"]

    # Calculate the number of days to fetch
    start_dt = dt.datetime.strptime(chunk_start, "%Y-%m-%d").date()
    end_dt = dt.datetime.strptime(chunk_end, "%Y-%m-%d").date()
    length = (end_dt - start_dt).days + 1

    if length > 13:
        raise ValueError(f"Chunk too large: {length} days. Braze API limit is 13 days.")

    logger.debug(
        f"Fetching {length} days of data for Canvas {canvas_id} from {chunk_start} to {chunk_end}"
    )

    try:
        resp = requests.get(
            f"{BRAZE_ENDPOINT}/canvas/data_series",
            headers=headers,
            params={
                "canvas_id": canvas_id,
                "length": length,
                "ending_at": chunk_end,
                "include_step_breakdown": "true",
                "include_variant_breakdown": "true",
            },
            timeout=30,
        )
        logger.debug(f"Response status: {resp.status_code} for Canvas {canvas_id}")
        resp.raise_for_status()
    except requests.HTTPError as e:
        error_msg = "Unknown error"
        status_code = "Unknown"
        try:
            if hasattr(e, "response") and e.response is not None:
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
    if (
        "data" not in api_data
        or "stats" not in api_data["data"]
        or not api_data["data"]["stats"]
    ):
        logger.debug(
            f"No data returned for Canvas {canvas_id} chunk {chunk_start} to {chunk_end}"
        )
        return []

    # Process and normalize each day's data with step breakdown
    normalized_data = []
    stats_array = api_data["data"]["stats"]

    # The API returns data with explicit dates in the "time" field
    for stat_item in stats_array:
        time_str = stat_item.get("time", "")
        total_stats = stat_item.get("total_stats", {})
        step_stats = stat_item.get("step_stats", {})

        # Parse the date from the time field
        try:
            stat_date = dt.datetime.strptime(time_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid date format in API response: {time_str}")
            continue

        # Create the normalized record with step breakdown
        normalized_record: Dict[str, Any] = {
            "date": stat_date.isoformat(),
            "total_stats": {
                "entries": total_stats.get("entries", 0),
                "revenue": total_stats.get("revenue", 0.0),
                "conversions": total_stats.get("conversions", 0),
                "conversions_by_entry_time": total_stats.get(
                    "conversions_by_entry_time", 0
                ),
            },
            "step_stats": {},
        }

        # Process step-level statistics
        for step_id, step_data in step_stats.items():
            step_info = {
                "name": step_data.get("name", "Unknown"),
                "revenue": step_data.get("revenue", 0.0),
                "conversions": step_data.get("conversions", 0),
                "conversions_by_entry_time": step_data.get(
                    "conversions_by_entry_time", 0
                ),
                "unique_recipients": step_data.get("unique_recipients", 0),
                "messages": step_data.get("messages", {}),
            }
            normalized_record["step_stats"][step_id] = step_info

        normalized_data.append(normalized_record)

    logger.debug(
        f"Processed {len(normalized_data)} days of data for Canvas {canvas_id}"
    )
    return normalized_data


def get_canvas_data_series(
    canvas_id: str, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """
    Fetch historical data series for a Canvas ID, breaking large ranges into chunks.

    Args:
        canvas_id: Canvas ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of daily statistics dictionaries
    """
    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1

    logger.debug(
        f"Fetching {total_days} days of data for Canvas {canvas_id}, breaking into chunks"
    )

    all_data = []
    chunk_size = 13  # Maximum allowed by Braze API

    current_start = start_dt

    while current_start <= end_dt:
        # Calculate chunk end date (don't exceed the overall end date)
        chunk_end = min(current_start + dt.timedelta(days=chunk_size - 1), end_dt)

        chunk_start_str = current_start.isoformat()
        chunk_end_str = chunk_end.isoformat()

        try:
            chunk_data = get_canvas_data_chunk(
                canvas_id, chunk_start_str, chunk_end_str
            )
            all_data.extend(chunk_data)
            logger.debug(
                f"✓ Successfully fetched chunk {chunk_start_str} to {chunk_end_str} for Canvas {canvas_id}: "
                f"{len(chunk_data)} records"
            )

            # Small delay between chunks to be gentle on the API
            time.sleep(0.1)

        except Exception as e:
            logger.warning(
                f"✗ Failed to fetch chunk {chunk_start_str} to {chunk_end_str} for Canvas {canvas_id}: "
                f"{type(e).__name__}: {e}"
            )

        # Move to next chunk
        current_start = chunk_end + dt.timedelta(days=1)

    # Sort by date (oldest first) for consistent order
    all_data.sort(key=lambda x: x["date"])

    logger.debug(
        f"Fetched total of {len(all_data)} days of data for Canvas {canvas_id}"
    )
    return all_data


def append_canvas_data(canvas_id: str, daily_records: List[Dict[str, Any]]) -> int:
    """
    Append historical records to Canvas step-based directory structure.
    If data for the same date already exists, it will be replaced with the new data.

    Args:
        canvas_id: Canvas ID
        daily_records: List of daily statistics dictionaries with step breakdown

    Returns:
        Number of new/updated records added across all steps
    """
    canvas_dir = DATA_DIR / canvas_id
    total_new_records = 0

    if not daily_records:
        logger.debug(f"No records to process for Canvas {canvas_id}")
        return 0

    # Create canvas directory if it doesn't exist
    canvas_dir.mkdir(exist_ok=True)

    # Process each day's data
    for record in daily_records:
        date = record["date"]
        step_stats = record.get("step_stats", {})

        # Process each step
        for step_id, step_data in step_stats.items():
            step_dir = canvas_dir / step_id
            step_dir.mkdir(exist_ok=True)

            # Create step metadata file if it doesn't exist
            step_metadata_file = step_dir / "metadata.json"
            if not step_metadata_file.exists():
                metadata = {
                    "step_id": step_id,
                    "step_name": step_data.get("name", "Unknown"),
                    "canvas_id": canvas_id,
                    "created_at": dt.datetime.now().isoformat(),
                }
                with step_metadata_file.open("w") as f:
                    json.dump(metadata, f, indent=2)

            # Process each message channel
            messages = step_data.get("messages", {})
            for channel, channel_data in messages.items():
                if not isinstance(channel_data, list) or not channel_data:
                    continue

                channel_file = step_dir / f"{channel}.jsonl"

                # Read existing records and filter out records with the same date
                existing_records: List[Dict[str, Any]] = []
                records_replaced = 0
                if channel_file.exists():
                    with channel_file.open("r") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    existing_record = json.loads(line)
                                    if existing_record.get("date") != date:
                                        existing_records.append(existing_record)
                                    else:
                                        records_replaced += 1
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "Skipping malformed line in {}".format(channel_file)
                                    )
                                    continue

                # Prepare the new channel record
                channel_record = {
                    "date": date,
                    "step_id": step_id,
                    "step_name": step_data.get("name", "Unknown"),
                    "canvas_id": canvas_id,
                    "channel": channel,
                }

                # Add all the message metrics
                if len(channel_data) > 0:
                    metrics = channel_data[0]  # Usually first element contains the metrics
                    channel_record.update(metrics)

                # Add the new record to the list
                existing_records.append(channel_record)

                # Write all records back to the file (overwriting the entire file)
                with channel_file.open("w") as f:
                    for record in existing_records:
                        f.write(json.dumps(record) + "\n")

                total_new_records += 1
                if records_replaced > 0:
                    logger.debug(
                        "Updated record for {}/{}/{} on {} (replaced {} existing record(s))".format(
                            canvas_id, step_id, channel, date, records_replaced
                        )
                    )
                else:
                    logger.debug(
                        "Added new record for {}/{}/{} on {}".format(
                            canvas_id, step_id, channel, date
                        )
                    )

    return total_new_records


def ingest_historical_data(
    start_date: str, end_date: str, filter_prefix: Optional[str] = None
) -> None:
    """
    Main function to ingest historical data for all Canvas IDs.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        filter_prefix: Optional prefix to filter Canvas names (case-insensitive)
    """
    logger.info(
        f"Starting historical Canvas data ingestion from {start_date} to {end_date}"
    )
    if filter_prefix:
        logger.info(f"Filtering canvases by name prefix: '{filter_prefix}'")

    try:
        # Get Canvas IDs and names
        canvas_ids = get_canvas_ids(filter_prefix)
        logger.info(f"Processing {len(canvas_ids)} Canvas IDs")

        # Get name mapping for better reporting
        name_mapping = get_canvas_name_mapping(filter_prefix)

        success_count = 0
        partial_success_count = 0
        error_count = 0
        total_records_processed = 0

        for i, canvas_id in enumerate(canvas_ids, 1):
            canvas_name = name_mapping.get(canvas_id, "Unknown")

            try:
                # Fetch historical data for this Canvas
                daily_records = get_canvas_data_series(canvas_id, start_date, end_date)

                if daily_records:
                    # Process and update JSONL file
                    records_processed = append_canvas_data(canvas_id, daily_records)
                    total_records_processed += records_processed

                    # Calculate expected vs actual records to detect partial failures
                    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
                    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
                    expected_days = (end_dt - start_dt).days + 1

                    if len(daily_records) == expected_days:
                        success_count += 1
                        logger.info(
                            "✓ ({}/{}) {:30} | {:3} records processed | {:3}/{:3} days (complete)".format(
                                i, len(canvas_ids), canvas_name[:30], records_processed,
                                len(daily_records), expected_days
                            )
                        )
                    else:
                        partial_success_count += 1
                        logger.warning(
                            "◐ ({}/{}) {:30} | {:3} records processed | {:3}/{:3} days (partial)".format(
                                i, len(canvas_ids), canvas_name[:30], records_processed,
                                len(daily_records), expected_days
                            )
                        )
                else:
                    error_count += 1
                    logger.warning(
                        "◯ ({}/{}) {:30} | No data available".format(i, len(canvas_ids), canvas_name[:30])
                    )

            except Exception as exc:
                error_count += 1
                logger.error(
                    "⚠ ({}/{}) {:30} | Unexpected error: {}: {}".format(
                        i, len(canvas_ids), canvas_name[:30], type(exc).__name__, exc
                    )
                )

            finally:
                # Rate limiting and progress updates
                time.sleep(0.3)  # Small delay between Canvas requests

                # Progress update every 10 canvases
                if i % 10 == 0:
                    print("Progress update: processed {} canvases".format(i))

        # Summary
        logger.info("Historical ingestion complete:")
        logger.info("  • Complete success: {}".format(success_count))
        logger.info("  • Partial success: {}".format(partial_success_count))
        logger.info("  • Failed: {}".format(error_count))
        logger.info("  • Total records processed: {}".format(total_records_processed))
        logger.info(
            "  • Date range: {} to {}".format(
                start_date, end_date
            )
        )

    except Exception as exc:
        logger.error(
            "Fatal error during historical ingestion: {}".format(exc)
        )
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

  # Ingest only canvases with names starting with "Campaign"
  BRAZE_REST_KEY=your-key python src/ingest_historical.py --days 30 --filter-prefix Campaign

  # Ingest specific date range with filter
  BRAZE_REST_KEY=your-key python src/ingest_historical.py --start-date 2023-11-01 \
    --end-date 2023-12-31 --filter-prefix "Weekly"
        """,
    )

    parser.add_argument(
        "--days",
        type=int,
        help="Number of days back from today to ingest (e.g., 60 for last 60 days)",
    )

    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format")

    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--filter-prefix",
        type=str,
        help="Only ingest canvases whose names start with this prefix (case-insensitive)",
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
            dt.datetime.strptime(start_date_str, "%Y-%m-%d")
            dt.datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            sys.exit(1)

    try:
        ingest_historical_data(start_date_str, end_date_str, args.filter_prefix)
        logger.info("✅ Historical ingestion completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
