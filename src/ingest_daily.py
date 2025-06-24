"""Daily Canvas data ingestion script.

This script pulls daily Canvas statistics from the Braze API and stores them
in append-only JSONL files, one per Canvas ID.

Run with:
    BRAZE_REST_KEY=<<your-key>> python src/ingest_daily.py
"""

import datetime as dt
import json
import os
from pathlib import Path
from typing import Final, List, Dict, Any
import logging

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file if present
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
        logging.FileHandler('braze_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API configuration
YESTERDAY = (dt.date.today() - dt.timedelta(days=1)).isoformat()


def get_api_config():
    """Get API configuration, checking for required environment variables."""
    api_key = os.environ.get("BRAZE_REST_KEY")
    if not api_key:
        raise ValueError("BRAZE_REST_KEY environment variable is required")

    return {
        "api_key": api_key,
        "headers": {"Authorization": f"Bearer {api_key}"}
    }


def canvas_ids() -> List[str]:
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


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def daily_row(cid: str, date: str = YESTERDAY) -> Dict[str, Any]:
    """
    Fetch daily statistics for a specific Canvas ID.

    Args:
        cid: Canvas ID
        date: Date in YYYY-MM-DD format (defaults to yesterday)

    Returns:
        Dictionary with daily statistics including date, entries, sends, etc.
    """
    api_config = get_api_config()
    headers = api_config["headers"]

    resp = requests.get(
        f"{BRAZE_ENDPOINT}/canvas/data_series",
        headers=headers,
        params={"canvas_id": cid, "length": 1, "ending_at": date},
        timeout=10,
    )
    resp.raise_for_status()

    api_data = resp.json()

    # Handle empty or malformed responses
    if "data" not in api_data or not api_data["data"]:
        return {
            "date": date,
            "entries": 0,
            "sends": 0,
            "delivered": 0,
            "opens": 0,
            "conversions": 0
        }

    data = api_data["data"][0]  # single-day slice

    # Normalize the data structure and add explicit date field
    normalized_data = {
        "date": date,
        "entries": data.get("entries", 0),
        "sends": data.get("sends", 0),
        "delivered": data.get("delivered", 0),
        "opens": data.get("opens", 0),
        "conversions": data.get("conversions", 0)
    }

    return normalized_data


def append_jsonl(cid: str, row: Dict[str, Any]) -> bool:
    """
    Append a daily record to the Canvas JSONL file.

    Args:
        cid: Canvas ID
        row: Daily statistics dictionary

    Returns:
        True if data was appended, False if already exists
    """
    path = DATA_DIR / f"{cid}.jsonl"
    date = row["date"]

    # Skip if this date already present (idempotent operation)
    if path.exists():
        with path.open('r') as fp:
            for line in fp:
                if line.strip() and f'"date":"{date}"' in line:
                    logger.debug(f"Data for {cid} on {date} already exists, skipping")
                    return False

    # Append new record
    with path.open("a") as fp:
        fp.write(json.dumps(row) + "\n")

    logger.debug(f"Appended data for {cid} on {date}")
    return True


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


def main(target_date: str = None) -> None:
    """
    Main ingestion function.

    Args:
        target_date: Optional date override in YYYY-MM-DD format
    """
    date_to_process = target_date or YESTERDAY
    logger.info(f"Starting daily Canvas data ingestion for {date_to_process}")

    try:
        # Get Canvas IDs and names (this will check for API key)
        ids = canvas_ids()
        logger.info(f"Processing {len(ids)} Canvas IDs")

        # Get name mapping for better reporting
        name_mapping = get_canvas_name_mapping()

        success_count = 0
        skip_count = 0
        error_count = 0

        for i, cid in enumerate(ids, 1):
            canvas_name = name_mapping.get(cid, "Unknown")

            try:
                row = daily_row(cid, date_to_process)
                was_appended = append_jsonl(cid, row)

                if was_appended:
                    success_count += 1
                    logger.info(f"✓ ({i}/{len(ids)}) {canvas_name[:30]:30} | {row['sends']:5} sends | {row['entries']:5} entries")
                else:
                    skip_count += 1
                    logger.debug(f"◯ ({i}/{len(ids)}) {canvas_name[:30]:30} | Already exists")

            except Exception as exc:
                error_count += 1
                logger.error(f"⚠ ({i}/{len(ids)}) {canvas_name[:30]:30} | Error: {exc}")

        # Summary
        logger.info(f"Ingestion complete: {success_count} new records, {skip_count} skipped, {error_count} errors")

    except Exception as exc:
        logger.error(f"Fatal error during ingestion: {exc}")
        raise


if __name__ == "__main__":
    import sys

    # Check for API key early when running directly
    if not os.environ.get("BRAZE_REST_KEY"):
        logger.error("BRAZE_REST_KEY environment variable is required")
        sys.exit(1)

    # Allow date override via command line argument
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    if target_date:
        # Validate date format
        try:
            dt.datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {target_date}. Use YYYY-MM-DD")
            sys.exit(1)

    main(target_date)