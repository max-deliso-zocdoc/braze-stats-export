"""Utility functions for the Braze Canvas export tool."""

from pathlib import Path
from typing import Optional
import json
import logging

# Configuration
BRAZE_DASHBOARD_BASE_URL = "https://dashboard-02.braze.com"

logger = logging.getLogger(__name__)


def get_first_step_id_for_canvas(
    canvas_id: str, data_dir: Path = None
) -> Optional[str]:
    """
    Get the first available step ID for a given Canvas ID.

    Args:
        canvas_id: The Canvas ID
        data_dir: Path to the data directory (defaults to 'data/')

    Returns:
        First step ID found for the Canvas, or None if not found
    """
    if data_dir is None:
        data_dir = Path("data")

    canvas_dir = data_dir / canvas_id
    if not canvas_dir.exists():
        return None

    # Look for the first step directory
    for step_dir in canvas_dir.iterdir():
        if step_dir.is_dir():
            return step_dir.name

    return None


def generate_canvas_url(
    canvas_id: str,
    step_id: str = None,
    dashboard_base_url: str = BRAZE_DASHBOARD_BASE_URL,
) -> str:
    """
    Generate a Braze dashboard URL for a Canvas.

    Args:
        canvas_id: The Canvas ID
        step_id: The Step ID (optional - if not provided, will try to find one)
        dashboard_base_url: Base URL for the Braze dashboard

    Returns:
        Clickable URL to the Canvas in the Braze dashboard
    """
    # If no step_id provided, try to find the first available one
    if step_id is None:
        step_id = get_first_step_id_for_canvas(canvas_id)
        if step_id is None:
            # Fallback to old format if no step found
            logger.warning(
                f"No step ID found for Canvas {canvas_id}, using fallback URL format"
            )
            # Strip dashes from canvas_id for URL
            clean_canvas_id = canvas_id.replace("-", "")
            return f"{dashboard_base_url}/engagement/canvas/{clean_canvas_id}?locale=en&version=flow"

    # Strip dashes from both IDs for the dashboard URL
    clean_canvas_id = canvas_id.replace("-", "")
    clean_step_id = step_id.replace("-", "")

    # Use the correct format with step_id
    return f"{dashboard_base_url}/engagement/canvas/{clean_canvas_id}/{clean_step_id}?locale=en&version=flow&isEditing=false"


def format_canvas_name_with_url(
    canvas_name: str, canvas_id: str, step_id: str = None, max_length: int = 40
) -> str:
    """
    Format a canvas name with its URL for display.

    Args:
        canvas_name: The name of the Canvas
        canvas_id: The Canvas ID
        step_id: The Step ID (optional)
        max_length: Maximum length for the name display

    Returns:
        Formatted string with name and URL
    """
    truncated_name = (
        canvas_name[:max_length] if len(canvas_name) > max_length else canvas_name
    )
    canvas_url = generate_canvas_url(canvas_id, step_id)
    return f"{truncated_name:40} | {canvas_url}"
