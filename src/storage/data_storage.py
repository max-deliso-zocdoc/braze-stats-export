"""Data storage implementation for JSON persistence."""

import json
import logging
from dataclasses import asdict
from typing import List, Optional

from ..models import CanvasListResponse, CanvasDetails

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles storing and loading canvas data."""

    @staticmethod
    def save_canvas_list(
        canvas_list: CanvasListResponse, filename: str = "canvas_list.json"
    ):
        """Save canvas list to JSON file."""
        data = asdict(canvas_list)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Canvas list saved to {filename} ({len(canvas_list.canvases)} canvases)"
        )

    @staticmethod
    def save_canvas_details(
        canvas_details: List[CanvasDetails], filename: str = "canvas_details.json"
    ):
        """Save detailed canvas information to JSON file."""
        data = [asdict(canvas) for canvas in canvas_details]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Canvas details saved to {filename} ({len(canvas_details)} canvases)"
        )

    @staticmethod
    def load_canvas_list(
        filename: str = "canvas_list.json",
    ) -> Optional[CanvasListResponse]:
        """Load canvas list from JSON file."""
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return CanvasListResponse.from_dict(data)
        except FileNotFoundError:
            logger.warning(f"File {filename} not found")
            return None
