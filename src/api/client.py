"""Braze API client implementation."""

import json
import logging
from datetime import datetime
from typing import List

import requests
from requests import Response

from ..models import CanvasListResponse, CanvasDetails, RequestLog

logger = logging.getLogger(__name__)

BRAZE_ENDPOINT = "https://rest.iad-02.braze.com"
TIMEOUT = 10  # seconds


class BrazeAPIClient:
    """Client for interacting with the Braze REST API."""

    def __init__(self, api_key: str, endpoint: str = BRAZE_ENDPOINT):
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.request_log: List[RequestLog] = []

    def _make_request(self, path: str, method: str = "GET", **params) -> Response:
        """Make a request to the Braze API and log it."""
        url = f"{self.endpoint}{path}"
        start_time = datetime.now()

        try:
            logger.info(f"Making {method} request to {url} with params: {params}")

            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=TIMEOUT)
            else:
                response = requests.request(method, url, headers=self.headers, json=params, timeout=TIMEOUT)

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            # Log the request
            log_entry = RequestLog(
                timestamp=start_time.isoformat(),
                endpoint=path,
                method=method,
                status_code=response.status_code,
                response_time_ms=response_time,
                success=response.ok
            )

            if not response.ok:
                log_entry.error_message = response.text
                logger.error(f"API request failed: {response.status_code} - {response.text}")
            else:
                logger.info(f"API request successful: {response.status_code} (took {response_time:.2f}ms)")

            self.request_log.append(log_entry)
            response.raise_for_status()
            return response

        except requests.RequestException as err:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            log_entry = RequestLog(
                timestamp=start_time.isoformat(),
                endpoint=path,
                method=method,
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error_message=str(err)
            )
            self.request_log.append(log_entry)
            logger.error(f"Request failed: {err}")
            raise

    def get_canvas_list(self, limit: int = 100) -> CanvasListResponse:
        """Get list of all canvases."""
        response = self._make_request("/canvas/list", limit=limit)
        data = response.json()
        return CanvasListResponse.from_dict(data)

    def get_canvas_details(self, canvas_id: str) -> CanvasDetails:
        """Get detailed information about a specific canvas."""
        response = self._make_request("/canvas/details", canvas_id=canvas_id)
        data = response.json()

        # The response might be wrapped in a different structure
        canvas_data = data.get('canvas', data) if 'canvas' in data else data
        return CanvasDetails.from_dict(canvas_data, canvas_id)

    def save_request_log(self, filename: str = "request_log.json"):
        """Save the request log to a JSON file."""
        log_data = [log.to_dict() for log in self.request_log]
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Request log saved to {filename}")