#!/usr/bin/env python
"""Stub Braze API caller.

This script hits the Braze REST API at the iad‑02 cluster and prints the JSON
response.  It expects an environment variable `BRAZE_REST_KEY` containing a
valid Braze REST key with permission to call the chosen endpoint.

Run with:
    BRAZE_REST_KEY=<<your‑key>> python src/main.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Final

import requests
from requests import Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BRAZE_ENDPOINT: Final[str] = "https://rest.iad-02.braze.com"
DEFAULT_PATH: Final[str] = "/canvas/list"  # change to any endpoint you need
TIMEOUT: Final[int] = 10  # seconds


def call_braze(path: str = DEFAULT_PATH, **params) -> Response:
    """Make a GET request to the given Braze path with the REST key header."""

    api_key = os.getenv("BRAZE_REST_KEY")
    if not api_key:
        sys.exit("BRAZE_REST_KEY environment variable is not set.")

    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{BRAZE_ENDPOINT}{path}"

    response: Response = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response


def main() -> None:
    """Entry‑point: hit the API and pretty‑print the JSON payload."""

    try:
        resp = call_braze(limit=1)  # keep it light for the stub
    except requests.HTTPError as err:
        sys.exit(f"Braze API error: {err} → {err.response.text}")

    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
