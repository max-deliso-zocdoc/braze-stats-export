"""Request logging data model."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class RequestLog:
    """Log entry for API requests."""

    timestamp: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
