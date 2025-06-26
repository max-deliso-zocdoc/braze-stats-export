"""Linear decay forecasting model for Canvas quiet date prediction.

This module implements a simple linear regression model to forecast when
Canvas message metrics will decay to approximately zero (the "quiet date").
Works with step-based directory structure for granular Canvas analytics.
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, NamedTuple, Set
from dataclasses import dataclass
import math
from collections import defaultdict
import os
import requests
from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


class DailyAggregates(TypedDict):
    total_sent: int
    total_opens: int
    total_unique_opens: int
    total_clicks: int
    total_unique_clicks: int
    total_delivered: int
    total_bounces: int
    total_unsubscribes: int
    active_steps: Set[str]
    active_channels: Set[str]


@dataclass
class ChannelMetrics:
    """Represents aggregated daily metrics for a specific channel."""

    date: date
    canvas_id: str
    channel: str
    sent: int = 0
    opens: int = 0
    unique_opens: int = 0
    clicks: int = 0
    unique_clicks: int = 0
    delivered: int = 0
    bounces: int = 0
    unsubscribes: int = 0
    # Add more metrics as needed


@dataclass
class CanvasMetrics:
    """Represents aggregated daily metrics for an entire Canvas."""

    date: date
    canvas_id: str
    total_sent: int = 0
    total_opens: int = 0
    total_unique_opens: int = 0
    total_clicks: int = 0
    total_unique_clicks: int = 0
    total_delivered: int = 0
    total_bounces: int = 0
    total_unsubscribes: int = 0
    active_steps: int = 0
    active_channels: int = 0


class ForecastResult(NamedTuple):
    """Result of quiet date forecast."""

    canvas_id: str
    canvas_name: str  # Add canvas name to the result
    quiet_date: Optional[date]
    confidence: float
    r_squared: float
    days_to_quiet: Optional[int]
    current_trend: str  # 'declining', 'stable', 'growing', 'insufficient_data'
    model_params: Dict[str, float]
    metric_used: str  # Which metric was used for forecasting


def _read_all_jsonl(canvas_dir: Path) -> pd.DataFrame:
    """Read all JSONL files in the canvas directory structure into a single DataFrame."""
    # Collect DataFrames for each file
    dfs: list[pd.DataFrame] = []
    for jsonl_path in canvas_dir.rglob("*.jsonl"):        # walks step/channel folders
        if jsonl_path.stat().st_size == 0:
            continue

        # Build two helper columns so we can compute "active steps/channels" later
        step_id     = jsonl_path.parent.name              # .../step_id/channel.jsonl
        channel_key = f"{step_id}:{jsonl_path.stem}"

        df = pd.read_json(jsonl_path, lines=True)
        df["step_id"]   = step_id
        df["chan_key"]  = channel_key
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()          # empty safeguard
    return pd.concat(dfs, ignore_index=True)


def _pandas_aggregate(canvas_dir: Path, canvas_id: str) -> List[CanvasMetrics]:
    """Vectorized aggregation using pandas pipeline."""
    # Read all JSONL files in one shot
    raw = _read_all_jsonl(canvas_dir)
    if raw.empty:
        return []

    # Parse & normalise dates in bulk
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()  # midnight
    raw = raw.dropna(subset=["date"])                                          # toss bad rows

    # Fill missing columns with 0 (for channels that don't have certain metrics)
    metric_columns = [
        "sent", "opens", "unique_opens", "clicks", "unique_clicks",
        "delivered", "bounces", "unsubscribes"
    ]
    for col in metric_columns:
        if col not in raw.columns:
            raw[col] = 0
        else:
            raw[col] = raw[col].fillna(0).astype(int)

    # Aggregate with one groupby.agg
    agg = (raw
           .groupby("date")
           .agg(
               total_sent          = ("sent",            "sum"),
               total_opens         = ("opens",           "sum"),
               total_unique_opens  = ("unique_opens",    "sum"),
               total_clicks        = ("clicks",          "sum"),
               total_unique_clicks = ("unique_clicks",   "sum"),
               total_delivered     = ("delivered",       "sum"),
               total_bounces       = ("bounces",         "sum"),
               total_unsubscribes  = ("unsubscribes",    "sum"),
               active_steps        = ("step_id",         "nunique"),
               active_channels     = ("chan_key",        "nunique"),
           )
           .reset_index()
           .sort_values("date")
    )

    # Convert rows → CanvasMetrics objects
    result: list[CanvasMetrics] = [
        CanvasMetrics(
            date              = row.date.date(),      # convert Timestamp → date
            canvas_id         = canvas_id,
            total_sent        = int(row.total_sent),
            total_opens       = int(row.total_opens),
            total_unique_opens= int(row.total_unique_opens),
            total_clicks      = int(row.total_clicks),
            total_unique_clicks=int(row.total_unique_clicks),
            total_delivered   = int(row.total_delivered),
            total_bounces     = int(row.total_bounces),
            total_unsubscribes= int(row.total_unsubscribes),
            active_steps      = int(row.active_steps),
            active_channels   = int(row.active_channels),
        )
        for row in agg.itertuples(index=False)
    ]
    return result


class StepBasedForecaster:
    """
    Linear decay model for forecasting Canvas quiet dates using step-based data.

    Aggregates metrics across all Canvas steps and channels to predict when
    message activity will reach approximately zero.
    """

    def __init__(self, quiet_threshold: int = 5, min_data_points: int = 7):
        """
        Initialize the forecaster.

        Args:
            quiet_threshold: Daily sends below this are considered "quiet"
            min_data_points: Minimum data points required for prediction
        """
        self.quiet_threshold = quiet_threshold
        self.min_data_points = min_data_points

    def load_canvas_metrics(
        self, canvas_id: str, data_dir: Optional[Path] = None
    ) -> List[CanvasMetrics]:
        """Load and aggregate Canvas metrics from step-based directory structure."""
        data_dir = data_dir or Path("data")
        canvas_dir = data_dir / canvas_id

        if not canvas_dir.exists():
            logger.warning(f"No data directory found for Canvas {canvas_id}")
            return []

        try:
            canvas_metrics = _pandas_aggregate(canvas_dir, canvas_id)
            logger.debug(
                f"Loaded {len(canvas_metrics)} days of aggregated data for Canvas {canvas_id}"
            )
            return canvas_metrics
        except Exception as e:
            logger.error(f"Error loading data for Canvas {canvas_id}: {e}")
            return []

    def _linear_decay_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear decay function: y = ax + b"""
        return a * x + b

    def _exponential_decay_func(
        self, x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Exponential decay function: y = a * exp(-bx) + c"""
        return a * np.exp(-b * x) + c

    def _log_exponential_decay_func(
        self, x: np.ndarray, log_a: float, b: float, c: float
    ) -> np.ndarray:
        """Log-space exponential decay function: y = exp(log_a - bx) + c"""
        return np.exp(log_a - b * x) + c

    def _fit_exponential_log_space(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Fit exponential decay in log space for better numerical stability.

        Args:
            x: Independent variable (days)
            y: Dependent variable (metrics)

        Returns:
            Tuple of (parameters, r_squared)
        """
        # Filter out zero and negative values for log transformation
        positive_mask = y > 0
        if np.sum(positive_mask) < self.min_data_points:
            raise ValueError("Insufficient positive data points for log-space fitting")

        x_positive = x[positive_mask]
        y_positive = y[positive_mask]

        # Ensure we have enough variation in the data
        if np.max(y_positive) - np.min(y_positive) < 1:
            raise ValueError("Insufficient variation in data for exponential fitting")

        # Transform to log space: log(y - c) = log_a - bx
        # We need to estimate c first, then fit log(y - c) vs x
        c_estimate = np.min(y_positive) * 0.1  # Small offset to avoid log(0)

        # Ensure y - c_estimate is positive for all values
        y_adjusted = y_positive - c_estimate
        if np.any(y_adjusted <= 0):
            # Adjust c_estimate to ensure all values are positive
            c_estimate = np.min(y_positive) - 0.1
            y_adjusted = y_positive - c_estimate
            if np.any(y_adjusted <= 0):
                raise ValueError("Cannot ensure positive values for log transformation")

        # Transform: log(y - c_estimate) = log_a - bx
        y_transformed = np.log(y_adjusted)

        # Check for infinite or NaN values
        if np.any(~np.isfinite(y_transformed)):
            raise ValueError("Log transformation produced infinite or NaN values")

        # Fit linear model: log(y - c) = log_a - bx
        # This is equivalent to: log(y - c) = log_a - bx
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_positive, y_transformed
        )

        # Extract parameters: slope = -b, intercept = log_a
        b_fitted = -slope
        log_a_fitted = intercept

        # Validate parameters
        if not np.isfinite(b_fitted) or not np.isfinite(log_a_fitted):
            raise ValueError("Fitted parameters are not finite")

        # Ensure b is positive (decay rate should be positive)
        if b_fitted <= 0:
            raise ValueError("Decay rate must be positive")

        # Calculate R-squared
        y_pred_transformed = log_a_fitted - b_fitted * x_positive
        ss_res = np.sum((y_transformed - y_pred_transformed) ** 2)
        ss_tot = np.sum((y_transformed - np.mean(y_transformed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return np.array([log_a_fitted, b_fitted, c_estimate]), r_squared

    def predict_quiet_date(
        self, canvas_id: str, data_dir: Optional[Path] = None
    ) -> ForecastResult:
        """
        Predict the quiet date for a Canvas using aggregated step metrics.

        Args:
            canvas_id: Canvas ID to analyze
            data_dir: Directory containing step-based data

        Returns:
            ForecastResult with prediction details
        """
        canvas_metrics = self.load_canvas_metrics(canvas_id, data_dir)

        if len(canvas_metrics) < self.min_data_points:
            return ForecastResult(
                canvas_id=canvas_id,
                canvas_name="",
                quiet_date=None,
                confidence=0.0,
                r_squared=0.0,
                days_to_quiet=None,
                current_trend="insufficient_data",
                model_params={},
                metric_used="none",
            )

        # Try different metrics to find the best one for forecasting
        metrics_to_try = [
            ("total_sent", [m.total_sent for m in canvas_metrics]),
            ("total_opens", [m.total_opens for m in canvas_metrics]),
            ("total_clicks", [m.total_clicks for m in canvas_metrics]),
            ("total_delivered", [m.total_delivered for m in canvas_metrics]),
        ]

        best_result: Optional[ForecastResult] = None
        best_r_squared = -1

        for metric_name, y_values in metrics_to_try:
            # Skip if no variation in the data
            if max(y_values) - min(y_values) < 1:
                continue

            # Convert dates to days since start
            dates = [m.date for m in canvas_metrics]
            base_date = dates[0]
            x_values = np.array([(d - base_date).days for d in dates])
            y_values_array = np.array(y_values)

            # Filter out days with zero values for exponential model
            non_zero_mask = y_values_array > 0

            # Try different models
            models = []

            # 1. Linear regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_values, y_values_array
                )
                linear_r_squared = r_value**2

                models.append(
                    {
                        "type": "linear",
                        "params": {"slope": slope, "intercept": intercept},
                        "r_squared": linear_r_squared,
                        "function": lambda x: slope * x + intercept,
                    }
                )
            except Exception as e:
                logger.debug(
                    f"Linear regression failed for {canvas_id} {metric_name}: {e}"
                )

            # 2. Exponential decay (if we have enough non-zero data)
            if np.sum(non_zero_mask) >= self.min_data_points:
                try:
                    x_nonzero = x_values[non_zero_mask]
                    y_nonzero = y_values_array[non_zero_mask]

                    # Use log-space fitting for better numerical stability
                    popt, exp_r_squared = self._fit_exponential_log_space(x_nonzero, y_nonzero)
                    log_a, b, c = popt

                    # Convert log_a back to a for compatibility
                    a = np.exp(log_a)

                    models.append(
                        {
                            "type": "exponential",
                            "params": {"a": a, "b": b, "c": c, "log_a": log_a},
                            "r_squared": exp_r_squared,
                            "function": lambda x: self._log_exponential_decay_func(x, log_a, b, c),
                        }
                    )
                except Exception as e:
                    logger.debug(
                        f"Exponential regression failed for {canvas_id} {metric_name}: {e}"
                    )

            if not models:
                continue

            # Choose the best model for this metric
            best_model = max(models, key=lambda m: m["r_squared"])

            if best_model["r_squared"] > best_r_squared:
                best_r_squared = best_model["r_squared"]

                # Determine current trend
                recent_values = y_values_array[-min(7, len(y_values_array)) :]  # Last week
                if len(recent_values) >= 2:
                    recent_trend = np.mean(np.diff(recent_values))
                    if recent_trend < -3:
                        trend = "declining"
                    elif recent_trend > 3:
                        trend = "growing"
                    else:
                        trend = "stable"
                else:
                    trend = "insufficient_data"

                # Predict quiet date
                quiet_date: Optional[date] = None
                days_to_quiet: Optional[int] = None
                confidence = best_model["r_squared"]

                if best_model["type"] == "linear":
                    # For linear model: solve slope * x + intercept = quiet_threshold
                    slope = best_model["params"]["slope"]
                    intercept = best_model["params"]["intercept"]

                    if slope < 0:  # Declining trend
                        days_to_threshold = (self.quiet_threshold - intercept) / slope
                        if days_to_threshold > 0:
                            quiet_date = base_date + timedelta(
                                days=int(days_to_threshold)
                            )
                            days_to_quiet = int(days_to_threshold) - len(canvas_metrics)

                elif best_model["type"] == "exponential":
                    # For exponential model: solve a * exp(-bx) + c = quiet_threshold
                    a, b, c = (
                        best_model["params"]["a"],
                        best_model["params"]["b"],
                        best_model["params"]["c"],
                    )

                    if (
                        a > 0
                        and b > 0
                        and (self.quiet_threshold - c) > 0
                        and (self.quiet_threshold - c) < a
                    ):
                        try:
                            days_to_threshold = (
                                -math.log((self.quiet_threshold - c) / a) / b
                            )
                            if days_to_threshold > 0:
                                quiet_date = base_date + timedelta(
                                    days=int(days_to_threshold)
                                )
                                days_to_quiet = int(days_to_threshold) - len(
                                    canvas_metrics
                                )
                        except (ValueError, ZeroDivisionError):
                            pass

                best_result = ForecastResult(
                    canvas_id=canvas_id,
                    canvas_name="",
                    quiet_date=quiet_date,
                    confidence=min(confidence, 1.0),
                    r_squared=best_model["r_squared"],
                    days_to_quiet=days_to_quiet,
                    current_trend=trend,
                    model_params=best_model["params"],
                    metric_used=metric_name,
                )

        # Return best result or insufficient data
        if best_result:
            return best_result
        else:
            return ForecastResult(
                canvas_id=canvas_id,
                canvas_name="",
                quiet_date=None,
                confidence=0.0,
                r_squared=0.0,
                days_to_quiet=None,
                current_trend="insufficient_data",
                model_params={},
                metric_used="none",
            )


class QuietDatePredictor:
    """High-level interface for Canvas quiet date prediction using step-based data."""

    def __init__(self, data_dir: Optional[Path] = None, quiet_threshold: int = 5):
        """
        Initialize the predictor.

        Args:
            data_dir: Directory containing step-based Canvas data
            quiet_threshold: Daily sends below this are considered "quiet"
        """
        self.data_dir = data_dir or Path("data")
        self.forecaster = StepBasedForecaster(quiet_threshold=quiet_threshold)
        self._canvas_names: Optional[Dict[str, str]] = None  # Cache for canvas names
        self.canvas_name_filter: Optional[str] = None  # Optional filter prefix for canvas names

    def _get_canvas_name_mapping(self) -> Dict[str, str]:
        """Get a mapping of Canvas ID to Canvas name from saved index or Braze API."""
        if self._canvas_names is not None:
            return self._canvas_names

        # First try to load from saved canvas index
        index_path = self.data_dir / "canvas_index.json"
        name_map: Dict[str, str] = {}

        if index_path.exists():
            try:
                with index_path.open("r") as f:
                    name_map = json.load(f)
                logger.info(f"Loaded {len(name_map)} canvas names from saved index")
            except Exception as e:
                logger.warning(f"Failed to load canvas index from {index_path}: {e}")

        # If we have names from index and they're recent enough, use them
        # Otherwise, fetch fresh data from API
        should_fetch_from_api = (
            len(name_map) == 0  # No saved data
            or not index_path.exists()  # No index file
            or (
                datetime.now() - datetime.fromtimestamp(index_path.stat().st_mtime)
            ).days
            > 1  # Index older than 1 day
        )

        if should_fetch_from_api:
            # Get API configuration
            rest_key = os.getenv("BRAZE_REST_KEY")
            if not rest_key:
                logger.warning(
                    "BRAZE_REST_KEY not found, using saved canvas names only"
                )
                self._canvas_names = name_map
                return name_map

            braze_endpoint = os.getenv("BRAZE_ENDPOINT", "iad-02")
            base_url = f"https://rest.{braze_endpoint}.braze.com"

            headers = {
                "Authorization": f"Bearer {rest_key}",
                "Content-Type": "application/json",
            }

            api_name_map = {}
            page = 0
            limit = 100

            try:
                logger.info("Fetching fresh canvas names from Braze API...")
                while True:
                    params = {
                        "page": page,
                        "limit": limit,
                        "include_archived": "false",
                        "sort_direction": "desc",
                    }

                    resp = requests.get(
                        f"{base_url}/canvas/list",
                        headers=headers,
                        params=params,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    canvas_batch = data.get("canvases", [])
                    for canvas in canvas_batch:
                        api_name_map[canvas["id"]] = canvas["name"]

                    # Check if we got fewer canvases than the limit, indicating last page
                    if len(canvas_batch) < limit:
                        break

                    page += 1

                # Use API data if successful
                name_map = api_name_map
                logger.info(f"Loaded {len(name_map)} canvas names from API")

            except Exception as e:
                logger.warning(f"Failed to fetch canvas names from API: {e}")
                if name_map:
                    logger.info(
                        f"Falling back to saved canvas names ({len(name_map)} canvases)"
                    )
                else:
                    logger.warning(
                        "No canvas names available - neither from API nor saved index"
                    )

        self._canvas_names = name_map

        # Save canvas index to data directory (update existing or create new)
        if name_map:
            try:
                with index_path.open("w") as f:
                    json.dump(name_map, f, indent=2)
                logger.info(f"Canvas index updated and saved to {index_path}")
            except Exception as e:
                logger.warning(f"Failed to save canvas index: {e}")

        return name_map

    def predict_all_canvases(self) -> List[ForecastResult]:
        """Predict quiet dates for all available canvases."""
        canvas_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        results: List[ForecastResult] = []

        logger.info(f"Analyzing {len(canvas_dirs)} Canvas directories...")

        for canvas_dir in canvas_dirs:
            canvas_id = canvas_dir.name

            # Skip if we have a name filter and this canvas doesn't match
            if self.canvas_name_filter:
                canvas_name = self._get_canvas_name_mapping().get(canvas_id, "")
                if not canvas_name.lower().startswith(self.canvas_name_filter.lower()):
                    continue

            try:
                result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

                # Add canvas name to result
                canvas_name = self._get_canvas_name_mapping().get(canvas_id, canvas_id)
                result = ForecastResult(
                    canvas_id=result.canvas_id,
                    canvas_name=canvas_name,
                    quiet_date=result.quiet_date,
                    confidence=result.confidence,
                    r_squared=result.r_squared,
                    days_to_quiet=result.days_to_quiet,
                    current_trend=result.current_trend,
                    model_params=result.model_params,
                    metric_used=result.metric_used,
                )

                results.append(result)

                if result.quiet_date:
                    logger.info(
                        f"Canvas {canvas_name} ({canvas_id}): "
                        f"Quiet date {result.quiet_date} "
                        f"(confidence: {result.confidence:.1%}, "
                        f"trend: {result.current_trend})"
                    )
                else:
                    logger.debug(
                        f"Canvas {canvas_name} ({canvas_id}): "
                        f"No quiet date predicted "
                        f"(trend: {result.current_trend})"
                    )

            except Exception as e:
                logger.error(f"Error predicting quiet date for Canvas {canvas_id}: {e}")
                continue

        logger.info(f"Completed analysis of {len(results)} canvases")
        return results

    def generate_forecast_report(self) -> Dict[str, Any]:
        """Generate a comprehensive forecast report."""
        results = self.predict_all_canvases()

        # Calculate summary statistics
        total_canvases = len(results)
        predictable = sum(1 for r in results if r.quiet_date is not None)
        unpredictable = total_canvases - predictable

        # Count canvases going quiet soon (≤30 days) vs later
        going_quiet_soon = 0
        going_quiet_later = 0
        for result in results:
            if result.quiet_date and result.days_to_quiet:
                if result.days_to_quiet <= 30:
                    going_quiet_soon += 1
                else:
                    going_quiet_later += 1

        # Analyze trends
        trend_counts: Dict[str, int] = {}
        for result in results:
            trend = result.current_trend
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        # Analyze confidence distribution
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for result in results:
            if result.confidence >= 0.7:
                confidence_distribution["high"] += 1
            elif result.confidence >= 0.4:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1

        # Prepare detailed results
        all_canvases = []
        for result in results:
            canvas_data = {
                "canvas_id": result.canvas_id,
                "canvas_name": result.canvas_name,
                "quiet_date": result.quiet_date.isoformat() if result.quiet_date else None,
                "days_to_quiet": result.days_to_quiet,
                "confidence": result.confidence,
                "r_squared": result.r_squared,
                "trend": result.current_trend,
                "metric_used": result.metric_used,
                "model_params": result.model_params,
            }
            all_canvases.append(canvas_data)

        # Calculate averages for predictable canvases
        predictable_results = []
        for r in results:
            if r.quiet_date is not None and r.days_to_quiet is not None:
                predictable_results.append(r)

        if predictable_results:
            total_days: float = 0.0
            total_confidence: float = 0.0
            for r in predictable_results:
                total_days += float(r.days_to_quiet)  # type: ignore
                total_confidence += r.confidence
            avg_days_to_quiet = total_days / len(predictable_results)
            avg_confidence = total_confidence / len(predictable_results)
        else:
            avg_days_to_quiet = 0
            avg_confidence = 0

        return {
            "summary": {
                "total_canvases": total_canvases,
                "predictable": predictable,
                "unpredictable": unpredictable,
                "going_quiet_soon": going_quiet_soon,
                "going_quiet_later": going_quiet_later,
                "prediction_rate": (predictable / total_canvases) if total_canvases else 0,
                "avg_days_to_quiet": avg_days_to_quiet,
                "avg_confidence": avg_confidence,
            },
            "trends": trend_counts,
            "confidence_distribution": confidence_distribution,
            "all_canvases": all_canvases,
            "generated_at": datetime.now().isoformat(),
        }
