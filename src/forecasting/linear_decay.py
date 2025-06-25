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
from scipy import stats
from scipy.optimize import curve_fit


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

        # Dictionary to aggregate metrics by date
        daily_aggregates: Dict[date, DailyAggregates] = defaultdict(
            lambda: {
                "total_sent": 0,
                "total_opens": 0,
                "total_unique_opens": 0,
                "total_clicks": 0,
                "total_unique_clicks": 0,
                "total_delivered": 0,
                "total_bounces": 0,
                "total_unsubscribes": 0,
                "active_steps": set(),
                "active_channels": set(),
            }
        )

        try:
            # Iterate through all step directories
            for step_dir in canvas_dir.iterdir():
                if not step_dir.is_dir():
                    continue

                step_id = step_dir.name

                # Process each channel file in the step
                for channel_file in step_dir.iterdir():
                    if not channel_file.is_file() or not channel_file.name.endswith(
                        ".jsonl"
                    ):
                        continue

                    channel_name = channel_file.stem

                    # Read the channel data
                    with channel_file.open("r") as f:
                        for line in f:
                            if not line.strip():
                                continue

                            try:
                                record = json.loads(line)
                                record_date = record.get("date")

                                if not record_date:
                                    continue

                                # Parse date
                                try:
                                    parsed_date = datetime.strptime(
                                        record_date, "%Y-%m-%d"
                                    ).date()
                                except ValueError:
                                    logger.warning(
                                        f"Invalid date format: {record_date}"
                                    )
                                    continue

                                # Aggregate metrics for this date
                                day_data = daily_aggregates[parsed_date]
                                day_data["total_sent"] += record.get("sent", 0)
                                day_data["total_opens"] += record.get("opens", 0)
                                day_data["total_unique_opens"] += record.get(
                                    "unique_opens", 0
                                )
                                day_data["total_clicks"] += record.get("clicks", 0)
                                day_data["total_unique_clicks"] += record.get(
                                    "unique_clicks", 0
                                )
                                day_data["total_delivered"] += record.get(
                                    "delivered", 0
                                )
                                day_data["total_bounces"] += record.get("bounces", 0)
                                day_data["total_unsubscribes"] += record.get(
                                    "unsubscribes", 0
                                )
                                day_data["active_steps"].add(step_id)
                                day_data["active_channels"].add(
                                    f"{step_id}:{channel_name}"
                                )

                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Error parsing JSON in {channel_file}: {e}"
                                )
                                continue

        except Exception as e:
            logger.error(f"Error loading data for Canvas {canvas_id}: {e}")
            return []

        # Convert aggregated data to CanvasMetrics objects
        canvas_metrics = []
        for date_key, metrics in daily_aggregates.items():
            canvas_metric = CanvasMetrics(
                date=date_key,
                canvas_id=canvas_id,
                total_sent=metrics["total_sent"],
                total_opens=metrics["total_opens"],
                total_unique_opens=metrics["total_unique_opens"],
                total_clicks=metrics["total_clicks"],
                total_unique_clicks=metrics["total_unique_clicks"],
                total_delivered=metrics["total_delivered"],
                total_bounces=metrics["total_bounces"],
                total_unsubscribes=metrics["total_unsubscribes"],
                active_steps=len(metrics["active_steps"]),
                active_channels=len(metrics["active_channels"]),
            )
            canvas_metrics.append(canvas_metric)

        # Sort by date (oldest first)
        canvas_metrics.sort(key=lambda x: x.date)
        logger.debug(
            f"Loaded {len(canvas_metrics)} days of aggregated data for Canvas {canvas_id}"
        )

        return canvas_metrics

    def _linear_decay_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear decay function: y = ax + b"""
        return a * x + b

    def _exponential_decay_func(
        self, x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Exponential decay function: y = a * exp(-bx) + c"""
        return a * np.exp(-b * x) + c

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

                    # Initial guess for exponential decay
                    initial_guess = [y_nonzero[0], 0.1, self.quiet_threshold]

                    popt, _ = curve_fit(
                        self._exponential_decay_func,
                        x_nonzero,
                        y_nonzero,
                        p0=initial_guess,
                        maxfev=1000,
                    )

                    # Calculate R-squared for exponential model
                    y_pred_exp = self._exponential_decay_func(x_nonzero, *popt)
                    ss_res = np.sum((y_nonzero - y_pred_exp) ** 2)
                    ss_tot = np.sum((y_nonzero - np.mean(y_nonzero)) ** 2)
                    exp_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    models.append(
                        {
                            "type": "exponential",
                            "params": {"a": popt[0], "b": popt[1], "c": popt[2]},
                            "r_squared": exp_r_squared,
                            "function": lambda x: self._exponential_decay_func(
                                x, *popt
                            ),
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
                    if recent_trend < -1:
                        trend = "declining"
                    elif recent_trend > 1:
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

                # Adjust confidence based on data quality
                if len(canvas_metrics) < 14:
                    confidence *= 0.7  # Reduce confidence for limited data

                if trend == "growing":
                    confidence *= 0.5  # Reduce confidence if trend is growing

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

        return {
            "summary": {
                "total_canvases": total_canvases,
                "predictable": predictable,
                "unpredictable": unpredictable,
                "going_quiet_soon": going_quiet_soon,
                "going_quiet_later": going_quiet_later,
            },
            "trends": trend_counts,
            "confidence_distribution": confidence_distribution,
            "all_canvases": all_canvases,
            "generated_at": datetime.now().isoformat(),
        }
