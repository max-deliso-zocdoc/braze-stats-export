"""Linear decay forecasting model for Canvas quiet date prediction.

This module implements a simple linear regression model to forecast when
Canvas sends will decay to approximately zero (the "quiet date").
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Represents daily Canvas statistics."""
    date: date
    entries: int
    sends: int
    delivered: int
    opens: int
    conversions: int


class ForecastResult(NamedTuple):
    """Result of quiet date forecast."""
    canvas_id: str
    quiet_date: Optional[date]
    confidence: float
    r_squared: float
    days_to_quiet: Optional[int]
    current_trend: str  # 'declining', 'stable', 'growing', 'insufficient_data'
    model_params: Dict[str, float]


class LinearDecayForecaster:
    """
    Linear decay model for forecasting Canvas quiet dates.

    Uses simple linear regression on send data to predict when sends will
    reach approximately zero.
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

    def load_canvas_data(self, canvas_id: str, data_dir: Path = None) -> List[DailyStats]:
        """Load time series data for a Canvas from JSONL file."""
        data_dir = data_dir or Path("data")
        jsonl_path = data_dir / f"{canvas_id}.jsonl"

        if not jsonl_path.exists():
            logger.warning(f"No data file found for Canvas {canvas_id}")
            return []

        daily_stats = []
        try:
            with jsonl_path.open('r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        daily_stats.append(DailyStats(
                            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
                            entries=data.get('entries', 0),
                            sends=data.get('sends', 0),
                            delivered=data.get('delivered', 0),
                            opens=data.get('opens', 0),
                            conversions=data.get('conversions', 0)
                        ))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error loading data for Canvas {canvas_id}: {e}")
            return []

        # Sort by date to ensure proper time series order
        daily_stats.sort(key=lambda x: x.date)
        logger.debug(f"Loaded {len(daily_stats)} data points for Canvas {canvas_id}")

        return daily_stats

    def _linear_decay_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear decay function: y = ax + b"""
        return a * x + b

    def _exponential_decay_func(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Exponential decay function: y = a * exp(-bx) + c"""
        return a * np.exp(-b * x) + c

    def predict_quiet_date(self, canvas_id: str, data_dir: Path = None) -> ForecastResult:
        """
        Predict the quiet date for a Canvas using linear regression.

        Args:
            canvas_id: Canvas ID to analyze
            data_dir: Directory containing JSONL data files

        Returns:
            ForecastResult with prediction details
        """
        daily_stats = self.load_canvas_data(canvas_id, data_dir)

        if len(daily_stats) < self.min_data_points:
            return ForecastResult(
                canvas_id=canvas_id,
                quiet_date=None,
                confidence=0.0,
                r_squared=0.0,
                days_to_quiet=None,
                current_trend='insufficient_data',
                model_params={}
            )

        # Prepare data for regression
        dates = [stat.date for stat in daily_stats]
        sends = [stat.sends for stat in daily_stats]

        # Convert dates to days since the first date
        base_date = dates[0]
        x_days = np.array([(d - base_date).days for d in dates])
        y_sends = np.array(sends)

        # Filter out days with zero sends to avoid log issues for exponential model
        non_zero_mask = y_sends > 0

        # Try different models and choose the best one
        models = []

        # 1. Linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_days, y_sends)
            linear_r_squared = r_value ** 2

            models.append({
                'type': 'linear',
                'params': {'slope': slope, 'intercept': intercept},
                'r_squared': linear_r_squared,
                'function': lambda x: slope * x + intercept
            })
        except Exception as e:
            logger.debug(f"Linear regression failed for {canvas_id}: {e}")

        # 2. Exponential decay (if we have non-zero data)
        if np.sum(non_zero_mask) >= self.min_data_points:
            try:
                x_nonzero = x_days[non_zero_mask]
                y_nonzero = y_sends[non_zero_mask]

                # Initial guess for exponential decay
                initial_guess = [y_nonzero[0], 0.1, self.quiet_threshold]

                popt, _ = curve_fit(
                    self._exponential_decay_func,
                    x_nonzero,
                    y_nonzero,
                    p0=initial_guess,
                    maxfev=1000
                )

                # Calculate R-squared for exponential model
                y_pred_exp = self._exponential_decay_func(x_nonzero, *popt)
                ss_res = np.sum((y_nonzero - y_pred_exp) ** 2)
                ss_tot = np.sum((y_nonzero - np.mean(y_nonzero)) ** 2)
                exp_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                models.append({
                    'type': 'exponential',
                    'params': {'a': popt[0], 'b': popt[1], 'c': popt[2]},
                    'r_squared': exp_r_squared,
                    'function': lambda x: self._exponential_decay_func(x, *popt)
                })
            except Exception as e:
                logger.debug(f"Exponential regression failed for {canvas_id}: {e}")

        if not models:
            return ForecastResult(
                canvas_id=canvas_id,
                quiet_date=None,
                confidence=0.0,
                r_squared=0.0,
                days_to_quiet=None,
                current_trend='insufficient_data',
                model_params={}
            )

        # Choose the best model based on R-squared
        best_model = max(models, key=lambda m: m['r_squared'])

        # Determine current trend
        recent_sends = sends[-min(7, len(sends)):]  # Last week
        if len(recent_sends) >= 2:
            recent_trend = np.mean(np.diff(recent_sends))
            if recent_trend < -1:
                trend = 'declining'
            elif recent_trend > 1:
                trend = 'growing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        # Predict quiet date
        quiet_date = None
        days_to_quiet = None
        confidence = best_model['r_squared']

        if best_model['type'] == 'linear':
            # For linear model: solve slope * x + intercept = quiet_threshold
            slope = best_model['params']['slope']
            intercept = best_model['params']['intercept']

            if slope < 0:  # Declining trend
                days_to_threshold = (self.quiet_threshold - intercept) / slope
                if days_to_threshold > 0:
                    quiet_date = base_date + timedelta(days=int(days_to_threshold))
                    days_to_quiet = int(days_to_threshold) - len(daily_stats)

        elif best_model['type'] == 'exponential':
            # For exponential model: solve a * exp(-bx) + c = quiet_threshold
            a, b, c = best_model['params']['a'], best_model['params']['b'], best_model['params']['c']

            if a > 0 and b > 0 and (self.quiet_threshold - c) > 0 and (self.quiet_threshold - c) < a:
                days_to_threshold = -math.log((self.quiet_threshold - c) / a) / b
                if days_to_threshold > 0:
                    quiet_date = base_date + timedelta(days=int(days_to_threshold))
                    days_to_quiet = int(days_to_threshold) - len(daily_stats)

        # Adjust confidence based on data quality
        if len(daily_stats) < 14:
            confidence *= 0.7  # Reduce confidence for limited data

        if trend == 'growing':
            confidence *= 0.5  # Reduce confidence if trend is growing

        return ForecastResult(
            canvas_id=canvas_id,
            quiet_date=quiet_date,
            confidence=min(confidence, 1.0),
            r_squared=best_model['r_squared'],
            days_to_quiet=days_to_quiet,
            current_trend=trend,
            model_params=best_model['params']
        )


class QuietDatePredictor:
    """High-level interface for Canvas quiet date prediction."""

    def __init__(self, data_dir: Path = None, quiet_threshold: int = 5):
        """
        Initialize the predictor.

        Args:
            data_dir: Directory containing JSONL data files
            quiet_threshold: Daily sends below this are considered "quiet"
        """
        self.data_dir = data_dir or Path("data")
        self.forecaster = LinearDecayForecaster(quiet_threshold=quiet_threshold)

    def predict_all_canvases(self) -> List[ForecastResult]:
        """Predict quiet dates for all Canvases with data files."""
        results = []

        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return results

        # Find all JSONL files
        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} Canvas data files")

        for jsonl_file in jsonl_files:
            canvas_id = jsonl_file.stem  # Remove .jsonl extension
            try:
                result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)
                results.append(result)

                if result.quiet_date:
                    logger.info(f"Canvas {canvas_id}: Predicted quiet date {result.quiet_date} "
                              f"(confidence: {result.confidence:.2f}, trend: {result.current_trend})")
                else:
                    logger.debug(f"Canvas {canvas_id}: Could not predict quiet date "
                               f"(trend: {result.current_trend})")

            except Exception as e:
                logger.error(f"Error predicting quiet date for Canvas {canvas_id}: {e}")

        return results

    def generate_forecast_report(self) -> Dict[str, Any]:
        """Generate a comprehensive forecast report."""
        results = self.predict_all_canvases()

        # Categorize results
        predictable = [r for r in results if r.quiet_date is not None]
        unpredictable = [r for r in results if r.quiet_date is None]

        # Sort predictable by days to quiet
        predictable_soon = [r for r in predictable if r.days_to_quiet and r.days_to_quiet <= 30]
        predictable_later = [r for r in predictable if r.days_to_quiet and r.days_to_quiet > 30]

        # Trend analysis
        trend_counts = {}
        for result in results:
            trend_counts[result.current_trend] = trend_counts.get(result.current_trend, 0) + 1

        # Confidence distribution
        high_confidence = [r for r in predictable if r.confidence >= 0.7]
        medium_confidence = [r for r in predictable if 0.4 <= r.confidence < 0.7]
        low_confidence = [r for r in predictable if r.confidence < 0.4]

        report = {
            'summary': {
                'total_canvases': len(results),
                'predictable': len(predictable),
                'unpredictable': len(unpredictable),
                'going_quiet_soon': len(predictable_soon),  # Within 30 days
                'going_quiet_later': len(predictable_later)
            },
            'trends': trend_counts,
            'confidence_distribution': {
                'high': len(high_confidence),
                'medium': len(medium_confidence),
                'low': len(low_confidence)
            },
            'urgent_canvases': [
                {
                    'canvas_id': r.canvas_id,
                    'quiet_date': r.quiet_date.isoformat() if r.quiet_date else None,
                    'days_to_quiet': r.days_to_quiet,
                    'confidence': round(r.confidence, 3),
                    'trend': r.current_trend
                }
                for r in sorted(predictable_soon, key=lambda x: x.days_to_quiet or 999)[:10]
            ],
            'generated_at': datetime.now().isoformat()
        }

        return report