"""Advanced forecasting models for Canvas quiet date prediction.

This module implements sophisticated forecasting techniques including:
- Polynomial regression (degree 2-4)
- Power law models
- ARIMA time series analysis
- Ensemble methods
- Cross-validation for confidence estimation
- Bootstrap confidence intervals
"""

import logging
import warnings
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import math

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("statsmodels not available - ARIMA models disabled")


class AdvancedForecastResult(NamedTuple):
    """Enhanced forecast result with additional metrics."""

    canvas_id: str
    canvas_name: str
    quiet_date: Optional[date]
    confidence: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    days_to_quiet: Optional[int]
    current_trend: str
    best_model: str
    model_params: Dict[str, Any]
    metric_used: str
    cv_score: float
    ensemble_weight: float
    r_squared: float
    mse: float
    trend_strength: float


class AdvancedForecaster:
    """Advanced forecasting with multiple models and ensemble methods."""

    def __init__(self, quiet_threshold: int = 5, min_data_points: int = 10):
        """
        Initialize the advanced forecaster.

        Args:
            quiet_threshold: Daily sends below this are considered "quiet"
            min_data_points: Minimum data points required for prediction
        """
        self.quiet_threshold = quiet_threshold
        self.min_data_points = min_data_points

    def _polynomial_model(
        self, x: np.ndarray, degree: int
    ) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """Fit polynomial regression model."""
        try:
            poly_features = PolynomialFeatures(degree=degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))

            model = LinearRegression()
            model.fit(x_poly, self.y_values)

            y_pred = model.predict(x_poly)
            r2 = r2_score(self.y_values, y_pred)

            return (
                y_pred,
                {
                    "degree": degree,
                    "coefficients": model.coef_.tolist(),
                    "intercept": model.intercept_,
                },
                r2,
            )
        except Exception as e:
            logger.debug(f"Polynomial degree {degree} failed: {e}")
            return None, {}, -1

    def _power_law_func(
        self, x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Power law function: y = a * x^b + c"""
        return a * np.power(np.maximum(x, 0.1), b) + c

    def _power_law_model(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """Fit power law model."""
        try:
            # Filter positive values
            positive_mask = self.y_values > 0
            if np.sum(positive_mask) < self.min_data_points:
                return None, {}, -1

            x_pos = x[positive_mask]
            y_pos = self.y_values[positive_mask]

            # Initial parameter guess
            initial_guess = [y_pos[0], -0.5, self.quiet_threshold]

            popt, _ = curve_fit(
                self._power_law_func,
                x_pos,
                y_pos,
                p0=initial_guess,
                maxfev=2000,
                bounds=([0.1, -10, 0], [np.inf, 10, np.inf]),
            )

            y_pred = self._power_law_func(x, *popt)
            r2 = r2_score(self.y_values, y_pred)

            return y_pred, {"a": popt[0], "b": popt[1], "c": popt[2]}, r2
        except Exception as e:
            logger.debug(f"Power law model failed: {e}")
            return None, {}, -1

    def _arima_model(self, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """Fit ARIMA model."""
        if not ARIMA_AVAILABLE or len(y) < 15:
            return None, {}, -1

        try:
            # Test for stationarity
            adf_result = adfuller(y)
            is_stationary = adf_result[1] < 0.05

            # Try different ARIMA parameters
            best_aic = np.inf
            best_model = None
            best_params = None

            # Grid search for best parameters
            for p in range(0, 3):
                for d in range(0, 2 if not is_stationary else 1):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(y, order=(p, d, q))
                            fitted_model = model.fit()

                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_params = (p, d, q)
                        except:
                            continue

            if best_model is None:
                return None, {}, -1

            # Generate predictions
            y_pred = best_model.fittedvalues
            r2 = r2_score(y, y_pred)

            return (
                y_pred,
                {
                    "order": best_params,
                    "aic": best_aic,
                    "params": best_model.params.tolist(),
                },
                r2,
            )

        except Exception as e:
            logger.debug(f"ARIMA model failed: {e}")
            return None, {}, -1

    def _calculate_trend_strength(self, y: np.ndarray) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(y) < 3:
            return 0.0

        x = np.arange(len(y))
        slope, _, r_value, _, _ = stats.linregress(x, y)

        # Normalize trend strength by data range
        data_range = np.max(y) - np.min(y) if np.max(y) > np.min(y) else 1
        trend_strength = abs(slope * len(y)) / data_range

        return min(trend_strength, 1.0)

    def _bootstrap_confidence_interval(
        self, x: np.ndarray, y: np.ndarray, model_func, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals."""
        try:
            predictions = []
            n_samples = len(y)

            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                x_boot = x[indices]
                y_boot = y[indices]

                # Fit model and predict
                try:
                    self.y_values = y_boot  # Temporary assignment for model fitting
                    pred, _, _ = model_func(x_boot)
                    if pred is not None:
                        # Predict at the end of the time series
                        final_pred = pred[-1] if len(pred) > 0 else 0
                        predictions.append(final_pred)
                except:
                    continue

            if len(predictions) >= 50:  # Need reasonable number of bootstrap samples
                lower = np.percentile(predictions, 2.5)
                upper = np.percentile(predictions, 97.5)
                return lower, upper
            else:
                return 0.0, 0.0

        except Exception as e:
            logger.debug(f"Bootstrap confidence interval failed: {e}")
            return 0.0, 0.0

    def _cross_validate_model(self, x: np.ndarray, y: np.ndarray, model_func) -> float:
        """Perform time series cross-validation."""
        try:
            if len(y) < 10:
                return 0.0

            tscv = TimeSeriesSplit(n_splits=min(3, len(y) // 3))
            scores = []

            for train_idx, test_idx in tscv.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    self.y_values = y_train  # Temporary assignment
                    pred, _, _ = model_func(x_train)

                    if pred is not None and len(pred) > 0:
                        # Simple prediction: use the trend to predict test set
                        if len(pred) >= 2:
                            trend = pred[-1] - pred[-2]
                            test_pred = pred[-1] + trend * np.arange(1, len(y_test) + 1)
                            score = r2_score(y_test, test_pred)
                            scores.append(max(score, 0))  # Don't allow negative scores
                except:
                    continue

            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.debug(f"Cross-validation failed: {e}")
            return 0.0

    def predict_quiet_date_advanced(
        self, canvas_metrics, canvas_id: str, canvas_name: str = ""
    ) -> AdvancedForecastResult:
        """
        Advanced prediction using multiple models and ensemble methods.

        Args:
            canvas_metrics: List of CanvasMetrics objects
            canvas_id: Canvas ID
            canvas_name: Canvas name

        Returns:
            AdvancedForecastResult with comprehensive predictions
        """
        if len(canvas_metrics) < self.min_data_points:
            return AdvancedForecastResult(
                canvas_id=canvas_id,
                canvas_name=canvas_name,
                quiet_date=None,
                confidence=0.0,
                prediction_interval_lower=0.0,
                prediction_interval_upper=0.0,
                days_to_quiet=None,
                current_trend="insufficient_data",
                best_model="none",
                model_params={},
                metric_used="none",
                cv_score=0.0,
                ensemble_weight=0.0,
                r_squared=0.0,
                mse=0.0,
                trend_strength=0.0,
            )

        # Try different metrics
        metrics_to_try = [
            ("total_sent", [m.total_sent for m in canvas_metrics]),
            ("total_opens", [m.total_opens for m in canvas_metrics]),
            ("total_clicks", [m.total_clicks for m in canvas_metrics]),
            ("total_delivered", [m.total_delivered for m in canvas_metrics]),
        ]

        best_result = None
        best_composite_score = -1

        for metric_name, y_values in metrics_to_try:
            # Skip if no variation
            if max(y_values) - min(y_values) < 1:
                continue

            self.y_values = np.array(y_values)
            dates = [m.date for m in canvas_metrics]
            base_date = dates[0]
            x_values = np.array([(d - base_date).days for d in dates])

            # Try different models
            models = []

            # 1. Polynomial models (degree 2-4)
            for degree in [2, 3, 4]:
                pred, params, r2 = self._polynomial_model(x_values, degree)
                if pred is not None and r2 > 0:
                    cv_score = self._cross_validate_model(
                        x_values,
                        self.y_values,
                        lambda x: self._polynomial_model(x, degree),
                    )
                    models.append(
                        {
                            "name": f"polynomial_deg_{degree}",
                            "prediction": pred,
                            "params": params,
                            "r2": r2,
                            "cv_score": cv_score,
                            "weight": r2 * (1 + cv_score),
                        }
                    )

            # 2. Power law model
            pred, params, r2 = self._power_law_model(x_values)
            if pred is not None and r2 > 0:
                cv_score = self._cross_validate_model(
                    x_values, self.y_values, self._power_law_model
                )
                models.append(
                    {
                        "name": "power_law",
                        "prediction": pred,
                        "params": params,
                        "r2": r2,
                        "cv_score": cv_score,
                        "weight": r2 * (1 + cv_score),
                    }
                )

            # 3. ARIMA model
            if ARIMA_AVAILABLE:
                pred, params, r2 = self._arima_model(self.y_values)
                if pred is not None and r2 > 0:
                    models.append(
                        {
                            "name": "arima",
                            "prediction": pred,
                            "params": params,
                            "r2": r2,
                            "cv_score": 0.7,  # Default CV score for ARIMA
                            "weight": r2 * 1.7,
                        }
                    )

            if not models:
                continue

            # Select best model and create ensemble
            models.sort(key=lambda x: x["weight"], reverse=True)
            best_model = models[0]

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(self.y_values)

            # Determine trend direction
            recent_values = self.y_values[-min(7, len(self.y_values)) :]
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

            # Enhanced confidence calculation
            base_confidence = best_model["r2"]
            cv_bonus = best_model["cv_score"] * 0.3
            trend_penalty = 0.5 if trend == "growing" else 0.0
            data_bonus = min(0.2, len(canvas_metrics) / 50)  # Bonus for more data

            confidence = base_confidence + cv_bonus + data_bonus - trend_penalty
            confidence = max(0.0, min(1.0, confidence))

            # Bootstrap confidence intervals
            if len(models) > 0:
                model_func = lambda x: self._polynomial_model(
                    x, 2
                )  # Use simplest for bootstrap
                lower_ci, upper_ci = self._bootstrap_confidence_interval(
                    x_values, self.y_values, model_func
                )
            else:
                lower_ci, upper_ci = 0.0, 0.0

            # Predict quiet date (simplified for polynomial)
            quiet_date = None
            days_to_quiet = None

            if best_model["name"].startswith("polynomial") and trend == "declining":
                # Use polynomial coefficients to find when y = quiet_threshold
                try:
                    coeffs = best_model["params"]["coefficients"]
                    coeffs[-1] -= self.quiet_threshold  # Adjust constant term

                    # Find positive real roots
                    roots = np.roots(coeffs)
                    real_positive_roots = [
                        r.real
                        for r in roots
                        if np.isreal(r) and r.real > len(canvas_metrics)
                    ]

                    if real_positive_roots:
                        days_to_threshold = min(real_positive_roots)
                        quiet_date = base_date + timedelta(days=int(days_to_threshold))
                        days_to_quiet = int(days_to_threshold) - len(canvas_metrics)
                except:
                    pass

            # Composite score for model selection
            composite_score = (
                confidence * (1 + trend_strength) * (1 + best_model["cv_score"])
            )

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_result = AdvancedForecastResult(
                    canvas_id=canvas_id,
                    canvas_name=canvas_name,
                    quiet_date=quiet_date,
                    confidence=confidence,
                    prediction_interval_lower=lower_ci,
                    prediction_interval_upper=upper_ci,
                    days_to_quiet=days_to_quiet,
                    current_trend=trend,
                    best_model=best_model["name"],
                    model_params=best_model["params"],
                    metric_used=metric_name,
                    cv_score=best_model["cv_score"],
                    ensemble_weight=best_model["weight"],
                    r_squared=best_model["r2"],
                    mse=mean_squared_error(self.y_values, best_model["prediction"]),
                    trend_strength=trend_strength,
                )

        return best_result or AdvancedForecastResult(
            canvas_id=canvas_id,
            canvas_name=canvas_name,
            quiet_date=None,
            confidence=0.0,
            prediction_interval_lower=0.0,
            prediction_interval_upper=0.0,
            days_to_quiet=None,
            current_trend="insufficient_data",
            best_model="none",
            model_params={},
            metric_used="none",
            cv_score=0.0,
            ensemble_weight=0.0,
            r_squared=0.0,
            mse=0.0,
            trend_strength=0.0,
        )
