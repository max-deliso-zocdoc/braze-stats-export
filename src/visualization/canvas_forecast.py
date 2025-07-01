"""Visualization utilities for Canvas quiet date forecasting."""

import re
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forecasting.linear_decay import CanvasMetrics, StepBasedForecaster, ForecastResult, MultiForecastResult


def validate_canvas_data(
    metrics: List[CanvasMetrics], metric_col: str = "total_sent"
) -> bool:
    """
    Validate canvas data before attempting regression.

    Args:
        metrics: List of CanvasMetrics objects
        metric_col: Column name to validate

    Returns:
        True if data is valid for regression, False otherwise
    """
    if not metrics or len(metrics) < 2:
        return False

    # Convert to DataFrame
    df = canvas_metrics_to_dataframe(metrics)
    if df.empty:
        return False

    # Get the metric values
    y = df[metric_col]

    # Check for sufficient data points
    if len(y) < 2:
        return False

    # Check for variation in data
    if y.var() == 0:
        return False

    # Check for invalid values
    if y.isna().any() or np.isinf(y).any():
        return False

    # Check for all zeros (common in inactive canvases)
    if (y == 0).all():
        return False

    return True


def canvas_metrics_to_dataframe(metrics: List[CanvasMetrics]) -> pd.DataFrame:
    """Convert CanvasMetrics list to pandas DataFrame for analysis."""
    if not metrics:
        return pd.DataFrame()

    data = []
    for metric in metrics:
        data.append(
            {
                "date": metric.date,
                "total_sent": metric.total_sent,
                "total_opens": metric.total_opens,
                "total_unique_opens": metric.total_unique_opens,
                "total_clicks": metric.total_clicks,
                "total_unique_clicks": metric.total_unique_clicks,
                "total_delivered": metric.total_delivered,
                "total_bounces": metric.total_bounces,
                "total_unsubscribes": metric.total_unsubscribes,
                "active_steps": metric.active_steps,
                "active_channels": metric.active_channels,
            }
        )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_all_forecast_results(metrics: List[CanvasMetrics], quiet_threshold: int = 5) -> Optional[MultiForecastResult]:
    """
    Use the existing forecasting logic to get all forecast results.

    Args:
        metrics: List of CanvasMetrics objects
        quiet_threshold: Daily sends below this are considered "quiet"

    Returns:
        MultiForecastResult if successful, None otherwise
    """
    if not metrics or len(metrics) < 7:  # Minimum data points for forecasting
        return None

    # Create a temporary forecaster instance
    forecaster = StepBasedForecaster(quiet_threshold=quiet_threshold, min_data_points=7)

    # We need to simulate the forecasting process since we already have the metrics
    # The forecaster expects to load from files, but we can work around this

    # Try different metrics to find the best one for forecasting
    metrics_to_try = [
        ("total_sent", [m.total_sent for m in metrics]),
        ("total_opens", [m.total_opens for m in metrics]),
        ("total_clicks", [m.total_clicks for m in metrics]),
        ("total_delivered", [m.total_delivered for m in metrics]),
    ]

    all_forecasts: List[ForecastResult] = []
    total_models_tried = 0
    successful_models = 0

    for metric_name, y_values in metrics_to_try:
        # Skip if no variation in the data
        if max(y_values) - min(y_values) < 1:
            continue

        # Convert dates to days since start
        dates = [m.date for m in metrics]
        base_date = dates[0]
        x_values = np.array([(d - base_date).days for d in dates])
        y_values_array = np.array(y_values)

        # Filter out days with zero values for exponential model
        non_zero_mask = y_values_array > 0

        # Try different models
        models = []

        # 1. Linear regression
        total_models_tried += 1
        try:
            from scipy import stats
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
            successful_models += 1
        except Exception:
            continue

        # 2. Exponential decay (if we have enough non-zero data)
        if np.sum(non_zero_mask) >= 7:
            total_models_tried += 1
            try:
                x_nonzero = x_values[non_zero_mask]
                y_nonzero = y_values_array[non_zero_mask]

                # Use log-space fitting for better numerical stability
                popt, exp_r_squared = forecaster._fit_exponential_log_space(
                    x_nonzero, y_nonzero
                )
                log_a, b, c = popt

                # Convert log_a back to a for compatibility
                a = np.exp(log_a)

                models.append(
                    {
                        "type": "exponential",
                        "params": {"a": a, "b": b, "c": c, "log_a": log_a},
                        "r_squared": exp_r_squared,
                        "function": lambda x: forecaster._log_exponential_decay_func(
                            x, log_a, b, c
                        ),
                    }
                )
                successful_models += 1
            except Exception:
                continue

        # Process all successful models for this metric
        for model in models:
            # Determine current trend
            recent_values = y_values_array[
                -min(7, len(y_values_array)) :
            ]  # Last week
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
            confidence = model["r_squared"]

            # Reduce confidence for growing trends since they don't make sense for quiet date prediction
            if trend == "growing":
                confidence *= 0.3  # Significantly reduce confidence for growing trends
            elif trend == "stable":
                confidence *= 0.7  # Reduce confidence for stable trends as well

            if model["type"] == "linear":
                # For linear model: solve slope * x + intercept = quiet_threshold
                slope = model["params"]["slope"]
                intercept = model["params"]["intercept"]

                if slope < 0:  # Declining trend
                    days_to_threshold = (quiet_threshold - intercept) / slope
                    if days_to_threshold > 0:
                        quiet_date = base_date + timedelta(
                            days=int(days_to_threshold)
                        )
                        days_to_quiet = int(days_to_threshold) - len(metrics)

            elif model["type"] == "exponential":
                # For exponential model: solve a * exp(-bx) + c = quiet_threshold
                a, b, c = (
                    model["params"]["a"],
                    model["params"]["b"],
                    model["params"]["c"],
                )

                if (
                    a > 0
                    and b > 0
                    and (quiet_threshold - c) > 0
                    and (quiet_threshold - c) < a
                ):
                    try:
                        import math
                        days_to_threshold = (
                            -math.log((quiet_threshold - c) / a) / b
                        )
                        if days_to_threshold > 0:
                            quiet_date = base_date + timedelta(
                                days=int(days_to_threshold)
                            )
                            days_to_quiet = int(days_to_threshold) - len(metrics)
                    except (ValueError, ZeroDivisionError):
                        pass

            forecast_result = ForecastResult(
                canvas_id=metrics[0].canvas_id,
                canvas_name="",
                quiet_date=quiet_date,
                confidence=min(confidence, 1.0),
                r_squared=model["r_squared"],
                days_to_quiet=days_to_quiet,
                current_trend=trend,
                model_params=model["params"],
                metric_used=metric_name,
                model_type=model["type"],
            )

            all_forecasts.append(forecast_result)

    # Find the best forecast based on R-squared
    best_forecast = None
    if all_forecasts:
        best_forecast = max(all_forecasts, key=lambda f: f.r_squared)

    return MultiForecastResult(
        canvas_id=metrics[0].canvas_id,
        canvas_name="",
        forecasts=all_forecasts,
        best_forecast=best_forecast,
        total_models_tried=total_models_tried,
        successful_models=successful_models,
    )


def _get_forecast_result(metrics: List[CanvasMetrics], quiet_threshold: int = 5) -> Optional[ForecastResult]:
    """
    Use the existing forecasting logic to get forecast results.
    This function returns only the best prediction for backward compatibility.

    Args:
        metrics: List of CanvasMetrics objects
        quiet_threshold: Daily sends below this are considered "quiet"

    Returns:
        ForecastResult if successful, None otherwise
    """
    multi_result = _get_all_forecast_results(metrics, quiet_threshold)
    return multi_result.best_forecast if multi_result else None


def _generate_predictions(
    forecast_result: ForecastResult,
    df: pd.DataFrame,
    horizon_days: int = 365
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate future predictions based on the forecast result.

    Args:
        forecast_result: The forecast result from the sophisticated forecasting logic
        df: DataFrame with historical data
        horizon_days: Number of days to predict into the future (max 365)

    Returns:
        Tuple of (prediction band DataFrame, future dates array)
    """
    # Enforce maximum horizon of one year
    horizon_days = min(horizon_days, 365)

    if not forecast_result.model_params:
        # Return empty predictions if no model
        future_dates = pd.date_range(
            start=df["date"].max() + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D"
        )
        empty_pred = pd.DataFrame({
            "mean": [np.nan] * horizon_days,
            "mean_ci_lower": [np.nan] * horizon_days,
            "mean_ci_upper": [np.nan] * horizon_days,
        })
        return empty_pred, future_dates

    # Convert dates to days since start
    base_date = df["date"].min()
    df["days_since_start"] = (df["date"] - base_date).dt.days

    # Generate future days
    future_days = np.arange(
        df["days_since_start"].max() + 1,
        df["days_since_start"].max() + horizon_days + 1,
    )

    # Generate predictions based on model type
    if forecast_result.model_params.get("type") == "linear" or "slope" in forecast_result.model_params:
        # Linear model
        slope = forecast_result.model_params.get("slope", 0)
        intercept = forecast_result.model_params.get("intercept", 0)

        predictions = slope * future_days + intercept

        # Simple confidence interval (could be improved)
        std_err = np.std(df[forecast_result.metric_used]) * 0.1  # Rough estimate
        ci_lower = predictions - 1.96 * std_err
        ci_upper = predictions + 1.96 * std_err

    elif "log_a" in forecast_result.model_params:
        # Exponential model
        log_a = forecast_result.model_params["log_a"]
        b = forecast_result.model_params["b"]
        c = forecast_result.model_params["c"]

        predictions = np.exp(log_a) * np.exp(-b * future_days) + c

        # Simple confidence interval
        std_err = np.std(df[forecast_result.metric_used]) * 0.1
        ci_lower = predictions - 1.96 * std_err
        ci_upper = predictions + 1.96 * std_err

    else:
        # Fallback
        predictions = np.full(horizon_days, np.nan)
        ci_lower = np.full(horizon_days, np.nan)
        ci_upper = np.full(horizon_days, np.nan)

    # Create prediction band DataFrame
    pred_band = pd.DataFrame({
        "mean": predictions,
        "mean_ci_lower": ci_lower,
        "mean_ci_upper": ci_upper,
    })

    # Generate future dates
    future_dates = base_date + pd.to_timedelta(future_days, unit="D")

    return pred_band, future_dates


def plot_canvas_forecast(
    metrics: List[CanvasMetrics],
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    horizon_days: int = 365,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> Optional[date]:
    """
    Create a comprehensive plot showing Canvas data, linear fit, and quiet date forecast.

    Args:
        metrics: List of CanvasMetrics objects
        metric_col: Column name to analyze (default: "total_sent")
        quiet_threshold: Daily sends below this are considered "quiet"
        horizon_days: Number of days to predict into the future (max 365)
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Predicted quiet date (if any) or None
    """
    # Enforce maximum horizon of one year
    horizon_days = min(horizon_days, 365)

    if not metrics:
        print("No metrics data provided")
        return None

    # Convert to DataFrame
    df = canvas_metrics_to_dataframe(metrics)
    if df.empty:
        print("No valid data to plot")
        return None

    # Create plot
    plt.figure(figsize=figsize)

    # Scatter of actual points
    plt.scatter(
        df["date"], df[metric_col], s=25, alpha=0.8, label="actual", color="blue"
    )

    try:
        # Use the sophisticated forecasting logic
        forecast_result = _get_all_forecast_results(metrics, quiet_threshold)

        if forecast_result and forecast_result.best_forecast:
            # Generate predictions
            pred_band, future_dates = _generate_predictions(forecast_result.best_forecast, df, horizon_days)

            # Regression mean line
            plt.plot(future_dates, pred_band["mean"], lw=2, label="forecast", color="red")

            # Confidence band
            plt.fill_between(
                future_dates,
                pred_band["mean_ci_lower"],
                pred_band["mean_ci_upper"],
                alpha=0.2,
                label="95% CI",
                color="red",
            )

            if forecast_result.best_forecast.quiet_date is not None:
                plt.axvline(
                    forecast_result.best_forecast.quiet_date,
                    ls=":",
                    color="green",
                    label=f"predicted quiet: {forecast_result.best_forecast.quiet_date}",
                )

            # Add model statistics
            r_squared = forecast_result.best_forecast.r_squared
            model_type = forecast_result.best_forecast.model_type
            trend = forecast_result.best_forecast.current_trend

            plt.title(
                f"{metric_col}: Canvas Forecast ({model_type}, R²={r_squared:.3f}, trend={trend})"
            )

            quiet_date = forecast_result.best_forecast.quiet_date
        else:
            plt.title(f"{metric_col}: Canvas Data (Insufficient data for forecasting)")
            quiet_date = None

    except Exception as e:
        # Handle errors gracefully
        plt.title(f"{metric_col}: Canvas Data (Forecasting Failed: {str(e)[:50]}...)")
        quiet_date = None
        print(f"Forecasting failed: {e}")

    # Threshold line (always show)
    plt.axhline(
        quiet_threshold,
        ls="--",
        color="orange",
        label=f"quiet threshold = {quiet_threshold}",
    )

    plt.ylabel(metric_col)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set x-axis limits to maximum one year from the last data point
    last_data_date = df["date"].max()
    max_future_date = last_data_date + pd.Timedelta(days=365)
    plt.xlim(df["date"].min(), max_future_date)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return quiet_date


def plot_multiple_canvases(
    canvas_data: dict[str, List[CanvasMetrics]],
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    max_canvases: int = 9,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """
    Create subplots for multiple canvases.

    Args:
        canvas_data: Dict mapping canvas_id to list of CanvasMetrics
        metric_col: Column name to analyze
        quiet_threshold: Daily sends below this are considered "quiet"
        max_canvases: Maximum number of canvases to plot
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    if not canvas_data:
        print("No canvas data provided")
        return

    # Limit number of canvases
    canvas_items = list(canvas_data.items())[:max_canvases]
    n_canvases = len(canvas_items)

    # Calculate subplot layout
    cols = min(3, n_canvases)
    rows = (n_canvases + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_canvases == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()

    for i, (canvas_id, metrics) in enumerate(canvas_items):
        if i >= len(axes):
            break

        ax = axes[i]

        if not metrics:
            ax.text(
                0.5,
                0.5,
                f"No data for {canvas_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Canvas: {canvas_id[:20]}...")
            continue

        # Convert to DataFrame
        df = canvas_metrics_to_dataframe(metrics)
        if df.empty:
            ax.text(
                0.5,
                0.5,
                f"No valid data for {canvas_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Canvas: {canvas_id[:20]}...")
            continue

        # Use sophisticated forecasting
        try:
            forecast_result = _get_all_forecast_results(metrics, quiet_threshold)

            if forecast_result and forecast_result.best_forecast:
                pred_band, future_dates = _generate_predictions(forecast_result.best_forecast, df)
                quiet_date = forecast_result.best_forecast.quiet_date

                # Plot
                ax.scatter(df["date"], df[metric_col], s=15, alpha=0.7, color="blue")
                ax.plot(future_dates, pred_band["mean"], lw=1.5, color="red")
                ax.fill_between(
                    future_dates,
                    pred_band["mean_ci_lower"],
                    pred_band["mean_ci_upper"],
                    alpha=0.2,
                    color="red",
                )
                ax.axhline(quiet_threshold, ls="--", color="orange", alpha=0.7)

                if quiet_date is not None:
                    ax.axvline(quiet_date, ls=":", color="green", alpha=0.7)

                # Add statistics
                r_squared = forecast_result.best_forecast.r_squared
                model_type = forecast_result.best_forecast.model_type
                trend = forecast_result.best_forecast.current_trend
                ax.set_title(f"{canvas_id[:20]}...\n{model_type}, R²={r_squared:.2f}, {trend}")

            else:
                # No valid forecast
                ax.scatter(df["date"], df[metric_col], s=15, alpha=0.7, color="blue")
                ax.axhline(quiet_threshold, ls="--", color="orange", alpha=0.7)
                ax.set_title(f"{canvas_id[:20]}...\nInsufficient data")

        except Exception as e:
            # Handle errors gracefully
            ax.scatter(df["date"], df[metric_col], s=15, alpha=0.7, color="blue")
            ax.axhline(quiet_threshold, ls="--", color="orange", alpha=0.7)
            ax.set_title(f"{canvas_id[:20]}...\nError: {str(e)[:30]}...")
            print(f"Forecasting failed for {canvas_id}: {e}")

        ax.set_ylabel(metric_col)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_canvases, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Multi-canvas plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-_\.]+", "", name)
    return name[:80]  # limit length


def create_forecast_report_plots(
    canvas_data: dict[str, List[CanvasMetrics]],
    output_dir: Path,
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    name_map: Optional[dict] = None,
    show_plot: bool = True,
) -> None:
    """
    Create and save forecast plots for multiple canvases, using canvas name in filename if available.

    Args:
        canvas_data: Dict mapping canvas_id to list of CanvasMetrics
        output_dir: Directory to save plots
        metric_col: Column name to analyze
        quiet_threshold: Daily sends below this are considered "quiet"
        name_map: Optional dict mapping canvas_id to canvas name
        show_plot: Whether to display the multi-canvas overview plot
    """
    output_dir.mkdir(exist_ok=True)
    name_map = name_map or {}

    # Create individual plots for each canvas
    for canvas_id, metrics in canvas_data.items():
        if not metrics:
            continue
        try:
            canvas_name = name_map.get(canvas_id, "")
            if canvas_name:
                fname = sanitize_filename(canvas_name)
            else:
                fname = canvas_id
            plot_path = output_dir / f"{fname}_forecast.png"
            plot_canvas_forecast(
                metrics,
                metric_col=metric_col,
                quiet_threshold=quiet_threshold,
                save_path=plot_path,
                show_plot=show_plot,
            )
        except Exception as e:
            print(f"Error creating plot for {canvas_id}: {e}")

    print(f"Forecast plots saved to {output_dir}")


def plot_canvas_forecast_all_models(
    metrics: List[CanvasMetrics],
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    horizon_days: int = 365,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> Optional[MultiForecastResult]:
    """
    Create a comprehensive plot showing ALL Canvas data sources and ALL forecast predictions.

    Args:
        metrics: List of CanvasMetrics objects
        metric_col: Column name to analyze (default: "total_sent")
        quiet_threshold: Daily sends below this are considered "quiet"
        horizon_days: Number of days to predict into the future (max 365)
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot

    Returns:
        MultiForecastResult with all predictions (if any) or None
    """
    # Enforce maximum horizon of one year
    horizon_days = min(horizon_days, 365)

    if not metrics:
        print("No metrics data provided")
        return None

    # Convert to DataFrame
    df = canvas_metrics_to_dataframe(metrics)
    if df.empty:
        print("No valid data to plot")
        return None

    # Create plot
    plt.figure(figsize=figsize)

    # Define colors for different metrics
    metric_colors = {
        'total_sent': 'blue',
        'total_opens': 'green',
        'total_clicks': 'red',
        'total_delivered': 'purple',
        'total_unique_opens': 'orange',
        'total_unique_clicks': 'brown',
        'total_bounces': 'pink',
        'total_unsubscribes': 'gray',
        'active_steps': 'olive',
        'active_channels': 'cyan'
    }

    # Define the metrics we want to show (both data and predictions)
    metrics_to_show = ['total_sent', 'total_opens', 'total_clicks', 'total_delivered']

    # Plot historical data for all metrics
    for metric in metrics_to_show:
        if metric in df.columns:
            color = metric_colors.get(metric, 'black')
            plt.scatter(
                df["date"],
                df[metric],
                s=20,
                alpha=0.6,
                label=f"historical {metric}",
                color=color,
                marker='o'
            )

    try:
        # Use the sophisticated forecasting logic to get all predictions
        multi_forecast_result = _get_all_forecast_results(metrics, quiet_threshold)

        if multi_forecast_result and multi_forecast_result.forecasts:
            # Plot each forecast with colors matching their data source
            linestyles = ['-', '--', '-.', ':']

            # Group forecasts by metric to assign consistent colors
            forecasts_by_metric: dict[str, list[ForecastResult]] = {}
            for forecast in multi_forecast_result.forecasts:
                if forecast.metric_used not in forecasts_by_metric:
                    forecasts_by_metric[forecast.metric_used] = []
                forecasts_by_metric[forecast.metric_used].append(forecast)

            for metric, forecasts in forecasts_by_metric.items():
                base_color = metric_colors.get(metric, 'black')

                for i, forecast in enumerate(forecasts):
                    if forecast.r_squared > 0:
                        # Generate predictions for this model
                        pred_band, future_dates = _generate_predictions(forecast, df, horizon_days)

                        # Use the same color as the data source, but different line style
                        linestyle = linestyles[i % len(linestyles)]

                        # Make the prediction line slightly darker than the data points
                        import matplotlib.colors as mcolors
                        pred_color = mcolors.to_rgb(base_color)
                        pred_color = tuple(max(0, min(1, c * 0.8)) for c in pred_color)  # Darken by 20%

                        # Plot the forecast line
                        plt.plot(
                            future_dates,
                            pred_band["mean"],
                            lw=2,
                            linestyle=linestyle,
                            color=pred_color,
                            label=f"{forecast.model_type} ({forecast.metric_used}, R²={forecast.r_squared:.2f})"
                        )

                        # Plot confidence band with transparency
                        plt.fill_between(
                            future_dates,
                            pred_band["mean_ci_lower"],
                            pred_band["mean_ci_upper"],
                            alpha=0.1,
                            color=pred_color,
                        )

                        # Plot quiet date prediction if available
                        if forecast.quiet_date is not None:
                            plt.axvline(
                                forecast.quiet_date,
                                ls=":",
                                color=pred_color,
                                alpha=0.8,
                                linewidth=2,
                                label=f"quiet: {forecast.quiet_date} ({forecast.model_type}, {forecast.metric_used})"
                            )

            # Add summary statistics
            best_forecast = multi_forecast_result.best_forecast
            if best_forecast:
                plt.title(
                    f"Canvas Forecast - {len(multi_forecast_result.forecasts)} models "
                    f"(Best: {best_forecast.model_type} on {best_forecast.metric_used}, "
                    f"R²={best_forecast.r_squared:.3f}, trend={best_forecast.current_trend})"
                )
            else:
                plt.title(f"Canvas Forecast - {len(multi_forecast_result.forecasts)} models")

            # Add legend with better organization
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)

        else:
            plt.title("Canvas Data (Insufficient data for forecasting)")

    except Exception as e:
        # Handle errors gracefully
        plt.title(f"Canvas Data (Forecasting Failed: {str(e)[:50]}...)")
        print(f"Forecasting failed: {e}")

    # Threshold line (always show)
    plt.axhline(
        quiet_threshold,
        ls="--",
        color="orange",
        label=f"quiet threshold = {quiet_threshold}",
        linewidth=2,
        alpha=0.8
    )

    plt.ylabel("Metric Values")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)

    # Set x-axis limits to maximum one year from the last data point
    last_data_date = df["date"].max()
    max_future_date = last_data_date + pd.Timedelta(days=365)
    plt.xlim(df["date"].min(), max_future_date)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return multi_forecast_result
