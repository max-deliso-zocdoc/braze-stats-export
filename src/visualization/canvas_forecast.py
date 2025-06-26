"""Visualization utilities for Canvas quiet date forecasting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import timedelta
from typing import List, Optional, Tuple
from pathlib import Path
import re

from ..forecasting.linear_decay import CanvasMetrics


def validate_canvas_data(metrics: List[CanvasMetrics], metric_col: str = "total_sent") -> bool:
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
        data.append({
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
        })

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fit_linear_model(df: pd.DataFrame, metric_col: str = "total_sent") -> Tuple[sm.OLS, pd.DataFrame]:
    """Fit linear regression model to the data."""
    df = df.sort_values("date").copy()
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    X = sm.add_constant(df["days_since_start"])
    y = df[metric_col]

    # Check for valid data before fitting
    if len(y) < 2:
        raise ValueError("Insufficient data points for regression")

    if y.var() == 0:
        raise ValueError("No variation in data (all values are the same)")

    if y.isna().any() or np.isinf(y).any():
        raise ValueError("Data contains NaN or infinite values")

    linmod = sm.OLS(y, X).fit()

    # Check for valid R²
    if np.isnan(linmod.rsquared) or np.isinf(linmod.rsquared):
        raise ValueError("Invalid R² value calculated")

    return linmod, df


def predict_future(linmod: sm.OLS, df: pd.DataFrame, horizon_days: int = 180) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate future predictions with confidence intervals."""
    future_days = np.arange(
        df["days_since_start"].max() + 1,
        df["days_since_start"].max() + horizon_days + 1
    )

    X_future = sm.add_constant(future_days)
    pred = linmod.get_prediction(X_future)
    pred_band = pred.summary_frame(alpha=0.05)  # 95% CI on the mean

    future_dates = df["date"].min() + pd.to_timedelta(future_days, unit="D")

    return pred_band, future_dates


def calculate_quiet_date(linmod: sm.OLS, df: pd.DataFrame, quiet_threshold: int = 5) -> Optional[pd.Timestamp]:
    """Calculate the predicted quiet date based on linear regression."""
    slope, intercept = linmod.params[1], linmod.params[0]

    if slope < 0 and (quiet_threshold - intercept) / slope > 0:
        quiet_offset = (quiet_threshold - intercept) / slope
        quiet_date = df["date"].min() + timedelta(days=quiet_offset)
        return pd.Timestamp(quiet_date)
    else:
        return None  # model says "never" or series is growing


def plot_canvas_forecast(
    metrics: List[CanvasMetrics],
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    horizon_days: int = 180,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> Optional[pd.Timestamp]:
    """
    Create a comprehensive plot showing Canvas data, linear fit, and quiet date forecast.

    Args:
        metrics: List of CanvasMetrics objects
        metric_col: Column name to analyze (default: "total_sent")
        quiet_threshold: Daily sends below this are considered "quiet"
        horizon_days: Number of days to predict into the future
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Predicted quiet date (if any) or None
    """
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
    plt.scatter(df["date"], df[metric_col], s=25, alpha=0.8, label="actual", color="blue")

    try:
        # Fit model
        linmod, df = fit_linear_model(df, metric_col)

        # Generate predictions
        pred_band, future_dates = predict_future(linmod, df, horizon_days)

        # Calculate quiet date
        quiet_date = calculate_quiet_date(linmod, df, quiet_threshold)

        # Regression mean line
        plt.plot(future_dates, pred_band["mean"], lw=2, label="linear fit", color="red")

        # Confidence band
        plt.fill_between(
            future_dates,
            pred_band["mean_ci_lower"],
            pred_band["mean_ci_upper"],
            alpha=0.2,
            label="95% CI (mean)",
            color="red"
        )

        if quiet_date is not None:
            plt.axvline(quiet_date, ls=":", color="green",
                       label=f"predicted quiet: {quiet_date.date()}")

        # Add model statistics
        r_squared = linmod.rsquared
        slope = linmod.params[1]
        intercept = linmod.params[0]

        plt.title(f"{metric_col}: Canvas Forecast (R²={r_squared:.3f}, slope={slope:.3f})")

    except (ValueError, np.linalg.LinAlgError) as e:
        # Handle regression errors gracefully
        plt.title(f"{metric_col}: Canvas Data (Regression Failed: {str(e)[:50]}...)")
        quiet_date = None
        print(f"Regression failed: {e}")

    # Threshold line (always show)
    plt.axhline(quiet_threshold, ls="--", color="orange",
                label=f"quiet threshold = {quiet_threshold}")

    plt.ylabel(metric_col)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    save_path: Optional[Path] = None
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
            ax.text(0.5, 0.5, f"No data for {canvas_id}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Canvas: {canvas_id[:20]}...")
            continue

        # Convert to DataFrame
        df = canvas_metrics_to_dataframe(metrics)
        if df.empty:
            ax.text(0.5, 0.5, f"No valid data for {canvas_id}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Canvas: {canvas_id[:20]}...")
            continue

        # Fit model
        try:
            linmod, df = fit_linear_model(df, metric_col)
            pred_band, future_dates = predict_future(linmod, df)
            quiet_date = calculate_quiet_date(linmod, df, quiet_threshold)

            # Plot
            ax.scatter(df["date"], df[metric_col], s=15, alpha=0.7, color="blue")
            ax.plot(future_dates, pred_band["mean"], lw=1.5, color="red")
            ax.fill_between(future_dates, pred_band["mean_ci_lower"],
                          pred_band["mean_ci_upper"], alpha=0.2, color="red")
            ax.axhline(quiet_threshold, ls="--", color="orange", alpha=0.7)

            if quiet_date is not None:
                ax.axvline(quiet_date, ls=":", color="green", alpha=0.7)

            # Add statistics
            r_squared = linmod.rsquared
            slope = linmod.params[1]
            ax.set_title(f"{canvas_id[:20]}...\nR²={r_squared:.2f}, slope={slope:.2f}")

        except (ValueError, np.linalg.LinAlgError) as e:
            # Handle regression errors gracefully
            ax.scatter(df["date"], df[metric_col], s=15, alpha=0.7, color="blue")
            ax.axhline(quiet_threshold, ls="--", color="orange", alpha=0.7)
            ax.set_title(f"{canvas_id[:20]}...\nRegression failed: {str(e)[:30]}...")
            print(f"Regression failed for {canvas_id}: {e}")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:30]}...",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Canvas: {canvas_id[:20]}...")

        ax.set_ylabel(metric_col)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_canvases, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Multi-canvas plot saved to {save_path}")

    plt.show()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    name = name.strip().replace(' ', '_')
    name = re.sub(r'[^\w\-_\.]+', '', name)
    return name[:80]  # limit length


def create_forecast_report_plots(
    canvas_data: dict[str, List[CanvasMetrics]],
    output_dir: Path,
    metric_col: str = "total_sent",
    quiet_threshold: int = 5,
    name_map: dict = None
) -> None:
    """
    Create and save forecast plots for multiple canvases, using canvas name in filename if available.

    Args:
        canvas_data: Dict mapping canvas_id to list of CanvasMetrics
        output_dir: Directory to save plots
        metric_col: Column name to analyze
        quiet_threshold: Daily sends below this are considered "quiet"
        name_map: Optional dict mapping canvas_id to canvas name
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
                show_plot=False
            )
        except Exception as e:
            print(f"Error creating plot for {canvas_id}: {e}")

    # Create multi-canvas overview plot
    try:
        multi_plot_path = output_dir / "multi_canvas_overview.png"
        plot_multiple_canvases(
            canvas_data,
            metric_col=metric_col,
            quiet_threshold=quiet_threshold,
            save_path=multi_plot_path
        )
    except Exception as e:
        print(f"Error creating multi-canvas plot: {e}")

    print(f"Forecast plots saved to {output_dir}")