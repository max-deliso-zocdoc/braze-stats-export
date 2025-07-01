# flake8: noqa
#!/usr/bin/env python3
"""
Test script to demonstrate the new multi-model forecasting functionality.
This script shows how to use the updated forecasting system that displays
all predicted vectors instead of just the best one.
"""

import sys
from pathlib import Path
from forecasting.linear_decay import QuietDatePredictor, StepBasedForecaster
from visualization.canvas_forecast import plot_canvas_forecast_all_models

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))


def test_multi_model_forecasting():
    """Test the new multi-model forecasting functionality."""

    # Initialize the predictor
    data_dir = Path("data")
    if not data_dir.exists():
        print("Data directory not found. Please ensure you have data in the 'data' directory.")
        return

    predictor = QuietDatePredictor(data_dir=data_dir, quiet_threshold=5)

    # Get all canvas directories
    canvas_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not canvas_dirs:
        print("No canvas data found in the data directory.")
        return

    print(f"Found {len(canvas_dirs)} canvas directories")

    # Test with the first canvas that has sufficient data
    for canvas_dir in canvas_dirs:
        canvas_id = canvas_dir.name
        print(f"\nTesting multi-model forecasting for Canvas: {canvas_id}")

        # Create forecaster instance
        forecaster = StepBasedForecaster(quiet_threshold=5, min_data_points=7)

        # Get all predictions for this canvas
        multi_result = forecaster.predict_quiet_date_all_models(canvas_id, data_dir)

        if multi_result.forecasts:
            print(f"  Total models tried: {multi_result.total_models_tried}")
            print(f"  Successful models: {multi_result.successful_models}")
            print(f"  Number of forecasts: {len(multi_result.forecasts)}")

            # Display all forecasts
            for i, forecast in enumerate(multi_result.forecasts):
                print(f"  Forecast {i+1}:")
                print(f"    Model: {forecast.model_type}")
                print(f"    Metric: {forecast.metric_used}")
                print(f"    R²: {forecast.r_squared:.3f}")
                print(f"    Confidence: {forecast.confidence:.1%}")
                print(f"    Trend: {forecast.current_trend}")
                if forecast.quiet_date:
                    print(f"    Quiet date: {forecast.quiet_date}")
                    print(f"    Days to quiet: {forecast.days_to_quiet}")
                else:
                    print("    Quiet date: None")
                print()

            # Show best forecast
            if multi_result.best_forecast:
                best = multi_result.best_forecast
                print("  Best forecast:")
                print(f"    Model: {best.model_type}")
                print(f"    Metric: {best.metric_used}")
                print(f"    R²: {best.r_squared:.3f}")
                if best.quiet_date:
                    print(f"    Quiet date: {best.quiet_date}")
                    print(f"    Days to quiet: {best.days_to_quiet}")
                print()

            # Test visualization
            try:
                # Load metrics for visualization
                metrics = forecaster.load_canvas_metrics(canvas_id, data_dir)
                if metrics:
                    print(f"  Generating visualization...")
                    plot_canvas_forecast_all_models(
                        metrics=metrics,
                        metric_col="total_sent",
                        quiet_threshold=5,
                        horizon_days=180,
                        save_path=Path(f"plots/multi_model_forecast_{canvas_id}.png"),
                        show_plot=False
                    )
                    print(f"  Visualization saved to plots/multi_model_forecast_{canvas_id}.png")
            except Exception as e:
                print(f"  Visualization failed: {e}")

            break  # Only test the first canvas with data
        else:
            print("  No successful forecasts for this canvas")

    # Test the batch prediction
    print("\nTesting batch prediction with all models...")
    try:
        multi_results = predictor.predict_all_canvases_all_models()
        print(f"Processed {len(multi_results)} canvases")

        # Show summary
        total_models_tried = sum(r.total_models_tried for r in multi_results)
        total_successful = sum(r.successful_models for r in multi_results)
        total_forecasts = sum(len(r.forecasts) for r in multi_results)

        print(f"Total models tried across all canvases: {total_models_tried}")
        print(f"Total successful models: {total_successful}")
        print(f"Total forecasts generated: {total_forecasts}")

        # Show canvases with multiple forecasts
        multi_forecast_canvases = [r for r in multi_results if len(r.forecasts) > 1]
        print(f"Canvases with multiple forecasts: {len(multi_forecast_canvases)}")

        for result in multi_forecast_canvases[:3]:  # Show first 3
            print(f"  {result.canvas_name}: {len(result.forecasts)} forecasts")

    except Exception as e:
        print(f"Batch prediction failed: {e}")


if __name__ == "__main__":
    test_multi_model_forecasting()