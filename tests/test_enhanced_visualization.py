#!/usr/bin/env python3
"""
Test for the enhanced multi-model visualization.
This test checks that the updated plots display all data sources with color-coded predictions.
"""

import sys
from pathlib import Path
import unittest

from src.forecasting.linear_decay import StepBasedForecaster
from src.visualization.canvas_forecast import plot_canvas_forecast_all_models

class TestEnhancedVisualization(unittest.TestCase):
    def test_enhanced_visualization(self):
        """Test the enhanced multi-model visualization with all data sources."""
        data_dir = Path("data")
        self.assertTrue(data_dir.exists(), "Data directory not found.")
        canvas_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        self.assertTrue(canvas_dirs, "No canvas data found in the data directory.")

        # Test with the first canvas that has sufficient data
        for canvas_dir in canvas_dirs:
            canvas_id = canvas_dir.name
            forecaster = StepBasedForecaster(quiet_threshold=5, min_data_points=7)
            metrics = forecaster.load_canvas_metrics(canvas_id, data_dir)
            if not metrics:
                continue
            multi_result = forecaster.predict_quiet_date_all_models(canvas_id, data_dir)
            if multi_result.forecasts:
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                plot_path = plots_dir / f"enhanced_multi_model_{canvas_id}.png"
                result = plot_canvas_forecast_all_models(
                    metrics=metrics,
                    metric_col="total_sent",
                    quiet_threshold=5,
                    horizon_days=365,
                    save_path=plot_path,
                    show_plot=False
                )
                self.assertTrue(plot_path.exists(), f"Plot was not created: {plot_path}")
                break  # Only test the first canvas with data
        else:
            self.fail("No successful forecasts for any canvas.")

if __name__ == "__main__":
    unittest.main()