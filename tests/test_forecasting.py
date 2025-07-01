"""Tests for the forecasting module."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.forecasting.linear_decay import (CanvasMetrics, ForecastResult,
                                          QuietDatePredictor,
                                          StepBasedForecaster, MultiForecastResult)


class TestCanvasMetrics(unittest.TestCase):
    """Test the CanvasMetrics dataclass."""

    def test_canvas_metrics_creation(self):
        """Test creating CanvasMetrics instances."""
        stats = CanvasMetrics(
            date=date(2023, 12, 1),
            canvas_id="test-canvas",
            total_sent=95,
            total_delivered=90,
            total_opens=15,
            total_clicks=2,
        )

        self.assertEqual(stats.date, date(2023, 12, 1))
        self.assertEqual(stats.canvas_id, "test-canvas")
        self.assertEqual(stats.total_sent, 95)
        self.assertEqual(stats.total_delivered, 90)
        self.assertEqual(stats.total_opens, 15)
        self.assertEqual(stats.total_clicks, 2)


class TestStepBasedForecaster(unittest.TestCase):
    """Test the StepBasedForecaster class."""

    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = StepBasedForecaster(quiet_threshold=5, min_data_points=7)
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_data(self, canvas_id: str, data_points: list) -> None:
        """Create test data in the expected step-based directory structure."""
        canvas_dir = self.data_dir / canvas_id
        canvas_dir.mkdir(exist_ok=True)

        # Create a step directory
        step_dir = canvas_dir / "step1"
        step_dir.mkdir(exist_ok=True)

        # Create channel file
        channel_file = step_dir / "email.jsonl"
        with channel_file.open("w") as f:
            for point in data_points:
                # Convert test data to the expected format
                record = {
                    "date": point["date"],
                    "sent": point.get("sends", 0),
                    "delivered": point.get("delivered", 0),
                    "opens": point.get("opens", 0),
                    "clicks": point.get("conversions", 0),
                    "unique_opens": point.get("opens", 0),
                    "unique_clicks": point.get("conversions", 0),
                    "bounces": 0,
                    "unsubscribes": 0,
                }
                f.write(json.dumps(record) + "\n")

    def test_load_canvas_metrics_success(self):
        """Test successful data loading."""
        canvas_id = "test-canvas"
        test_data = [
            {
                "date": "2023-12-01",
                "entries": 100,
                "sends": 95,
                "delivered": 90,
                "opens": 15,
                "conversions": 2,
            },
            {
                "date": "2023-12-02",
                "entries": 90,
                "sends": 85,
                "delivered": 80,
                "opens": 12,
                "conversions": 1,
            },
        ]

        self._create_test_data(canvas_id, test_data)

        canvas_metrics = self.forecaster.load_canvas_metrics(canvas_id, self.data_dir)

        self.assertEqual(len(canvas_metrics), 2)
        self.assertEqual(canvas_metrics[0].date, date(2023, 12, 1))
        self.assertEqual(canvas_metrics[0].total_sent, 95)
        self.assertEqual(canvas_metrics[1].date, date(2023, 12, 2))
        self.assertEqual(canvas_metrics[1].total_sent, 85)

    def test_load_canvas_metrics_missing_file(self):
        """Test loading data when file doesn't exist."""
        canvas_metrics = self.forecaster.load_canvas_metrics(
            "nonexistent", self.data_dir
        )
        self.assertEqual(len(canvas_metrics), 0)

    def test_load_canvas_metrics_malformed_json(self):
        """Test loading data with malformed JSON."""
        canvas_id = "malformed-canvas"
        jsonl_path = self.data_dir / f"{canvas_id}.jsonl"

        with jsonl_path.open("w") as f:
            f.write('{"date": "2023-12-01", "sends": 95}\n')
            f.write("malformed json line\n")
            f.write('{"date": "2023-12-02", "sends": 85}\n')

        canvas_metrics = self.forecaster.load_canvas_metrics(canvas_id, self.data_dir)
        self.assertEqual(len(canvas_metrics), 0)  # Should return empty list on error

    def test_predict_quiet_date_insufficient_data(self):
        """Test prediction with insufficient data points."""
        canvas_id = "insufficient-data"
        test_data = [
            {"date": "2023-12-01", "sends": 100},
            {"date": "2023-12-02", "sends": 90},
        ]

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertIsNone(result.quiet_date)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.current_trend, "insufficient_data")

    def test_predict_quiet_date_declining_trend(self):
        """Test prediction with declining trend."""
        canvas_id = "declining-canvas"
        base_date = date(2023, 12, 1)

        # Create declining trend data for all metrics, ensuring last 7 days are not flat
        test_data = []
        for i in range(14):  # 14 days of data
            current_date = base_date + timedelta(days=i)
            # Decline by 5 per day, so last 7 days are 45, 40, ..., 15
            sends = max(0, 90 - i * 5)
            delivered = max(0, 80 - i * 5)
            opens = max(0, 20 - i)
            conversions = max(0, 7 - i // 2)
            test_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": delivered,
                    "opens": opens,
                    "conversions": conversions,
                }
            )

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertIsNotNone(result.quiet_date)
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(result.current_trend, "declining")
        self.assertIsNotNone(result.days_to_quiet)

    def test_predict_quiet_date_growing_trend(self):
        """Test prediction with growing trend."""
        canvas_id = "growing-canvas"
        base_date = date(2023, 12, 1)

        # Create growing trend data
        test_data = []
        for i in range(10):  # 10 days of data
            current_date = base_date + timedelta(days=i)
            sends = 50 + i * 10  # Growing by 10 per day
            test_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": sends - 2,
                    "opens": int(sends * 0.1),
                    "conversions": int(sends * 0.02),
                }
            )

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertEqual(result.current_trend, "growing")
        # Growing trends should have reduced confidence
        if result.confidence > 0:
            self.assertLess(result.confidence, 0.7)

    def test_predict_quiet_date_stable_trend(self):
        """Test prediction with stable trend."""
        canvas_id = "stable-canvas"
        base_date = date(2023, 12, 1)

        # Create stable trend data
        test_data = []
        for i in range(10):  # 10 days of data
            current_date = base_date + timedelta(days=i)
            sends = 50 + np.random.randint(-5, 6)  # Stable around 50
            test_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": sends - 2,
                    "opens": int(sends * 0.1),
                    "conversions": int(sends * 0.02),
                }
            )

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertEqual(result.current_trend, "stable")

    def test_linear_decay_func(self):
        """Test the linear decay function."""
        x = np.array([1, 2, 3, 4, 5])
        result = self.forecaster._linear_decay_func(x, -2, 10)
        expected = np.array([8, 6, 4, 2, 0])
        np.testing.assert_array_equal(result, expected)

    def test_exponential_decay_func(self):
        """Test the exponential decay function."""
        x = np.array([0, 1, 2])
        result = self.forecaster._exponential_decay_func(x, 10, 0.5, 1)
        expected = np.array([11, 10 * np.exp(-0.5) + 1, 10 * np.exp(-1) + 1])
        np.testing.assert_array_almost_equal(result, expected)


class TestQuietDatePredictor(unittest.TestCase):
    """Test the QuietDatePredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.predictor = QuietDatePredictor(data_dir=self.data_dir, quiet_threshold=5)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_data(self, canvas_id: str, data_points: list) -> None:
        """Create test data in the expected step-based directory structure."""
        canvas_dir = self.data_dir / canvas_id
        canvas_dir.mkdir(exist_ok=True)

        # Create a step directory
        step_dir = canvas_dir / "step1"
        step_dir.mkdir(exist_ok=True)

        # Create channel file
        channel_file = step_dir / "email.jsonl"
        with channel_file.open("w") as f:
            for point in data_points:
                # Convert test data to the expected format
                record = {
                    "date": point["date"],
                    "sent": point.get("sends", 0),
                    "delivered": point.get("delivered", 0),
                    "opens": point.get("opens", 0),
                    "clicks": point.get("conversions", 0),
                    "unique_opens": point.get("opens", 0),
                    "unique_clicks": point.get("conversions", 0),
                    "bounces": 0,
                    "unsubscribes": 0,
                }
                f.write(json.dumps(record) + "\n")

    def test_predict_all_canvases_empty_dir(self):
        """Test predicting when no data files exist."""
        results = self.predictor.predict_all_canvases()
        self.assertEqual(len(results), 0)

    def test_predict_all_canvases_with_data(self):
        """Test predicting with multiple Canvas data files."""
        # Create test data for multiple canvases
        base_date = date(2023, 12, 1)

        for canvas_id in ["canvas1", "canvas2", "canvas3"]:
            test_data = []
            for i in range(10):
                current_date = base_date + timedelta(days=i)
                sends = 100 - i * 5  # Declining trend
                test_data.append(
                    {
                        "date": current_date.isoformat(),
                        "entries": sends + 5,
                        "sends": sends,
                        "delivered": sends - 2,
                        "opens": int(sends * 0.1),
                        "conversions": int(sends * 0.02),
                    }
                )

            self._create_test_data(canvas_id, test_data)

        results = self.predictor.predict_all_canvases()

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, ForecastResult)
            self.assertIn(result.canvas_id, ["canvas1", "canvas2", "canvas3"])

    def test_generate_forecast_report(self):
        """Test generating a comprehensive forecast report."""
        # Create test data with different trends
        base_date = date(2023, 12, 1)

        # Declining canvas
        declining_data = []
        for i in range(14):
            current_date = base_date + timedelta(days=i)
            sends = max(0, 100 - i * 10)
            declining_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": sends - 2,
                    "opens": int(sends * 0.1),
                    "conversions": int(sends * 0.02),
                }
            )

        self._create_test_data("declining-canvas", declining_data)

        # Stable canvas
        stable_data = []
        for i in range(10):
            current_date = base_date + timedelta(days=i)
            sends = 50 + np.random.randint(-5, 6)
            stable_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": sends - 2,
                    "opens": int(sends * 0.1),
                    "conversions": int(sends * 0.02),
                }
            )

        self._create_test_data("stable-canvas", stable_data)

        report = self.predictor.generate_forecast_report()

        # Check report structure
        self.assertIn("summary", report)
        self.assertIn("trends", report)
        self.assertIn("confidence_distribution", report)
        self.assertIn("all_canvases", report)

        # Check summary
        summary = report["summary"]
        self.assertEqual(summary["total_canvases"], 2)
        self.assertIsInstance(summary["predictable"], int)
        self.assertIsInstance(summary["unpredictable"], int)

        # Check trends
        trends = report["trends"]
        self.assertIsInstance(trends, dict)

        # Check confidence distribution
        confidence = report["confidence_distribution"]
        self.assertIn("high", confidence)
        self.assertIn("medium", confidence)
        self.assertIn("low", confidence)


class TestForecastResult(unittest.TestCase):
    """Test the ForecastResult named tuple."""

    def test_forecast_result_creation(self):
        """Test creating ForecastResult instances."""
        result = ForecastResult(
            canvas_id="test-canvas",
            canvas_name="Test Canvas",
            quiet_date=date(2023, 12, 31),
            confidence=0.85,
            r_squared=0.92,
            days_to_quiet=15,
            current_trend="declining",
            model_params={"slope": -2.5, "intercept": 50.0},
            metric_used="total_sent",
            model_type="linear",
        )

        self.assertEqual(result.canvas_id, "test-canvas")
        self.assertEqual(result.canvas_name, "Test Canvas")
        self.assertEqual(result.quiet_date, date(2023, 12, 31))
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.r_squared, 0.92)
        self.assertEqual(result.days_to_quiet, 15)
        self.assertEqual(result.current_trend, "declining")
        self.assertEqual(result.model_params, {"slope": -2.5, "intercept": 50.0})
        self.assertEqual(result.metric_used, "total_sent")
        self.assertEqual(result.model_type, "linear")

    def test_forecast_result_none_values(self):
        """Test ForecastResult with None values."""
        result = ForecastResult(
            canvas_id="test-canvas",
            canvas_name="Test Canvas",
            quiet_date=None,
            confidence=0.0,
            r_squared=0.0,
            days_to_quiet=None,
            current_trend="insufficient_data",
            model_params={},
            metric_used="none",
            model_type="none",
        )

        self.assertEqual(result.canvas_id, "test-canvas")
        self.assertEqual(result.canvas_name, "Test Canvas")
        self.assertIsNone(result.quiet_date)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.r_squared, 0.0)
        self.assertIsNone(result.days_to_quiet)
        self.assertEqual(result.current_trend, "insufficient_data")
        self.assertEqual(result.model_params, {})
        self.assertEqual(result.metric_used, "none")
        self.assertEqual(result.model_type, "none")


class TestMultiModelForecasting(unittest.TestCase):
    """Test the multi-model forecasting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = StepBasedForecaster(quiet_threshold=5, min_data_points=7)
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_test_data(self, canvas_id: str, data_points: list) -> None:
        """Create test data in the expected step-based directory structure."""
        canvas_dir = self.data_dir / canvas_id
        canvas_dir.mkdir(exist_ok=True)

        # Create a step directory
        step_dir = canvas_dir / "step1"
        step_dir.mkdir(exist_ok=True)

        # Create channel file
        channel_file = step_dir / "email.jsonl"
        with channel_file.open("w") as f:
            for point in data_points:
                # Convert test data to the expected format
                record = {
                    "date": point["date"],
                    "sent": point.get("sends", 0),
                    "delivered": point.get("delivered", 0),
                    "opens": point.get("opens", 0),
                    "clicks": point.get("conversions", 0),
                    "unique_opens": point.get("opens", 0),
                    "unique_clicks": point.get("conversions", 0),
                    "bounces": 0,
                    "unsubscribes": 0,
                }
                f.write(json.dumps(record) + "\n")

    def test_predict_quiet_date_all_models_insufficient_data(self):
        """Test multi-model prediction with insufficient data points."""
        canvas_id = "insufficient-data"
        test_data = [
            {"date": "2023-12-01", "sends": 100},
            {"date": "2023-12-02", "sends": 90},
        ]

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date_all_models(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertEqual(len(result.forecasts), 0)
        self.assertIsNone(result.best_forecast)
        self.assertEqual(result.total_models_tried, 0)
        self.assertEqual(result.successful_models, 0)

    def test_predict_quiet_date_all_models_declining_trend(self):
        """Test multi-model prediction with declining trend."""
        canvas_id = "declining-canvas"
        base_date = date(2023, 12, 1)

        # Create declining trend data for all metrics
        test_data = []
        for i in range(14):  # 14 days of data
            current_date = base_date + timedelta(days=i)
            # Decline by 5 per day
            sends = max(0, 90 - i * 5)
            delivered = max(0, 80 - i * 5)
            opens = max(0, 20 - i)
            conversions = max(0, 7 - i // 2)
            test_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": delivered,
                    "opens": opens,
                    "conversions": conversions,
                }
            )

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date_all_models(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertGreater(len(result.forecasts), 0)
        self.assertIsNotNone(result.best_forecast)
        self.assertGreater(result.total_models_tried, 0)
        self.assertGreater(result.successful_models, 0)

        # Check that all forecasts have the required fields
        for forecast in result.forecasts:
            self.assertIsInstance(forecast.model_type, str)
            self.assertIsInstance(forecast.metric_used, str)
            self.assertIsInstance(forecast.r_squared, float)
            self.assertIsInstance(forecast.confidence, float)
            self.assertIsInstance(forecast.current_trend, str)

        # Check that best forecast has highest R-squared
        if result.best_forecast:
            best_r_squared = result.best_forecast.r_squared
            for forecast in result.forecasts:
                self.assertGreaterEqual(best_r_squared, forecast.r_squared)

    def test_predict_quiet_date_all_models_multiple_metrics(self):
        """Test that multiple metrics are tried."""
        canvas_id = "multi-metric-canvas"
        base_date = date(2023, 12, 1)

        # Create data with different patterns for different metrics
        test_data = []
        for i in range(10):  # 10 days of data
            current_date = base_date + timedelta(days=i)
            # Different patterns for different metrics
            sends = max(0, 100 - i * 8)  # Steep decline
            delivered = max(0, 90 - i * 6)  # Moderate decline
            opens = max(0, 30 - i * 2)  # Gentle decline
            conversions = max(0, 10 - i)  # Very gentle decline
            test_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": delivered,
                    "opens": opens,
                    "conversions": conversions,
                }
            )

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date_all_models(canvas_id, self.data_dir)

        # Should have multiple forecasts from different metrics
        self.assertGreater(len(result.forecasts), 1)

        # Check that different metrics are represented
        metrics_used = set(forecast.metric_used for forecast in result.forecasts)
        self.assertGreater(len(metrics_used), 1)

    def test_predict_quiet_date_all_models_backward_compatibility(self):
        """Test that the original predict_quiet_date still works."""
        canvas_id = "compatibility-canvas"
        base_date = date(2023, 12, 1)

        # Create declining trend data
        test_data = []
        for i in range(10):  # 10 days of data
            current_date = base_date + timedelta(days=i)
            sends = max(0, 80 - i * 6)
            test_data.append(
                {
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": sends - 2,
                    "opens": int(sends * 0.1),
                    "conversions": int(sends * 0.02),
                }
            )

        self._create_test_data(canvas_id, test_data)

        # Test original method
        original_result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        # Test new method
        multi_result = self.forecaster.predict_quiet_date_all_models(canvas_id, self.data_dir)

        # The best forecast from multi-result should match the original result
        if multi_result.best_forecast:
            self.assertEqual(original_result.quiet_date, multi_result.best_forecast.quiet_date)
            self.assertEqual(original_result.confidence, multi_result.best_forecast.confidence)
            self.assertEqual(original_result.r_squared, multi_result.best_forecast.r_squared)
            self.assertEqual(original_result.current_trend, multi_result.best_forecast.current_trend)
            self.assertEqual(original_result.metric_used, multi_result.best_forecast.metric_used)
            self.assertEqual(original_result.model_type, multi_result.best_forecast.model_type)


class TestMultiForecastResult(unittest.TestCase):
    """Test the MultiForecastResult class."""

    def test_multi_forecast_result_creation(self):
        """Test creating MultiForecastResult instances."""
        forecasts = [
            ForecastResult(
                canvas_id="test-canvas",
                canvas_name="Test Canvas",
                quiet_date=date(2024, 1, 15),
                confidence=0.85,
                r_squared=0.75,
                days_to_quiet=30,
                current_trend="declining",
                model_params={"slope": -2.5, "intercept": 100},
                metric_used="total_sent",
                model_type="linear",
            ),
            ForecastResult(
                canvas_id="test-canvas",
                canvas_name="Test Canvas",
                quiet_date=date(2024, 1, 20),
                confidence=0.80,
                r_squared=0.70,
                days_to_quiet=35,
                current_trend="declining",
                model_params={"a": 100, "b": 0.1, "c": 0},
                metric_used="total_opens",
                model_type="exponential",
            ),
        ]

        result = MultiForecastResult(
            canvas_id="test-canvas",
            canvas_name="Test Canvas",
            forecasts=forecasts,
            best_forecast=forecasts[0],  # First one has higher R-squared
            total_models_tried=4,
            successful_models=2,
        )

        self.assertEqual(result.canvas_id, "test-canvas")
        self.assertEqual(result.canvas_name, "Test Canvas")
        self.assertEqual(len(result.forecasts), 2)
        self.assertEqual(result.best_forecast, forecasts[0])
        self.assertEqual(result.total_models_tried, 4)
        self.assertEqual(result.successful_models, 2)

    def test_multi_forecast_result_empty(self):
        """Test creating MultiForecastResult with no forecasts."""
        result = MultiForecastResult(
            canvas_id="empty-canvas",
            canvas_name="Empty Canvas",
            forecasts=[],
            best_forecast=None,
            total_models_tried=0,
            successful_models=0,
        )

        self.assertEqual(result.canvas_id, "empty-canvas")
        self.assertEqual(len(result.forecasts), 0)
        self.assertIsNone(result.best_forecast)
        self.assertEqual(result.total_models_tried, 0)
        self.assertEqual(result.successful_models, 0)


if __name__ == "__main__":
    unittest.main()
