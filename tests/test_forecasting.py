"""Tests for the forecasting module."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.forecasting.linear_decay import (
    LinearDecayForecaster,
    QuietDatePredictor,
    DailyStats,
    ForecastResult
)


class TestDailyStats(unittest.TestCase):
    """Test the DailyStats dataclass."""

    def test_daily_stats_creation(self):
        """Test creating DailyStats instances."""
        stats = DailyStats(
            date=date(2023, 12, 1),
            entries=100,
            sends=95,
            delivered=90,
            opens=15,
            conversions=2
        )

        self.assertEqual(stats.date, date(2023, 12, 1))
        self.assertEqual(stats.entries, 100)
        self.assertEqual(stats.sends, 95)
        self.assertEqual(stats.delivered, 90)
        self.assertEqual(stats.opens, 15)
        self.assertEqual(stats.conversions, 2)


class TestLinearDecayForecaster(unittest.TestCase):
    """Test the LinearDecayForecaster class."""

    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = LinearDecayForecaster(quiet_threshold=5, min_data_points=7)
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_test_data(self, canvas_id: str, data_points: list) -> None:
        """Create test JSONL data file."""
        jsonl_path = self.data_dir / f"{canvas_id}.jsonl"
        with jsonl_path.open('w') as f:
            for point in data_points:
                f.write(json.dumps(point) + '\n')

    def test_load_canvas_data_success(self):
        """Test successful data loading."""
        canvas_id = "test-canvas"
        test_data = [
            {
                "date": "2023-12-01",
                "entries": 100,
                "sends": 95,
                "delivered": 90,
                "opens": 15,
                "conversions": 2
            },
            {
                "date": "2023-12-02",
                "entries": 90,
                "sends": 85,
                "delivered": 80,
                "opens": 12,
                "conversions": 1
            }
        ]

        self._create_test_data(canvas_id, test_data)

        daily_stats = self.forecaster.load_canvas_data(canvas_id, self.data_dir)

        self.assertEqual(len(daily_stats), 2)
        self.assertEqual(daily_stats[0].date, date(2023, 12, 1))
        self.assertEqual(daily_stats[0].sends, 95)
        self.assertEqual(daily_stats[1].date, date(2023, 12, 2))
        self.assertEqual(daily_stats[1].sends, 85)

    def test_load_canvas_data_missing_file(self):
        """Test loading data when file doesn't exist."""
        daily_stats = self.forecaster.load_canvas_data("nonexistent", self.data_dir)
        self.assertEqual(len(daily_stats), 0)

    def test_load_canvas_data_malformed_json(self):
        """Test loading data with malformed JSON."""
        canvas_id = "malformed-canvas"
        jsonl_path = self.data_dir / f"{canvas_id}.jsonl"

        with jsonl_path.open('w') as f:
            f.write('{"date": "2023-12-01", "sends": 95}\n')
            f.write('malformed json line\n')
            f.write('{"date": "2023-12-02", "sends": 85}\n')

        daily_stats = self.forecaster.load_canvas_data(canvas_id, self.data_dir)
        self.assertEqual(len(daily_stats), 0)  # Should return empty list on error

    def test_predict_quiet_date_insufficient_data(self):
        """Test prediction with insufficient data points."""
        canvas_id = "insufficient-data"
        test_data = [
            {"date": "2023-12-01", "sends": 100},
            {"date": "2023-12-02", "sends": 90}
        ]

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertIsNone(result.quiet_date)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.current_trend, 'insufficient_data')

    def test_predict_quiet_date_declining_trend(self):
        """Test prediction with declining trend."""
        canvas_id = "declining-canvas"
        base_date = date(2023, 12, 1)

        # Create declining trend data
        test_data = []
        for i in range(14):  # 14 days of data
            current_date = base_date + timedelta(days=i)
            sends = max(0, 100 - i * 10)  # Declining by 10 per day
            test_data.append({
                "date": current_date.isoformat(),
                "entries": sends + 5,
                "sends": sends,
                "delivered": sends - 2,
                "opens": int(sends * 0.1),
                "conversions": int(sends * 0.02)
            })

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertIsNotNone(result.quiet_date)
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(result.current_trend, 'declining')
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
            test_data.append({
                "date": current_date.isoformat(),
                "entries": sends + 5,
                "sends": sends,
                "delivered": sends - 2,
                "opens": int(sends * 0.1),
                "conversions": int(sends * 0.02)
            })

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertEqual(result.current_trend, 'growing')
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
            test_data.append({
                "date": current_date.isoformat(),
                "entries": sends + 5,
                "sends": sends,
                "delivered": sends - 2,
                "opens": int(sends * 0.1),
                "conversions": int(sends * 0.02)
            })

        self._create_test_data(canvas_id, test_data)

        result = self.forecaster.predict_quiet_date(canvas_id, self.data_dir)

        self.assertEqual(result.canvas_id, canvas_id)
        self.assertEqual(result.current_trend, 'stable')

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
        """Create test JSONL data file."""
        jsonl_path = self.data_dir / f"{canvas_id}.jsonl"
        with jsonl_path.open('w') as f:
            for point in data_points:
                f.write(json.dumps(point) + '\n')

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
                test_data.append({
                    "date": current_date.isoformat(),
                    "entries": sends + 5,
                    "sends": sends,
                    "delivered": sends - 2,
                    "opens": int(sends * 0.1),
                    "conversions": int(sends * 0.02)
                })

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
            declining_data.append({
                "date": current_date.isoformat(),
                "entries": sends + 5,
                "sends": sends,
                "delivered": sends - 2,
                "opens": int(sends * 0.1),
                "conversions": int(sends * 0.02)
            })

        self._create_test_data("declining-canvas", declining_data)

        # Stable canvas
        stable_data = []
        for i in range(10):
            current_date = base_date + timedelta(days=i)
            sends = 50 + np.random.randint(-5, 6)
            stable_data.append({
                "date": current_date.isoformat(),
                "entries": sends + 5,
                "sends": sends,
                "delivered": sends - 2,
                "opens": int(sends * 0.1),
                "conversions": int(sends * 0.02)
            })

        self._create_test_data("stable-canvas", stable_data)

        report = self.predictor.generate_forecast_report()

        # Check report structure
        self.assertIn('summary', report)
        self.assertIn('trends', report)
        self.assertIn('confidence_distribution', report)
        self.assertIn('urgent_canvases', report)
        self.assertIn('generated_at', report)

        # Check summary
        summary = report['summary']
        self.assertEqual(summary['total_canvases'], 2)
        self.assertIsInstance(summary['predictable'], int)
        self.assertIsInstance(summary['unpredictable'], int)

        # Check trends
        trends = report['trends']
        self.assertIsInstance(trends, dict)

        # Check confidence distribution
        confidence = report['confidence_distribution']
        self.assertIn('high', confidence)
        self.assertIn('medium', confidence)
        self.assertIn('low', confidence)


class TestForecastResult(unittest.TestCase):
    """Test the ForecastResult named tuple."""

    def test_forecast_result_creation(self):
        """Test creating ForecastResult instances."""
        result = ForecastResult(
            canvas_id="test-canvas",
            quiet_date=date(2023, 12, 31),
            confidence=0.85,
            r_squared=0.92,
            days_to_quiet=15,
            current_trend="declining",
            model_params={"slope": -2.5, "intercept": 50.0}
        )

        self.assertEqual(result.canvas_id, "test-canvas")
        self.assertEqual(result.quiet_date, date(2023, 12, 31))
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.r_squared, 0.92)
        self.assertEqual(result.days_to_quiet, 15)
        self.assertEqual(result.current_trend, "declining")
        self.assertEqual(result.model_params, {"slope": -2.5, "intercept": 50.0})

    def test_forecast_result_none_values(self):
        """Test ForecastResult with None values."""
        result = ForecastResult(
            canvas_id="test-canvas",
            quiet_date=None,
            confidence=0.0,
            r_squared=0.0,
            days_to_quiet=None,
            current_trend="insufficient_data",
            model_params={}
        )

        self.assertEqual(result.canvas_id, "test-canvas")
        self.assertIsNone(result.quiet_date)
        self.assertIsNone(result.days_to_quiet)
        self.assertEqual(result.current_trend, "insufficient_data")


if __name__ == '__main__':
    unittest.main()