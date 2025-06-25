"""Forecasting models for Canvas analytics."""

from .linear_decay import StepBasedForecaster, QuietDatePredictor

__all__ = [
    "StepBasedForecaster",
    "QuietDatePredictor",
]
