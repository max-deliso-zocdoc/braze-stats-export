"""Forecasting models for Canvas analytics."""

from .linear_decay import QuietDatePredictor, StepBasedForecaster

__all__ = [
    "StepBasedForecaster",
    "QuietDatePredictor",
]
