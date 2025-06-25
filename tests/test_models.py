"""Unit tests for the models module."""

import pytest
from datetime import datetime

from src.models import (
    Canvas,
    CanvasListResponse,
    CanvasVariant,
    CanvasStepPath,
    CanvasStep,
    CanvasDetails,
    RequestLog,
)


class TestCanvas:
    """Test cases for Canvas model."""

    def test_from_dict_basic(self):
        """Test creating Canvas from basic dictionary."""
        data = {
            "id": "test-id-123",
            "name": "Test Canvas",
            "tags": ["tag1", "tag2"],
            "last_edited": "2023-01-01T12:00:00+00:00",
        }

        canvas = Canvas.from_dict(data)

        assert canvas.id == "test-id-123"
        assert canvas.name == "Test Canvas"
        assert canvas.tags == ["tag1", "tag2"]
        assert canvas.last_edited == "2023-01-01T12:00:00+00:00"

    def test_from_dict_empty_tags(self):
        """Test creating Canvas with empty tags."""
        data = {
            "id": "test-id-123",
            "name": "Test Canvas",
            "tags": [],
            "last_edited": "2023-01-01T12:00:00+00:00",
        }

        canvas = Canvas.from_dict(data)
        assert canvas.tags == []


class TestCanvasListResponse:
    """Test cases for CanvasListResponse model."""

    def test_from_dict_multiple_canvases(self):
        """Test creating CanvasListResponse with multiple canvases."""
        data = {
            "canvases": [
                {
                    "id": "id-1",
                    "name": "Canvas 1",
                    "tags": ["tag1"],
                    "last_edited": "2023-01-01T12:00:00+00:00",
                },
                {
                    "id": "id-2",
                    "name": "Canvas 2",
                    "tags": ["tag2"],
                    "last_edited": "2023-01-02T12:00:00+00:00",
                },
            ],
            "message": "success",
        }

        response = CanvasListResponse.from_dict(data)

        assert len(response.canvases) == 2
        assert response.message == "success"
        assert response.canvases[0].id == "id-1"
        assert response.canvases[1].id == "id-2"

    def test_from_dict_empty_canvases(self):
        """Test creating CanvasListResponse with no canvases."""
        data = {"canvases": [], "message": "success"}

        response = CanvasListResponse.from_dict(data)

        assert len(response.canvases) == 0
        assert response.message == "success"


class TestCanvasVariant:
    """Test cases for CanvasVariant model."""

    def test_from_dict_complete(self):
        """Test creating CanvasVariant with all fields."""
        data = {
            "id": "variant-id",
            "name": "Variant 1",
            "first_step_ids": ["step-1", "step-2"],
            "first_step_id": "step-1",
        }

        variant = CanvasVariant.from_dict(data)

        assert variant.id == "variant-id"
        assert variant.name == "Variant 1"
        assert variant.first_step_ids == ["step-1", "step-2"]
        assert variant.first_step_id == "step-1"

    def test_from_dict_missing_fields(self):
        """Test creating CanvasVariant with missing fields."""
        data = {}

        variant = CanvasVariant.from_dict(data)

        assert variant.id == ""
        assert variant.name == ""
        assert variant.first_step_ids == []
        assert variant.first_step_id == ""


class TestCanvasStepPath:
    """Test cases for CanvasStepPath model."""

    def test_from_dict_complete(self):
        """Test creating CanvasStepPath with all fields."""
        data = {"name": "Path to Next", "next_step_id": "next-step-id"}

        path = CanvasStepPath.from_dict(data)

        assert path.name == "Path to Next"
        assert path.next_step_id == "next-step-id"

    def test_from_dict_missing_fields(self):
        """Test creating CanvasStepPath with missing fields."""
        data = {}

        path = CanvasStepPath.from_dict(data)

        assert path.name == ""
        assert path.next_step_id == ""


class TestCanvasStep:
    """Test cases for CanvasStep model."""

    def test_from_dict_with_paths(self):
        """Test creating CanvasStep with next paths."""
        data = {
            "id": "step-id",
            "name": "Test Step",
            "type": "message",
            "next_paths": [
                {"name": "Path 1", "next_step_id": "step-2"},
                {"name": "Path 2", "next_step_id": "step-3"},
            ],
        }

        step = CanvasStep.from_dict(data)

        assert step.id == "step-id"
        assert step.name == "Test Step"
        assert step.type == "message"
        assert len(step.next_paths) == 2
        assert step.next_paths[0].name == "Path 1"
        assert step.next_paths[1].next_step_id == "step-3"

    def test_from_dict_no_paths(self):
        """Test creating CanvasStep without next paths."""
        data = {"id": "step-id", "name": "Test Step", "type": "message"}

        step = CanvasStep.from_dict(data)

        assert step.id == "step-id"
        assert step.name == "Test Step"
        assert step.type == "message"
        assert len(step.next_paths) == 0


class TestCanvasDetails:
    """Test cases for CanvasDetails model."""

    def test_from_dict_complete(self):
        """Test creating CanvasDetails with all fields."""
        data = {
            "name": "Test Canvas",
            "description": "Test Description",
            "created_at": "2023-01-01T12:00:00+00:00",
            "updated_at": "2023-01-02T12:00:00+00:00",
            "archived": False,
            "draft": True,
            "enabled": True,
            "has_post_launch_draft": False,
            "schedule_type": "action_based",
            "first_entry": "2023-01-01T12:00:00Z",
            "last_entry": "2023-12-31T12:00:00Z",
            "channels": ["email", "push"],
            "tags": ["tag1", "tag2"],
            "teams": ["team1"],
            "variants": [
                {
                    "id": "var-1",
                    "name": "Variant 1",
                    "first_step_ids": ["step-1"],
                    "first_step_id": "step-1",
                }
            ],
            "steps": [
                {"id": "step-1", "name": "Step 1", "type": "message", "next_paths": []}
            ],
        }

        details = CanvasDetails.from_dict(data, "canvas-id-123")

        assert details.canvas_id == "canvas-id-123"
        assert details.name == "Test Canvas"
        assert details.description == "Test Description"
        assert details.archived is False
        assert details.draft is True
        assert details.enabled is True
        assert details.schedule_type == "action_based"
        assert details.channels == ["email", "push"]
        assert details.tags == ["tag1", "tag2"]
        assert len(details.variants) == 1
        assert len(details.steps) == 1
        assert details.variants[0].name == "Variant 1"
        assert details.steps[0].name == "Step 1"

    def test_from_dict_minimal(self):
        """Test creating CanvasDetails with minimal data."""
        data = {}

        details = CanvasDetails.from_dict(data, "canvas-id-123")

        assert details.canvas_id == "canvas-id-123"
        assert details.name == ""
        assert details.description == ""
        assert details.archived is False
        assert details.draft is False
        assert details.enabled is False
        assert details.channels == []
        assert details.tags == []
        assert details.variants == []
        assert details.steps == []


class TestRequestLog:
    """Test cases for RequestLog model."""

    def test_to_dict(self):
        """Test converting RequestLog to dictionary."""
        log = RequestLog(
            timestamp="2023-01-01T12:00:00",
            endpoint="/canvas/list",
            method="GET",
            status_code=200,
            response_time_ms=150.5,
            success=True,
            error_message=None,
        )

        log_dict = log.to_dict()

        expected = {
            "timestamp": "2023-01-01T12:00:00",
            "endpoint": "/canvas/list",
            "method": "GET",
            "status_code": 200,
            "response_time_ms": 150.5,
            "success": True,
            "error_message": None,
        }

        assert log_dict == expected

    def test_to_dict_with_error(self):
        """Test converting RequestLog with error to dictionary."""
        log = RequestLog(
            timestamp="2023-01-01T12:00:00",
            endpoint="/canvas/details",
            method="GET",
            status_code=500,
            response_time_ms=1000.0,
            success=False,
            error_message="Internal Server Error",
        )

        log_dict = log.to_dict()

        assert log_dict["success"] is False
        assert log_dict["error_message"] == "Internal Server Error"
        assert log_dict["status_code"] == 500
