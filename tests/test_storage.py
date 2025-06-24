"""Unit tests for the storage module."""

import json
import pytest
from unittest.mock import Mock, patch, mock_open

from src.storage import DataStorage
from src.models import CanvasListResponse, CanvasDetails, Canvas


class TestDataStorage:
    """Test cases for DataStorage."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test canvas list
        self.test_canvas = Canvas(
            id='canvas-123',
            name='Test Canvas',
            tags=['tag1', 'tag2'],
            last_edited='2023-01-01T12:00:00+00:00'
        )

        self.test_canvas_list = CanvasListResponse(
            canvases=[self.test_canvas],
            message='success'
        )

        # Create test canvas details
        self.test_canvas_details = CanvasDetails(
            canvas_id='canvas-123',
            name='Test Canvas',
            description='Test Description',
            created_at='2023-01-01T12:00:00+00:00',
            updated_at='2023-01-02T12:00:00+00:00',
            archived=False,
            draft=False,
            enabled=True,
            has_post_launch_draft=False,
            schedule_type='action_based',
            first_entry='2023-01-01T12:00:00Z',
            last_entry='2023-12-31T12:00:00Z',
            channels=['email'],
            tags=['tag1'],
            teams=[],
            variants=[],
            steps=[]
        )

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_canvas_list_default_filename(self, mock_json_dump, mock_file):
        """Test saving canvas list with default filename."""
        DataStorage.save_canvas_list(self.test_canvas_list)

        # Check file operations
        mock_file.assert_called_once_with('canvas_list.json', 'w')
        mock_json_dump.assert_called_once()

        # Check that the data is properly serialized
        call_args = mock_json_dump.call_args
        saved_data = call_args[0][0]

        assert 'canvases' in saved_data
        assert 'message' in saved_data
        assert saved_data['message'] == 'success'
        assert len(saved_data['canvases']) == 1
        assert saved_data['canvases'][0]['id'] == 'canvas-123'

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_canvas_list_custom_filename(self, mock_json_dump, mock_file):
        """Test saving canvas list with custom filename."""
        custom_filename = 'custom_canvas_list.json'
        DataStorage.save_canvas_list(self.test_canvas_list, custom_filename)

        mock_file.assert_called_once_with(custom_filename, 'w')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_canvas_details_default_filename(self, mock_json_dump, mock_file):
        """Test saving canvas details with default filename."""
        canvas_details_list = [self.test_canvas_details]
        DataStorage.save_canvas_details(canvas_details_list)

        # Check file operations
        mock_file.assert_called_once_with('canvas_details.json', 'w')
        mock_json_dump.assert_called_once()

        # Check that the data is properly serialized
        call_args = mock_json_dump.call_args
        saved_data = call_args[0][0]

        assert isinstance(saved_data, list)
        assert len(saved_data) == 1
        assert saved_data[0]['canvas_id'] == 'canvas-123'
        assert saved_data[0]['name'] == 'Test Canvas'
        assert saved_data[0]['enabled'] is True

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_canvas_details_custom_filename(self, mock_json_dump, mock_file):
        """Test saving canvas details with custom filename."""
        custom_filename = 'custom_canvas_details.json'
        canvas_details_list = [self.test_canvas_details]
        DataStorage.save_canvas_details(canvas_details_list, custom_filename)

        mock_file.assert_called_once_with(custom_filename, 'w')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_canvas_details_empty_list(self, mock_json_dump, mock_file):
        """Test saving empty canvas details list."""
        DataStorage.save_canvas_details([])

        mock_file.assert_called_once_with('canvas_details.json', 'w')

        # Check that empty list is saved
        call_args = mock_json_dump.call_args
        saved_data = call_args[0][0]
        assert saved_data == []

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_canvas_list_success(self, mock_json_load, mock_file):
        """Test successfully loading canvas list."""
        # Mock file content
        mock_data = {
            'canvases': [
                {
                    'id': 'canvas-123',
                    'name': 'Test Canvas',
                    'tags': ['tag1'],
                    'last_edited': '2023-01-01T12:00:00+00:00'
                }
            ],
            'message': 'success'
        }
        mock_json_load.return_value = mock_data

        # Load canvas list
        result = DataStorage.load_canvas_list()

        # Assertions
        assert isinstance(result, CanvasListResponse)
        assert len(result.canvases) == 1
        assert result.canvases[0].id == 'canvas-123'
        assert result.message == 'success'
        mock_file.assert_called_once_with('canvas_list.json', 'r')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_canvas_list_custom_filename(self, mock_json_load, mock_file):
        """Test loading canvas list with custom filename."""
        custom_filename = 'custom_canvas_list.json'
        mock_data = {'canvases': [], 'message': 'success'}
        mock_json_load.return_value = mock_data

        result = DataStorage.load_canvas_list(custom_filename)

        assert result is not None
        mock_file.assert_called_once_with(custom_filename, 'r')

    @patch('builtins.open')
    def test_load_canvas_list_file_not_found(self, mock_open):
        """Test loading canvas list when file doesn't exist."""
        mock_open.side_effect = FileNotFoundError("File not found")

        result = DataStorage.load_canvas_list()

        assert result is None

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_canvas_list_invalid_json(self, mock_json_load, mock_file):
        """Test loading canvas list with invalid JSON."""
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with pytest.raises(json.JSONDecodeError):
            DataStorage.load_canvas_list()

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_canvas_list_empty_canvases(self, mock_json_load, mock_file):
        """Test loading canvas list with empty canvases."""
        mock_data = {'canvases': [], 'message': 'success'}
        mock_json_load.return_value = mock_data

        result = DataStorage.load_canvas_list()

        assert isinstance(result, CanvasListResponse)
        assert len(result.canvases) == 0
        assert result.message == 'success'

    @patch('builtins.open')
    @patch('json.dump')
    def test_save_canvas_list_io_error(self, mock_json_dump, mock_open):
        """Test handling IO error when saving canvas list."""
        mock_open.side_effect = IOError("Permission denied")

        with pytest.raises(IOError):
            DataStorage.save_canvas_list(self.test_canvas_list)

    @patch('builtins.open')
    @patch('json.dump')
    def test_save_canvas_details_io_error(self, mock_json_dump, mock_open):
        """Test handling IO error when saving canvas details."""
        mock_open.side_effect = IOError("Permission denied")

        with pytest.raises(IOError):
            DataStorage.save_canvas_details([self.test_canvas_details])

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_preserves_data_structure(self, mock_json_dump, mock_file):
        """Test that saving preserves the complete data structure."""
        # Create a more complex canvas with multiple elements
        complex_canvas = Canvas(
            id='complex-canvas',
            name='Complex Canvas',
            tags=['tag1', 'tag2', 'tag3'],
            last_edited='2023-06-01T12:00:00+00:00'
        )

        canvas_list = CanvasListResponse(
            canvases=[self.test_canvas, complex_canvas],
            message='success'
        )

        DataStorage.save_canvas_list(canvas_list)

        # Get the saved data
        call_args = mock_json_dump.call_args
        saved_data = call_args[0][0]

        # Verify structure is preserved
        assert len(saved_data['canvases']) == 2
        assert saved_data['canvases'][1]['id'] == 'complex-canvas'
        assert saved_data['canvases'][1]['tags'] == ['tag1', 'tag2', 'tag3']