"""Unit tests for the API client module."""

import json
import pytest
from unittest.mock import Mock, patch, mock_open

import requests
from requests import Response

from src.api import BrazeAPIClient
from src.models import CanvasListResponse, CanvasDetails


class TestBrazeAPIClient:
    """Test cases for BrazeAPIClient."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.client = BrazeAPIClient(self.api_key)

    def test_init(self):
        """Test BrazeAPIClient initialization."""
        assert self.client.api_key == "test-api-key"
        assert self.client.endpoint == "https://rest.iad-02.braze.com"
        assert self.client.headers == {"Authorization": "Bearer test-api-key"}
        assert len(self.client.request_log) == 0

    def test_init_custom_endpoint(self):
        """Test BrazeAPIClient initialization with custom endpoint."""
        custom_endpoint = "https://rest.iad-03.braze.com"
        client = BrazeAPIClient(self.api_key, custom_endpoint)

        assert client.endpoint == custom_endpoint

    @patch('src.api.client.requests.get')
    def test_make_request_get_success(self, mock_get):
        """Test successful GET request."""
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response

        # Make request
        response = self.client._make_request("/test", limit=10)

        # Assertions
        assert response == mock_response
        mock_get.assert_called_once_with(
            "https://rest.iad-02.braze.com/test",
            headers={"Authorization": "Bearer test-api-key"},
            params={"limit": 10},
            timeout=10
        )
        assert len(self.client.request_log) == 1
        assert self.client.request_log[0].success is True
        assert self.client.request_log[0].endpoint == "/test"

    @patch('src.api.client.requests.get')
    def test_make_request_get_failure(self, mock_get):
        """Test failed GET request."""
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 404
        mock_response.ok = False
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        # Make request and expect exception
        with pytest.raises(requests.HTTPError):
            self.client._make_request("/test")

        # Assertions
        assert len(self.client.request_log) == 2  # One for failed response, one for exception
        assert self.client.request_log[0].success is False
        assert self.client.request_log[0].status_code == 404
        assert self.client.request_log[0].error_message == "Not Found"
        assert self.client.request_log[1].success is False
        assert self.client.request_log[1].status_code == 0  # Exception doesn't have status code
        assert "404 Client Error" in self.client.request_log[1].error_message

    @patch('src.api.client.requests.get')
    def test_make_request_connection_error(self, mock_get):
        """Test connection error during request."""
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        with pytest.raises(requests.ConnectionError):
            self.client._make_request("/test")

        assert len(self.client.request_log) == 1
        assert self.client.request_log[0].success is False
        assert self.client.request_log[0].status_code == 0
        assert "Connection failed" in self.client.request_log[0].error_message

    @patch('src.api.client.requests.request')
    def test_make_request_post(self, mock_request):
        """Test POST request."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response

        response = self.client._make_request("/test", method="POST", data={"key": "value"})

        assert response == mock_response
        mock_request.assert_called_once_with(
            "POST",
            "https://rest.iad-02.braze.com/test",
            headers={"Authorization": "Bearer test-api-key"},
            json={"data": {"key": "value"}},
            timeout=10
        )

    @patch.object(BrazeAPIClient, '_make_request')
    def test_get_canvas_list(self, mock_make_request):
        """Test get_canvas_list method."""
        # Mock response data
        response_data = {
            'canvases': [
                {
                    'id': 'canvas-1',
                    'name': 'Test Canvas',
                    'tags': ['tag1'],
                    'last_edited': '2023-01-01T12:00:00+00:00'
                }
            ],
            'message': 'success'
        }

        mock_response = Mock()
        mock_response.json.return_value = response_data
        mock_make_request.return_value = mock_response

        # Call method
        result = self.client.get_canvas_list(limit=50)

        # Assertions
        assert isinstance(result, CanvasListResponse)
        assert len(result.canvases) == 1
        assert result.canvases[0].id == 'canvas-1'
        assert result.message == 'success'
        mock_make_request.assert_called_once_with("/canvas/list", limit=50)

    @patch.object(BrazeAPIClient, '_make_request')
    def test_get_canvas_details(self, mock_make_request):
        """Test get_canvas_details method."""
        # Mock response data
        response_data = {
            'name': 'Test Canvas',
            'description': 'Test Description',
            'created_at': '2023-01-01T12:00:00+00:00',
            'updated_at': '2023-01-02T12:00:00+00:00',
            'archived': False,
            'draft': False,
            'enabled': True,
            'has_post_launch_draft': False,
            'schedule_type': 'action_based',
            'first_entry': '2023-01-01T12:00:00Z',
            'last_entry': '2023-12-31T12:00:00Z',
            'channels': ['email'],
            'tags': ['tag1'],
            'teams': [],
            'variants': [],
            'steps': []
        }

        mock_response = Mock()
        mock_response.json.return_value = response_data
        mock_make_request.return_value = mock_response

        # Call method
        result = self.client.get_canvas_details('canvas-123')

        # Assertions
        assert isinstance(result, CanvasDetails)
        assert result.canvas_id == 'canvas-123'
        assert result.name == 'Test Canvas'
        assert result.enabled is True
        mock_make_request.assert_called_once_with("/canvas/details", canvas_id='canvas-123')

    @patch.object(BrazeAPIClient, '_make_request')
    def test_get_canvas_details_wrapped_response(self, mock_make_request):
        """Test get_canvas_details with wrapped canvas response."""
        # Mock wrapped response data
        response_data = {
            'canvas': {
                'name': 'Test Canvas',
                'enabled': True,
                'channels': ['email'],
                'tags': [],
                'variants': [],
                'steps': []
            }
        }

        mock_response = Mock()
        mock_response.json.return_value = response_data
        mock_make_request.return_value = mock_response

        # Call method
        result = self.client.get_canvas_details('canvas-123')

        # Assertions
        assert isinstance(result, CanvasDetails)
        assert result.canvas_id == 'canvas-123'
        assert result.name == 'Test Canvas'

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_request_log(self, mock_json_dump, mock_file):
        """Test save_request_log method."""
        # Add some mock logs
        self.client.request_log = [
            Mock(to_dict=Mock(return_value={'test': 'log1'})),
            Mock(to_dict=Mock(return_value={'test': 'log2'}))
        ]

        # Call method
        self.client.save_request_log('test_log.json')

        # Assertions
        mock_file.assert_called_once_with('test_log.json', 'w')
        mock_json_dump.assert_called_once()

        # Check that the data passed to json.dump is correct
        call_args = mock_json_dump.call_args
        assert call_args[0][0] == [{'test': 'log1'}, {'test': 'log2'}]

    def test_request_log_timing(self):
        """Test that request timing is properly recorded."""
        with patch('src.api.client.requests.get') as mock_get:
            mock_response = Mock(spec=Response)
            mock_response.status_code = 200
            mock_response.ok = True
            mock_get.return_value = mock_response

            # Make request
            self.client._make_request("/test")

            # Check that timing was recorded
            assert len(self.client.request_log) == 1
            log_entry = self.client.request_log[0]
            assert log_entry.response_time_ms > 0
            assert isinstance(log_entry.response_time_ms, float)