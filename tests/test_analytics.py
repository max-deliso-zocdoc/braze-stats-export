"""Unit tests for the analytics module."""

import pytest
from datetime import datetime, timedelta

from src.analytics import CanvasStatistics
from src.models import (
    Canvas, CanvasListResponse, CanvasDetails, CanvasStep,
    CanvasStepPath, CanvasVariant, RequestLog
)


class TestCanvasStatistics:
    """Test cases for CanvasStatistics."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test canvases with different dates and tags
        self.canvas1 = Canvas(
            id='canvas-1',
            name='Canvas 1',
            tags=['Lifecycle', 'Welcome'],
            last_edited='2023-01-01T12:00:00+00:00'
        )

        self.canvas2 = Canvas(
            id='canvas-2',
            name='Canvas 2',
            tags=['Lifecycle', 'QA'],
            last_edited='2023-06-01T12:00:00+00:00'
        )

        self.canvas3 = Canvas(
            id='canvas-3',
            name='Canvas 3',
            tags=[],  # No tags
            last_edited='2023-12-01T12:00:00+00:00'
        )

        # Recent canvas (last 30 days) - use current date minus a few days
        recent_date = (datetime.now() - timedelta(days=10)).isoformat() + '+00:00'
        self.canvas_recent = Canvas(
            id='canvas-recent',
            name='Recent Canvas',
            tags=['Recent'],
            last_edited=recent_date
        )

        self.canvas_list = CanvasListResponse(
            canvases=[self.canvas1, self.canvas2, self.canvas3, self.canvas_recent],
            message='success'
        )

        # Create test canvas details
        self.canvas_details1 = CanvasDetails(
            canvas_id='canvas-1',
            name='Canvas 1',
            description='Test Canvas 1',
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
            tags=['Lifecycle'],
            teams=[],
            variants=[],
            steps=[
                CanvasStep(
                    id='step-1',
                    name='Step 1',
                    type='message',
                    next_paths=[]
                ),
                CanvasStep(
                    id='step-2',
                    name='Step 2',
                    type='delay',
                    next_paths=[]
                )
            ]
        )

        self.canvas_details2 = CanvasDetails(
            canvas_id='canvas-2',
            name='Canvas 2',
            description='Test Canvas 2',
            created_at='2023-02-01T12:00:00+00:00',
            updated_at='2023-02-02T12:00:00+00:00',
            archived=True,
            draft=True,
            enabled=False,
            has_post_launch_draft=True,
            schedule_type='time_based',
            first_entry='2023-02-01T12:00:00Z',
            last_entry='2023-12-31T12:00:00Z',
            channels=['push', 'webhook'],
            tags=['QA'],
            teams=['team1'],
            variants=[],
            steps=[
                CanvasStep(
                    id='step-3',
                    name='Step 3',
                    type='audience_paths',
                    next_paths=[
                        CanvasStepPath(name='Path 1', next_step_id='step-4'),
                        CanvasStepPath(name='Path 2', next_step_id='step-5')
                    ]
                )
            ]
        )

    def test_analyze_canvas_list_basic_stats(self):
        """Test basic canvas list analysis."""
        stats = CanvasStatistics.analyze_canvas_list(self.canvas_list)

        assert stats['total_canvases'] == 4
        assert stats['canvases_without_tags'] == 1  # canvas3 has no tags
        assert stats['total_tags'] == 4  # Lifecycle, Welcome, QA, Recent
        assert stats['recent_activity_count'] == 1  # Only canvas_recent

    def test_analyze_canvas_list_tag_distribution(self):
        """Test tag frequency analysis."""
        stats = CanvasStatistics.analyze_canvas_list(self.canvas_list)

        # Check tag distribution
        tag_dist = stats['tag_distribution']
        assert tag_dist['Lifecycle'] == 2  # canvas1 and canvas2
        assert tag_dist['Welcome'] == 1   # canvas1
        assert tag_dist['QA'] == 1        # canvas2
        assert tag_dist['Recent'] == 1    # canvas_recent

        # Check most common tags (should be sorted by frequency)
        most_common = stats['most_common_tags']
        assert most_common[0] == ('Lifecycle', 2)
        assert len(most_common) <= 10  # Limited to top 10

    def test_analyze_canvas_list_recent_activity(self):
        """Test recent activity analysis."""
        stats = CanvasStatistics.analyze_canvas_list(self.canvas_list)

        # Should only count canvas_recent as recent (within last 30 days)
        assert stats['recent_activity_count'] == 1

    def test_analyze_canvas_list_empty(self):
        """Test analysis with empty canvas list."""
        empty_list = CanvasListResponse(canvases=[], message='success')
        stats = CanvasStatistics.analyze_canvas_list(empty_list)

        assert stats['total_canvases'] == 0
        assert stats['total_tags'] == 0
        assert stats['canvases_without_tags'] == 0
        assert stats['recent_activity_count'] == 0
        assert stats['most_common_tags'] == []
        assert stats['tag_distribution'] == {}

    def test_analyze_canvas_list_invalid_dates(self):
        """Test handling of invalid date formats."""
        canvas_bad_date = Canvas(
            id='canvas-bad',
            name='Bad Date Canvas',
            tags=['test'],
            last_edited='invalid-date-format'
        )

        bad_date_list = CanvasListResponse(
            canvases=[canvas_bad_date],
            message='success'
        )

        stats = CanvasStatistics.analyze_canvas_list(bad_date_list)

        # Should handle invalid dates gracefully
        assert stats['total_canvases'] == 1
        assert stats['recent_activity_count'] == 0  # Invalid date not counted as recent

    def test_analyze_canvas_details_basic_stats(self):
        """Test basic canvas details analysis."""
        details_list = [self.canvas_details1, self.canvas_details2]
        stats = CanvasStatistics.analyze_canvas_details(details_list)

        assert stats['total_analyzed'] == 2

    def test_analyze_canvas_details_channel_distribution(self):
        """Test channel distribution analysis."""
        details_list = [self.canvas_details1, self.canvas_details2]
        stats = CanvasStatistics.analyze_canvas_details(details_list)

        channel_dist = stats['channel_distribution']
        assert channel_dist['email'] == 1      # canvas_details1
        assert channel_dist['push'] == 1       # canvas_details2
        assert channel_dist['webhook'] == 1    # canvas_details2

    def test_analyze_canvas_details_schedule_types(self):
        """Test schedule type distribution."""
        details_list = [self.canvas_details1, self.canvas_details2]
        stats = CanvasStatistics.analyze_canvas_details(details_list)

        schedule_dist = stats['schedule_type_distribution']
        assert schedule_dist['action_based'] == 1
        assert schedule_dist['time_based'] == 1

    def test_analyze_canvas_details_status_distribution(self):
        """Test canvas status analysis."""
        details_list = [self.canvas_details1, self.canvas_details2]
        stats = CanvasStatistics.analyze_canvas_details(details_list)

        status_dist = stats['status_distribution']
        assert status_dist['enabled'] == 1    # canvas_details1
        assert status_dist['disabled'] == 1   # canvas_details2
        assert status_dist['archived'] == 1   # canvas_details2
        assert status_dist['draft'] == 1      # canvas_details2
        assert status_dist['has_post_launch_draft'] == 1  # canvas_details2

    def test_analyze_canvas_details_step_analysis(self):
        """Test step type and complexity analysis."""
        details_list = [self.canvas_details1, self.canvas_details2]
        stats = CanvasStatistics.analyze_canvas_details(details_list)

        step_dist = stats['step_type_distribution']
        assert step_dist['message'] == 1
        assert step_dist['delay'] == 1
        assert step_dist['audience_paths'] == 1

        complexity = stats['complexity_analysis']
        assert complexity['total_steps'] == 3  # 2 + 1
        assert complexity['average_steps_per_canvas'] == 1.5  # 3/2
        assert complexity['min_steps'] == 1
        assert complexity['max_steps'] == 2

    def test_analyze_canvas_details_empty_list(self):
        """Test analysis with empty canvas details list."""
        stats = CanvasStatistics.analyze_canvas_details([])

        assert 'error' in stats
        assert stats['error'] == "No canvas details available"

    def test_analyze_canvas_details_missing_fields(self):
        """Test handling of missing fields in canvas details."""
        minimal_details = CanvasDetails(
            canvas_id='minimal',
            name='Minimal Canvas',
            description='',
            created_at='',
            updated_at='',
            archived=False,
            draft=False,
            enabled=True,
            has_post_launch_draft=False,
            schedule_type='',  # Empty schedule type
            first_entry='',
            last_entry='',
            channels=[],
            tags=[],
            teams=[],
            variants=[],
            steps=[]
        )

        stats = CanvasStatistics.analyze_canvas_details([minimal_details])

        # Should handle missing/empty fields gracefully
        assert stats['total_analyzed'] == 1
        assert stats['schedule_type_distribution']['unknown'] == 1
        assert stats['complexity_analysis']['total_steps'] == 0

    def test_generate_summary_report_complete(self):
        """Test generating complete summary report."""
        canvas_list_stats = {
            'total_canvases': 100,
            'total_tags': 25,
            'recent_activity_count': 5,
            'canvases_without_tags': 3,
            'most_common_tags': [('Lifecycle', 50), ('QA', 25), ('Welcome', 10)]
        }

        canvas_details_stats = {
            'total_analyzed': 5,
            'channel_distribution': {'email': 4, 'push': 2},
            'schedule_type_distribution': {'action_based': 3, 'time_based': 2},
            'status_distribution': {'enabled': 3, 'disabled': 2, 'archived': 1},
            'complexity_analysis': {
                'total_steps': 25,
                'average_steps_per_canvas': 5.0,
                'min_steps': 2,
                'max_steps': 10
            }
        }

        request_log = [
            RequestLog(
                timestamp='2023-01-01T12:00:00',
                endpoint='/canvas/list',
                method='GET',
                status_code=200,
                response_time_ms=150.0,
                success=True
            ),
            RequestLog(
                timestamp='2023-01-01T12:01:00',
                endpoint='/canvas/details',
                method='GET',
                status_code=200,
                response_time_ms=250.0,
                success=True
            )
        ]

        report = CanvasStatistics.generate_summary_report(
            canvas_list_stats,
            canvas_details_stats,
            request_log
        )

        # Check that report contains expected sections
        assert "BRAZE CANVAS EXPORT SUMMARY REPORT" in report
        assert "API PERFORMANCE:" in report
        assert "CANVAS OVERVIEW:" in report
        assert "MOST COMMON TAGS:" in report
        assert "DETAILED ANALYSIS:" in report

        # Check specific values
        assert "Total Canvases: 100" in report
        assert "Success Rate: 100.0%" in report
        assert "Average Response Time: 200.00ms" in report
        assert "Lifecycle: 50" in report

    def test_generate_summary_report_with_errors(self):
        """Test summary report generation with API errors."""
        canvas_list_stats = {
            'total_canvases': 50,
            'total_tags': 10,
            'recent_activity_count': 2,
            'canvases_without_tags': 1,
            'most_common_tags': [('Lifecycle', 25)]
        }

        canvas_details_stats = {'error': 'No canvas details available'}

        request_log = [
            RequestLog(
                timestamp='2023-01-01T12:00:00',
                endpoint='/canvas/list',
                method='GET',
                status_code=200,
                response_time_ms=150.0,
                success=True
            ),
            RequestLog(
                timestamp='2023-01-01T12:01:00',
                endpoint='/canvas/details',
                method='GET',
                status_code=500,
                response_time_ms=1000.0,
                success=False,
                error_message='Server Error'
            )
        ]

        report = CanvasStatistics.generate_summary_report(
            canvas_list_stats,
            canvas_details_stats,
            request_log
        )

        # Should include API performance but not detailed analysis
        assert "Success Rate: 50.0%" in report
        assert "DETAILED ANALYSIS:" not in report

    def test_generate_summary_report_empty_request_log(self):
        """Test summary report with empty request log."""
        canvas_list_stats = {'total_canvases': 0, 'total_tags': 0, 'recent_activity_count': 0, 'canvases_without_tags': 0, 'most_common_tags': []}
        canvas_details_stats = {'error': 'No canvas details available'}

        report = CanvasStatistics.generate_summary_report(
            canvas_list_stats,
            canvas_details_stats,
            []
        )

        assert "Total API Requests: 0" in report
        assert "Success Rate: N/A" in report