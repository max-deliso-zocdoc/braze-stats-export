"""Statistics and analytics for canvas data."""

from datetime import datetime
from typing import Dict, Any, List

from ..models import CanvasListResponse, CanvasDetails, RequestLog
from ..utils import generate_canvas_url


class CanvasStatistics:
    """Generate statistics and insights from canvas data."""

    @staticmethod
    def analyze_canvas_list(canvas_list: CanvasListResponse) -> Dict[str, Any]:
        """Generate statistics from the canvas list."""
        canvases = canvas_list.canvases

        # Tag frequency analysis
        tag_counts = {}
        for canvas in canvases:
            for tag in canvas.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Recent activity analysis (last 30 days)
        recent_canvases = []
        now = datetime.now()
        for canvas in canvases:
            try:
                last_edited = datetime.fromisoformat(
                    canvas.last_edited.replace("Z", "+00:00")
                )
                days_ago = (now - last_edited.replace(tzinfo=None)).days
                if days_ago <= 30:
                    recent_canvases.append(canvas)
            except ValueError:
                continue

        stats = {
            "total_canvases": len(canvases),
            "total_tags": len(tag_counts),
            "most_common_tags": sorted(
                tag_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "recent_activity_count": len(recent_canvases),
            "canvases_without_tags": len([c for c in canvases if not c.tags]),
            "tag_distribution": tag_counts,
        }

        return stats

    @staticmethod
    def analyze_canvas_details(canvas_details: List[CanvasDetails]) -> Dict[str, Any]:
        """Generate detailed statistics from canvas details."""
        if not canvas_details:
            return {"error": "No canvas details available"}

        # Channel distribution
        channel_counts = {}
        for canvas in canvas_details:
            for channel in canvas.channels:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1

        # Schedule type distribution
        schedule_types = {}
        for canvas in canvas_details:
            schedule_type = canvas.schedule_type or "unknown"
            schedule_types[schedule_type] = schedule_types.get(schedule_type, 0) + 1

        # Step type analysis
        step_type_counts = {}
        total_steps = 0
        for canvas in canvas_details:
            total_steps += len(canvas.steps)
            for step in canvas.steps:
                step_type = step.type or "unknown"
                step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1

        # Canvas status analysis
        status_counts = {
            "enabled": len([c for c in canvas_details if c.enabled]),
            "disabled": len([c for c in canvas_details if not c.enabled]),
            "archived": len([c for c in canvas_details if c.archived]),
            "draft": len([c for c in canvas_details if c.draft]),
            "has_post_launch_draft": len(
                [c for c in canvas_details if c.has_post_launch_draft]
            ),
        }

        # Complexity analysis (steps per canvas)
        step_counts = [len(canvas.steps) for canvas in canvas_details]
        avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0

        stats = {
            "total_analyzed": len(canvas_details),
            "channel_distribution": channel_counts,
            "schedule_type_distribution": schedule_types,
            "step_type_distribution": step_type_counts,
            "status_distribution": status_counts,
            "complexity_analysis": {
                "total_steps": total_steps,
                "average_steps_per_canvas": round(avg_steps, 2),
                "min_steps": min(step_counts) if step_counts else 0,
                "max_steps": max(step_counts) if step_counts else 0,
            },
        }

        return stats

    @staticmethod
    def generate_summary_report(
        canvas_list_stats: Dict[str, Any],
        canvas_details_stats: Dict[str, Any],
        request_log: List[RequestLog],
        canvas_details: List[CanvasDetails] = None,
    ) -> str:
        """Generate a comprehensive summary report."""

        report = []
        report.append("=" * 70)
        report.append("BRAZE CANVAS EXPORT SUMMARY REPORT")
        report.append("=" * 70)
        report.append("")

        # API Performance
        report.append("🔗 API PERFORMANCE:")
        total_requests = len(request_log)
        successful_requests = len([r for r in request_log if r.success])
        avg_response_time = (
            sum(r.response_time_ms for r in request_log) / total_requests
            if total_requests > 0
            else 0
        )

        report.append(f"  • Total API Requests: {total_requests}")
        report.append(f"  • Successful Requests: {successful_requests}")
        report.append(
            f"  • Success Rate: {(successful_requests/total_requests*100):.1f}%"
            if total_requests > 0
            else "  • Success Rate: N/A"
        )
        report.append(f"  • Average Response Time: {avg_response_time:.2f}ms")
        report.append("")

        # Canvas Overview
        report.append("📊 CANVAS OVERVIEW:")
        report.append(f"  • Total Canvases: {canvas_list_stats['total_canvases']}")
        report.append(f"  • Unique Tags: {canvas_list_stats['total_tags']}")
        report.append(
            f"  • Recently Updated (30 days): {canvas_list_stats['recent_activity_count']}"
        )
        report.append(
            f"  • Canvases Without Tags: {canvas_list_stats['canvases_without_tags']}"
        )
        report.append("")

        # Most Common Tags
        report.append("🏷️  MOST COMMON TAGS:")
        for tag, count in canvas_list_stats["most_common_tags"][:5]:
            report.append(f"  • {tag}: {count}")
        report.append("")

        # Detailed Analysis (if available)
        if "error" not in canvas_details_stats:
            report.append("🔍 DETAILED ANALYSIS:")
            report.append(
                f"  • Canvases Analyzed: {canvas_details_stats['total_analyzed']}"
            )

            report.append("  • Status Distribution:")
            for status, count in canvas_details_stats["status_distribution"].items():
                if count > 0:
                    report.append(f"    - {status.replace('_', ' ').title()}: {count}")

            report.append("  • Channel Distribution:")
            for channel, count in canvas_details_stats["channel_distribution"].items():
                report.append(f"    - {channel.title()}: {count}")

            report.append("  • Schedule Types:")
            for schedule_type, count in canvas_details_stats[
                "schedule_type_distribution"
            ].items():
                report.append(
                    f"    - {schedule_type.replace('_', ' ').title()}: {count}"
                )

            complexity = canvas_details_stats["complexity_analysis"]
            report.append("  • Complexity Metrics:")
            report.append(f"    - Total Steps: {complexity['total_steps']}")
            report.append(
                f"    - Avg Steps per Canvas: {complexity['average_steps_per_canvas']}"
            )
            report.append(
                f"    - Steps Range: {complexity['min_steps']} - {complexity['max_steps']}"
            )
            report.append("")

            # Add Canvas Dashboard Links section if canvas details are provided
            if canvas_details:
                report.append("🌐 CANVAS DASHBOARD LINKS:")
                for detail in canvas_details:
                    canvas_url = generate_canvas_url(detail.canvas_id)
                    status_icon = "✅" if detail.enabled else "❌"
                    report.append(
                        f"  • {status_icon} {detail.name[:35]:35} | {canvas_url}"
                    )
                report.append("")

        report.append("=" * 70)

        return "\n".join(report)
