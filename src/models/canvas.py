"""Canvas-related data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Canvas:
    """Represents a Braze canvas from the list endpoint."""
    id: str
    name: str
    tags: List[str]
    last_edited: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Canvas:
        return cls(
            id=data['id'],
            name=data['name'],
            tags=data['tags'],
            last_edited=data['last_edited']
        )


@dataclass
class CanvasListResponse:
    """Response from the /canvas/list endpoint."""
    canvases: List[Canvas]
    message: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasListResponse:
        canvases = [Canvas.from_dict(canvas_data) for canvas_data in data['canvases']]
        return cls(
            canvases=canvases,
            message=data['message']
        )


@dataclass
class CanvasVariant:
    """Represents a variant in a canvas."""
    id: str
    name: str
    first_step_ids: List[str]
    first_step_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasVariant:
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            first_step_ids=data.get('first_step_ids', []),
            first_step_id=data.get('first_step_id', '')
        )


@dataclass
class CanvasStepPath:
    """Represents a path in a canvas step."""
    name: str
    next_step_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasStepPath:
        return cls(
            name=data.get('name', ''),
            next_step_id=data.get('next_step_id', '')
        )


@dataclass
class CanvasStep:
    """Represents a step in a canvas workflow."""
    id: str
    name: str
    type: str
    next_paths: List[CanvasStepPath]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasStep:
        next_paths_data = data.get('next_paths', [])
        next_paths = [CanvasStepPath.from_dict(path_data) for path_data in next_paths_data] if next_paths_data else []

        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            type=data.get('type', ''),
            next_paths=next_paths
        )


@dataclass
class CanvasDetails:
    """Detailed information about a specific canvas."""
    canvas_id: str  # We'll set this manually since it's not in the response
    name: str
    description: str
    created_at: str
    updated_at: str
    archived: bool
    draft: bool
    enabled: bool
    has_post_launch_draft: bool
    schedule_type: str
    first_entry: str
    last_entry: str
    channels: List[str]
    tags: List[str]
    teams: List[str]
    variants: List[CanvasVariant]
    steps: List[CanvasStep]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], canvas_id: str = '') -> CanvasDetails:
        variants_data = data.get('variants', [])
        variants = [CanvasVariant.from_dict(variant_data) for variant_data in variants_data] if variants_data else []

        steps_data = data.get('steps', [])
        steps = [CanvasStep.from_dict(step_data) for step_data in steps_data] if steps_data else []

        return cls(
            canvas_id=canvas_id,
            name=data.get('name', ''),
            description=data.get('description', ''),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            archived=data.get('archived', False),
            draft=data.get('draft', False),
            enabled=data.get('enabled', False),
            has_post_launch_draft=data.get('has_post_launch_draft', False),
            schedule_type=data.get('schedule_type', ''),
            first_entry=data.get('first_entry', ''),
            last_entry=data.get('last_entry', ''),
            channels=data.get('channels', []),
            tags=data.get('tags', []),
            teams=data.get('teams', []),
            variants=variants,
            steps=steps
        )