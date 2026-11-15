"""Layout determination stage for ClipForge.

Layout choices:
- "none": Default talking head, action, or monologue.
- "screencast": Screen recordings, slides, or software tutorials.
- "split": Two speakers in the same shot speaking simultaneously or having a rapid dialogue.
- "camera_inset": Inset presenter webcam stacked with screen recording when presenter_cam is true.

Also integrates prompts.py LAYOUT_PROMPT_TEMPLATE and SHOT_PROMPT_TEMPLATE.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

from clipforge.prompts import LAYOUT_PROMPT_TEMPLATE, SHOT_PROMPT_TEMPLATE
from clipforge.stages.speaker import SpeakerInterval

logger = logging.getLogger(__name__)

LayoutMode = Literal["none", "screencast", "split", "camera_inset"]


@dataclass(frozen=True)
class LayoutSegment:
    """Reframed layout decision for a given video segment."""

    start: float
    end: float
    layout: LayoutMode
    primary_face_id: str | None = None
    secondary_face_id: str | None = None
    presenter_cam: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def determine_clip_layout(
    speaker_intervals: list[SpeakerInterval],
    is_screencast: bool = False,
    has_webcam_inset: bool = False,
    override_layout: LayoutMode | None = None,
) -> list[LayoutSegment]:
    """Determine reframed layout across segments.

    Rules:
    1. If explicit override_layout is provided, apply it.
    2. If is_screencast is True:
       - If has_webcam_inset is True, use 'camera_inset' (stack screen and webcam).
       - Otherwise, use 'screencast'.
    3. Multi-speaker rule:
       - When two speakers are talking simultaneously in the same shot (is_multi_speaker=True),
         layout MUST be 'split'. Omitting or cropping out one speaker while both are talking is strictly prohibited.
    4. Otherwise default to 'none' (standard single speaker, talking head, action).
    """
    if not speaker_intervals:
        # Default empty/fallback segment
        return [
            LayoutSegment(
                start=0.0,
                end=60.0,
                layout=override_layout or "none",
                reason="Default baseline layout",
            )
        ]

    segments: list[LayoutSegment] = []

    for interval in speaker_intervals:
        st = interval.start
        et = interval.end

        if override_layout:
            mode: LayoutMode = override_layout
            reason = f"Explicit override to {override_layout}"
        elif is_screencast:
            if has_webcam_inset:
                mode = "camera_inset"
                reason = "Screen recording with presenter webcam inset detected -> stacked view"
            else:
                mode = "screencast"
                reason = "Screencast / slide recording detected"
        elif interval.is_multi_speaker and interval.secondary_face_id:
            mode = "split"
            reason = "Two simultaneous speakers in shot -> split-screen to keep both visible"
        else:
            mode = "none"
            reason = "Standard talking head / single speaker"

        segments.append(
            LayoutSegment(
                start=st,
                end=et,
                layout=mode,
                primary_face_id=interval.active_face_id,
                secondary_face_id=interval.secondary_face_id,
                presenter_cam=has_webcam_inset if is_screencast else False,
                reason=reason,
            )
        )

    # Merge adjacent segments with identical layout and face IDs
    if not segments:
        return []

    merged: list[LayoutSegment] = []
    curr = segments[0]

    for nxt in segments[1:]:
        if (
            nxt.layout == curr.layout
            and nxt.primary_face_id == curr.primary_face_id
            and nxt.secondary_face_id == curr.secondary_face_id
            and nxt.presenter_cam == curr.presenter_cam
        ):
            curr = LayoutSegment(
                start=curr.start,
                end=nxt.end,
                layout=curr.layout,
                primary_face_id=curr.primary_face_id,
                secondary_face_id=curr.secondary_face_id,
                presenter_cam=curr.presenter_cam,
                reason=curr.reason,
            )
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)

    return merged


def build_layout_prompt(
    duration: float,
    face_count: int,
    is_screencast: bool,
    simultaneous_speakers: bool,
    summary: str,
) -> str:
    """Build Section 7.4 LAYOUT prompt using exact word-for-word template."""
    return LAYOUT_PROMPT_TEMPLATE.format(
        duration=round(duration, 1),
        face_count=face_count,
        is_screencast=str(is_screencast).lower(),
        simultaneous_speakers=str(simultaneous_speakers).lower(),
        summary=summary,
    )


def build_shot_prompt(
    shot_id: int,
    start_time: float,
    end_time: float,
    visual_description: str,
    has_webcam: bool,
) -> str:
    """Build Section 7.5 SHOT prompt using exact word-for-word template."""
    return SHOT_PROMPT_TEMPLATE.format(
        shot_id=shot_id,
        start_time=round(start_time, 2),
        end_time=round(end_time, 2),
        visual_description=visual_description,
        has_webcam=str(has_webcam).lower(),
    )
