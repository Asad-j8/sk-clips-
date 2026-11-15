"""Optional Remotion animated caption service wrapper.

Features:
- Opt-in via caption_engine='remotion'. Default remains 'ffmpeg'.
- Connects to Remotion HTTP service at port 3100.
- If Remotion service is down, timed out, or not installed, automatically falls back
  to FFmpeg caption generation and informs the user cleanly without crashing.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from clipforge.stages.captions import generate_ass_captions

logger = logging.getLogger(__name__)

DEFAULT_REMOTION_URL = "http://localhost:3100/render-captions"


@dataclass(frozen=True)
class CaptionRenderResult:
    """Result of the caption rendering stage."""

    engine_used: str  # 'remotion' or 'ffmpeg'
    ass_path: Path | None
    overlay_video_path: Path | None
    notice: str | None = None


def render_captions_with_fallback(
    words: Sequence[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_ass_path: Path | str,
    preset_name: str = "bold_pop",
    caption_engine: str = "ffmpeg",
    copy_language: str = "en",
    remotion_url: str = DEFAULT_REMOTION_URL,
    timeout_sec: float = 3.0,
) -> CaptionRenderResult:
    """Render captions using specified engine, falling back to FFmpeg if Remotion fails.

    Args:
        words: List of {'word': str, 'start': float, 'end': float}.
        clip_start: Clip start time in seconds.
        clip_end: Clip end time in seconds.
        output_ass_path: Output path for ASS subtitles file.
        preset_name: Caption preset name.
        caption_engine: 'ffmpeg' (default) or 'remotion'.
        copy_language: Language code for font/RTL rules.
        remotion_url: URL of the Remotion microservice.
        timeout_sec: Network timeout for checking Remotion service.

    Returns:
        CaptionRenderResult detailing engine used and any fallback notice.
    """
    engine_requested = caption_engine.strip().lower()

    if engine_requested == "remotion":
        logger.info("Attempting Remotion animated captions via service at %s", remotion_url)
        payload = json.dumps({
            "preset": preset_name,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "words": words,
            "copy_language": copy_language,
        }).encode("utf-8")

        req = urllib.request.Request(
            remotion_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                overlay_path_str = data.get("overlay_video_path")
                if overlay_path_str and Path(overlay_path_str).is_file():
                    return CaptionRenderResult(
                        engine_used="remotion",
                        ass_path=None,
                        overlay_video_path=Path(overlay_path_str),
                        notice=None,
                    )
        except Exception as e:
            logger.warning(
                "Remotion service unavailable or timed out (%s). Falling back to FFmpeg captions.",
                e,
            )
            # Graceful fallback to FFmpeg
            ass_path = generate_ass_captions(
                words=words,
                clip_start=clip_start,
                clip_end=clip_end,
                output_path=output_ass_path,
                preset_name=preset_name,
                copy_language=copy_language,
            )
            return CaptionRenderResult(
                engine_used="ffmpeg",
                ass_path=ass_path,
                overlay_video_path=None,
                notice="Animated captions service unreachable. Fell back to FFmpeg caption rendering.",
            )

    # Default FFmpeg caption path
    ass_path = generate_ass_captions(
        words=words,
        clip_start=clip_start,
        clip_end=clip_end,
        output_path=output_ass_path,
        preset_name=preset_name,
        copy_language=copy_language,
    )
    return CaptionRenderResult(
        engine_used="ffmpeg",
        ass_path=ass_path,
        overlay_video_path=None,
        notice=None,
    )
