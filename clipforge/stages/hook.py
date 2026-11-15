"""Hook generation and overlay module (Phase 3b).

Features:
- Hook teaser on/off toggle.
- Grounded hook: Extracted strictly from transcript, 10 words or fewer.
- Hook overlay placed in safe zones (top 12%, avoiding top header and lower captions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookOverlayConfig:
    """Configuration for clip hook title overlay."""

    enabled: bool
    text: str
    max_words: int = 10
    duration_sec: float = 3.5  # First 3.5s of the clip


def sanitize_hook_text(raw_text: str, max_words: int = 10) -> str:
    """Trim hook text strictly to 10 words or fewer and clean punctuation."""
    words = raw_text.strip().split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


def generate_hook_ass(
    hook_text: str,
    output_path: Path | str,
    duration_sec: float = 3.5,
    font_name: str = "DejaVu Sans",
    font_size: int = 68,
    margin_top_px: int = 240,  # Top safe zone margin
) -> Path:
    """Generate ASS subtitle file displaying hook title in upper-third safe zone."""
    clean_text = sanitize_hook_text(hook_text, max_words=10)

    ass_content = f"""[Script Info]
Title: ClipForge Hook
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HookStyle,{font_name},{font_size},&H0000FFFF,&H00000000,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,5,2,8,80,80,{margin_top_px},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.00,0:00:0{duration_sec:04.2f},HookStyle,,0,0,0,,{{\\fad(200,200)}}{clean_text}
"""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ass_content, encoding="utf-8")
    return out
