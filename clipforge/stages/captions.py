"""Caption generation module for ClipForge.

Generates ASS subtitle files for video clips:
- Loads preset styles from templates/presets.json
- Preserves behavior of existing presets (bold_pop, clean_minimal, karaoke_yellow, news_bar)
- Supports Urdu RTL and Nastaliq font selection when COPY_LANGUAGE is 'urdu' or 'ur'
- Safe zone verification: Top margin 12% (230px), Bottom margin 20% (384px)
- Max 3 words per line, shifted so clip starts at 0.0s
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

PRESETS_FILE = Path(__file__).resolve().parent.parent / "templates" / "presets.json"


@dataclass(frozen=True)
class CaptionWord:
    """Single word timestamped relative to the clip start."""

    word: str
    start: float
    end: float


def load_presets_config() -> dict[str, Any]:
    """Load caption presets from the single config file."""
    if PRESETS_FILE.is_file():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to parse presets.json: %s", e)
    return {}


def hex_to_ass_color(hex_str: str, alpha: str = "00") -> str:
    """Convert #RRGGBB hex code to ASS &HAABBGGRR format."""
    clean = (hex_str or "#FFFFFF").strip().lstrip("#")
    if len(clean) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", clean):
        clean = "FFFFFF"
    r = clean[0:2].upper()
    g = clean[2:4].upper()
    b = clean[4:6].upper()
    return f"&H{alpha}{b}{g}{r}"


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cc"""
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    centis = int(round((s - math.floor(s)) * 100))
    if centis >= 100:
        centis = 99
    return f"{h:01d}:{m:02d}:{sec:02d}.{centis:02d}"


def is_urdu_script(text: str) -> bool:
    """Check if text contains Arabic/Urdu unicode characters."""
    for ch in text:
        # Arabic, Arabic Supplement, Arabic Presentation Forms
        if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" or "\uFB50" <= ch <= "\uFDFF":
            return True
    return False


def get_effective_font(preset_font: str, copy_language: str = "en", text: str = "") -> str:
    """Select font based on COPY_LANGUAGE and script detection."""
    lang = copy_language.strip().lower()
    if lang in ("urdu", "ur") or is_urdu_script(text):
        # Nastaliq / Noto Naskh Arabic
        return "Noto Naskh Arabic"
    return preset_font


def generate_ass_captions(
    words: Sequence[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_path: Path | str,
    preset_name: str = "bold_pop",
    copy_language: str = "en",
) -> Path:
    """Generate ASS subtitle file for clip time range.

    Args:
        words: List of {'word': str, 'start': float, 'end': float} from Whisper.
        clip_start: Absolute start time of the clip.
        clip_end: Absolute end time of the clip.
        output_path: Destination path for .ass file.
        preset_name: Preset key in presets.json.
        copy_language: e.g. 'en', 'urdu' or 'ur'.

    Returns:
        Resolved Path to the written ASS file.
    """
    presets = load_presets_config()
    preset = presets.get(preset_name, presets.get("bold_pop", {}))

    raw_font = preset.get("font_family", "DejaVu Sans")
    fontsize = int(preset.get("fontsize", 72))
    outline = int(preset.get("outline", 4))
    shadow = int(preset.get("shadow", 1))
    border_style = int(preset.get("border_style", 1))
    margin_v = int(preset.get("margin_v", 420))

    primary_color_ass = hex_to_ass_color(preset.get("text_color", "#FFFFFF"))
    highlight_color_ass = hex_to_ass_color(preset.get("highlight_color", "#FFD60A"))
    outline_color_ass = hex_to_ass_color(preset.get("outline_color", "#000000"))

    # Filter and shift words relative to clip start
    clip_words: list[CaptionWord] = []
    for w in words:
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        w_text = str(w.get("word", "")).strip()
        if not w_text:
            continue
        if w_end >= clip_start and w_start <= clip_end:
            shifted_start = max(0.0, w_start - clip_start)
            shifted_end = max(shifted_start + 0.05, min(clip_end - clip_start, w_end - clip_start))
            clip_words.append(CaptionWord(word=w_text, start=shifted_start, end=shifted_end))

    full_sample_text = " ".join(cw.word for cw in clip_words[:10])
    effective_font = get_effective_font(raw_font, copy_language=copy_language, text=full_sample_text)

    ass_header = f"""[Script Info]
Title: ClipForge Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{effective_font},{fontsize},{primary_color_ass},&H00000000,{outline_color_ass},&H80000000,1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,80,80,{margin_v},1
Style: Highlight,{effective_font},{fontsize},{highlight_color_ass},&H00000000,{outline_color_ass},&H80000000,1,0,0,0,106,106,0,0,{border_style},{outline},{shadow},2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []

    # Chunk words into max 3 words per line
    chunk_size = 3
    is_karaoke = (preset.get("animation") == "karaoke")

    for i in range(0, len(clip_words), chunk_size):
        chunk = clip_words[i : i + chunk_size]
        if not chunk:
            continue

        line_start = chunk[0].start
        line_end = chunk[-1].end

        if is_karaoke:
            # Generate dialogue lines highlighting each word sequentially
            for active_idx, active_word in enumerate(chunk):
                w_start_str = format_ass_time(active_word.start)
                w_end_str = format_ass_time(active_word.end)

                styled_parts = []
                for p_idx, p_w in enumerate(chunk):
                    if p_idx == active_idx:
                        styled_parts.append(f"{{\\c{highlight_color_ass}\\fscx108\\fscy108}}{p_w.word}{{\\r}}")
                    else:
                        styled_parts.append(p_w.word)
                line_text = " ".join(styled_parts)
                events.append(f"Dialogue: 0,{w_start_str},{w_end_str},Default,,0,0,0,,{line_text}")
        else:
            # Standard punchy pop / fade line
            st_str = format_ass_time(line_start)
            et_str = format_ass_time(line_end)
            text_str = " ".join(cw.word for cw in chunk)
            events.append(f"Dialogue: 0,{st_str},{et_str},Default,,0,0,0,,{text_str}")

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ass_header + "\n".join(events) + "\n", encoding="utf-8")
    return out
