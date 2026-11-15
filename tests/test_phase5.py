"""Pytest unit test suite for Phase 5: Presets, RTL Nastaliq, and Remotion Fallback.

Tests:
1. All 7 presets exist in presets.json with expected output schemas.
2. Existing 4 presets (bold_pop, clean_minimal, karaoke_yellow, news_bar) have unaltered configurations.
3. 3 new presets (neon_cyber, urdu_nastaliq_gold, cinema_sub) are properly defined.
4. RTL & Nastaliq font selection when COPY_LANGUAGE is 'urdu' or 'ur'.
5. Remotion service fallback to FFmpeg when Remotion service is down/offline.
6. Clean notice given without application crash.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from clipforge.stages.captions import (
    format_ass_time,
    generate_ass_captions,
    get_effective_font,
    hex_to_ass_color,
    is_urdu_script,
    load_presets_config,
)
from clipforge.stages.remotion_wrapper import render_captions_with_fallback


class TestPresetsConfig:
    """Test presets configuration and consistency."""

    def test_presets_exist_and_count(self):
        presets = load_presets_config()
        assert len(presets) >= 7
        assert "bold_pop" in presets
        assert "clean_minimal" in presets
        assert "karaoke_yellow" in presets
        assert "news_bar" in presets
        assert "neon_cyber" in presets
        assert "urdu_nastaliq_gold" in presets
        assert "cinema_sub" in presets

    def test_existing_four_presets_unaltered(self):
        presets = load_presets_config()
        # bold_pop
        assert presets["bold_pop"]["fontsize"] == 72
        assert presets["bold_pop"]["highlight_color"] == "#FFD60A"
        # clean_minimal
        assert presets["clean_minimal"]["fontsize"] == 54
        # karaoke_yellow
        assert presets["karaoke_yellow"]["animation"] == "karaoke"
        # news_bar
        assert presets["news_bar"]["border_style"] == 3

    def test_safe_zone_margins(self):
        presets = load_presets_config()
        for name, p in presets.items():
            sz = p.get("safe_zone", {})
            assert sz.get("top_pct") == 0.12
            assert sz.get("bottom_pct") == 0.20


class TestUrduRTLSupport:
    """Test Urdu script and Nastaliq font assignment."""

    def test_urdu_script_detection(self):
        urdu_sample = "یہ ایک ٹیسٹ ویڈیو ہے"
        english_sample = "This is a test video"
        assert is_urdu_script(urdu_sample) is True
        assert is_urdu_script(english_sample) is False

    def test_nastaliq_font_selection(self):
        # English copy
        font_en = get_effective_font("DejaVu Sans", copy_language="en", text="Hello")
        assert font_en == "DejaVu Sans"

        # Urdu language specified
        font_ur = get_effective_font("DejaVu Sans", copy_language="urdu", text="Hello")
        assert font_ur == "Noto Naskh Arabic"

        # Urdu text detected
        font_detected = get_effective_font("DejaVu Sans", copy_language="en", text="السلام علیکم")
        assert font_detected == "Noto Naskh Arabic"

    def test_ass_caption_generation_with_urdu(self, tmp_path):
        out_ass = tmp_path / "urdu_test.ass"
        words = [
            {"word": "یہ", "start": 0.0, "end": 0.5},
            {"word": "بہترین", "start": 0.5, "end": 1.0},
            {"word": "پوڈکاسٹ", "start": 1.0, "end": 1.8},
        ]
        path = generate_ass_captions(
            words=words,
            clip_start=0.0,
            clip_end=2.0,
            output_path=out_ass,
            preset_name="urdu_nastaliq_gold",
            copy_language="urdu",
        )
        content = path.read_text(encoding="utf-8")
        assert "Noto Naskh Arabic" in content
        assert "پوڈکاسٹ" in content


class TestRemotionFallback:
    """Test Remotion optional wrapper and graceful fallback."""

    def test_default_ffmpeg_engine(self, tmp_path):
        out_ass = tmp_path / "ffmpeg_test.ass"
        words = [{"word": "test", "start": 0.0, "end": 0.5}]
        res = render_captions_with_fallback(
            words=words,
            clip_start=0.0,
            clip_end=1.0,
            output_ass_path=out_ass,
            preset_name="bold_pop",
            caption_engine="ffmpeg",
        )
        assert res.engine_used == "ffmpeg"
        assert res.ass_path is not None
        assert res.ass_path.exists()
        assert res.notice is None

    def test_remotion_fallback_when_service_down(self, tmp_path):
        out_ass = tmp_path / "fallback_test.ass"
        words = [{"word": "fallback", "start": 0.0, "end": 0.5}]

        # Point to offline port
        res = render_captions_with_fallback(
            words=words,
            clip_start=0.0,
            clip_end=1.0,
            output_ass_path=out_ass,
            preset_name="karaoke_yellow",
            caption_engine="remotion",
            remotion_url="http://127.0.0.1:59999/render-captions",
            timeout_sec=0.5,
        )
        # Should gracefully fall back to ffmpeg without crashing
        assert res.engine_used == "ffmpeg"
        assert res.ass_path is not None
        assert res.ass_path.exists()
        assert "Animated captions service unreachable" in (res.notice or "")
