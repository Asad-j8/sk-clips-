"""Tests for Phase 2: YouTube Link Ingest.

Tests:
- --no-playlist enforcement (noplaylist=True)
- User cookies first, fallback to anonymous try
- YouTube bot-check / CAPTCHA clear error handling (no app crash)
- Avoidance of proxies and cookie rotation
"""

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from clipforge.stages.downloader import (
    DownloadError,
    _execute_yt_dlp_download,
    download_or_copy_video,
    format_user_error,
    is_bot_check_error,
)


class TestYouTubeIngest(unittest.TestCase):
    """Test suite for YouTube video ingest rules."""

    def setUp(self):
        self.test_dir = Path("/data/user/0/com.termux/files/usr/tmp/clipforge_test_phase2_ingest")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file = self.test_dir / "user_cookies.txt"
        self.cookies_file.write_text("# Netscape HTTP Cookie File\n.youtube.com TRUE / FALSE 1999999999 SID fake_session\n")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("yt_dlp.YoutubeDL")
    def test_noplaylist_enforced(self, mock_ydl_cls):
        """Verify noplaylist is set to True."""
        mock_instance = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        target = self.test_dir / "source.mp4"
        _execute_yt_dlp_download("https://www.youtube.com/watch?v=sample", target)

        args, kwargs = mock_ydl_cls.call_args
        opts = args[0] if args else kwargs.get("params", {})
        self.assertTrue(opts.get("noplaylist"))

    def test_bot_check_detection(self):
        """Verify bot-check and CAPTCHA patterns are identified."""
        self.assertTrue(is_bot_check_error("Sign in to confirm you’re not a bot"))
        self.assertTrue(is_bot_check_error("Sign in to confirm you are not a bot"))
        self.assertTrue(is_bot_check_error("HTTP Error 429: Too Many Requests"))
        self.assertTrue(is_bot_check_error("Solving captcha required"))
        self.assertFalse(is_bot_check_error("Video unavailable: deleted by uploader"))

    def test_bot_check_clear_user_message(self):
        """Verify clear actionable message on bot detection without crash."""
        msg = format_user_error("ERROR: Sign in to confirm you’re not a bot")
        self.assertIn("YouTube Bot Detection / Verification Triggered", msg)
        self.assertIn("cookies.txt", msg)
        self.assertNotIn("Traceback", msg)

    @patch("clipforge.stages.downloader._execute_yt_dlp_download")
    @patch("clipforge.stages.downloader.get_video_metadata")
    def test_cookies_first_then_anonymous_on_failure(self, mock_meta, mock_exec):
        """Verify it first attempts with user cookies, and if that fails, tries anonymous."""
        import yt_dlp

        target_file = self.test_dir / "source.mp4"

        # First call (with cookies) fails with DownloadError, second call (anonymous) succeeds
        def exec_side_effect(url, target_path, cookies_path=None):
            if cookies_path:
                raise yt_dlp.utils.DownloadError("Cookie expired")
            # Anonymous call succeeds and creates source.mp4
            target_path.touch()

        mock_exec.side_effect = exec_side_effect
        mock_meta.return_value = MagicMock(duration=10.0, fps=30.0, width=1920, height=1080, file_path=str(target_file))

        download_or_copy_video(
            "https://www.youtube.com/watch?v=sample",
            output_dir=self.test_dir,
            user_cookies_path=self.cookies_file,
        )

        self.assertEqual(mock_exec.call_count, 2)
        # Call 1: cookies_path was passed
        self.assertEqual(mock_exec.call_args_list[0].kwargs["cookies_path"], self.cookies_file)
        # Call 2: cookies_path was None (anonymous)
        self.assertIsNone(mock_exec.call_args_list[1].kwargs["cookies_path"])

    @patch("clipforge.stages.downloader._execute_yt_dlp_download")
    def test_bot_check_raises_clear_download_error_no_crash(self, mock_exec):
        """Verify bot-check failure raises clean DownloadError with no crash."""
        import yt_dlp

        mock_exec.side_effect = yt_dlp.utils.DownloadError("ERROR: Sign in to confirm you're not a bot")

        with self.assertRaises(DownloadError) as ctx:
            download_or_copy_video(
                "https://www.youtube.com/watch?v=sample",
                output_dir=self.test_dir,
            )

        err_text = str(ctx.exception)
        self.assertIn("YouTube Bot Detection", err_text)


if __name__ == "__main__":
    unittest.main()
