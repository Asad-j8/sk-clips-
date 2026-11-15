"""Tests for Phase 1: Configuration, ffmpeg verification, and video downloader."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clipforge.config import AppConfig, ConfigurationError, check_ffmpeg
from clipforge.stages.downloader import (
    DownloadError,
    VideoMetadata,
    download_or_copy_video,
    get_video_metadata,
)


class TestConfig(unittest.TestCase):
    """Test configuration loading and ffmpeg availability."""

    def test_ffmpeg_check_success(self):
        """Verify ffmpeg is present on PATH."""
        ffmpeg_path = check_ffmpeg()
        self.assertTrue(os.path.exists(ffmpeg_path))
        self.assertTrue(os.access(ffmpeg_path, os.X_OK))

    def test_ffmpeg_check_missing(self):
        """Verify ConfigurationError when ffmpeg is not found."""
        with patch("shutil.which", return_value=None):
            with self.assertRaises(ConfigurationError) as ctx:
                check_ffmpeg()
            self.assertIn("ffmpeg is not found on PATH", str(ctx.exception))

    def test_app_config_load_valid(self):
        """Verify loading valid configuration values."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-1.5-flash",
                "LLM_API_KEY": "test_key",
                "WHISPER_MODEL": "base",
                "DEVICE": "cpu",
                "SMOOTHING_ALPHA": "0.2",
            },
        ):
            config = AppConfig.load()
            self.assertEqual(config.llm_provider, "gemini")
            self.assertEqual(config.llm_model, "gemini-1.5-flash")
            self.assertEqual(config.llm_api_key, "test_key")
            self.assertEqual(config.whisper_model, "base")
            self.assertEqual(config.device, "cpu")
            self.assertAlmostEqual(config.smoothing_alpha, 0.2)

    def test_app_config_invalid_provider(self):
        """Verify error on invalid LLM provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}):
            with self.assertRaises(ConfigurationError) as ctx:
                AppConfig.load()
            self.assertIn("Invalid LLM_PROVIDER", str(ctx.exception))

    def test_app_config_invalid_alpha(self):
        """Verify error on invalid smoothing alpha."""
        with patch.dict(os.environ, {"SMOOTHING_ALPHA": "2.5"}):
            with self.assertRaises(ConfigurationError) as ctx:
                AppConfig.load()
            self.assertIn("Invalid SMOOTHING_ALPHA", str(ctx.exception))


class TestDownloader(unittest.TestCase):
    """Test video downloading, local copying, and metadata extraction."""

    def setUp(self):
        self.test_dir = Path("/data/user/0/com.termux/files/usr/tmp/clipforge_test_phase1")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_video = self.test_dir / "sample.mp4"

        # Generate a synthetic 1-second 640x360 test video using ffmpeg
        ffmpeg_bin = shutil.which("ffmpeg")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(self.test_video),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_video_metadata(self):
        """Test metadata extraction from a local video."""
        meta = get_video_metadata(self.test_video)
        self.assertIsInstance(meta, VideoMetadata)
        self.assertEqual(meta.width, 640)
        self.assertEqual(meta.height, 360)
        self.assertAlmostEqual(meta.fps, 30.0, delta=1.0)
        self.assertGreaterEqual(meta.duration, 0.9)

    def test_download_or_copy_local_video(self):
        """Test copying a local video into a job directory."""
        job_dir = self.test_dir / "job_01"
        meta = download_or_copy_video(self.test_video, job_dir)

        target_file = job_dir / "source.mp4"
        self.assertTrue(target_file.exists())
        self.assertEqual(meta.width, 640)
        self.assertEqual(meta.height, 360)
        self.assertEqual(meta.file_path, str(target_file))

    def test_download_nonexistent_local_file(self):
        """Test error when local file does not exist."""
        job_dir = self.test_dir / "job_02"
        # Since it doesn't exist locally, it tries yt-dlp which fails gracefully
        with self.assertRaises(DownloadError) as ctx:
            download_or_copy_video(self.test_dir / "nonexistent.mp4", job_dir)
        self.assertIn("Video download failed", str(ctx.exception))

    @patch("yt_dlp.YoutubeDL")
    def test_yt_dlp_failure_user_friendly_message(self, mock_ydl_cls):
        """Test clean user-friendly error message on yt-dlp error."""
        import yt_dlp

        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.download.side_effect = yt_dlp.utils.DownloadError(
            "ERROR: [youtube] 12345: Video unavailable. This video has been removed by the uploader"
        )


        job_dir = self.test_dir / "job_03"
        with self.assertRaises(DownloadError) as ctx:
            download_or_copy_video("https://www.youtube.com/watch?v=12345", job_dir)

        error_message = str(ctx.exception)
        self.assertIn("[youtube] 12345: Video unavailable", error_message)
        self.assertNotIn("Traceback", error_message)



if __name__ == "__main__":
    unittest.main()
