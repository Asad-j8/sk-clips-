"""Video download and metadata extraction stage for ClipForge.

Uses yt-dlp to download video or copies local video to the job folder.
Extracts video metadata (duration, fps, width, height).
Includes robust YouTube ingest strategy:
- Enforces --no-playlist
- First attempts using user-provided cookies if available
- Falls back to anonymous attempt if cookies fail or are absent
- Detects YouTube bot-check / sign-in / CAPTCHA blocks and shows clear actionable error (no crash)
- Respects LOADER_TO_ENABLED (default OFF)
- Strictly avoids multiple account cookie rotation or paid proxies
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clipforge.config import get_config

logger = logging.getLogger(__name__)

# Patterns indicative of bot-detection, rate-limiting or sign-in blocks
BOT_CHECK_PATTERNS = [
    r"sign in to confirm you['’]?re not a bot",
    r"sign in to confirm you are not a bot",
    r"confirm you['’]?re not a bot",
    r"confirm you are not a bot",
    r"bot detection",
    r"captcha",
    r"http error 429",
    r"too many requests",
    r"automated queries",
    r"sign in if you['’]?ve been granted access",
]



class DownloadError(Exception):
    """User-facing error raised during the video downloading stage."""


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata of the downloaded or processed source video."""

    duration: float
    fps: float
    width: int
    height: int
    file_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)


def get_video_metadata(video_path: Path | str) -> VideoMetadata:
    """Extract duration, fps, width, and height using ffprobe or OpenCV fallback.

    Args:
        video_path: Path to the target video file.

    Returns:
        VideoMetadata containing duration, fps, width, height, and file_path.

    Raises:
        DownloadError: If metadata cannot be extracted from the video file.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise DownloadError(f"Video file not found at: {path}")

    # 1. Try ffprobe if available
    ffprobe_cmd = shutil.which("ffprobe")
    if ffprobe_cmd:
        try:
            cmd = [
                ffprobe_cmd,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,duration:format=duration",
                "-of",
                "json",
                str(path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(res.stdout)
            streams = probe_data.get("streams", [])
            format_data = probe_data.get("format", {})

            if streams:
                v_stream = streams[0]
                width = int(v_stream.get("width", 0))
                height = int(v_stream.get("height", 0))

                fps_str = v_stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/", 1)
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                else:
                    fps = float(fps_str) if fps_str else 30.0

                dur_str = v_stream.get("duration") or format_data.get("duration")
                duration = float(dur_str) if dur_str else 0.0

                if width > 0 and height > 0:
                    return VideoMetadata(
                        duration=duration,
                        fps=fps,
                        width=width,
                        height=height,
                        file_path=str(path),
                    )
        except Exception as e:
            logger.warning("ffprobe probe failed, falling back to cv2: %s", e)

    # 2. Try OpenCV cv2 fallback
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps <= 0:
                fps = 30.0
            duration = frame_count / fps if fps > 0 else 0.0

            if width > 0 and height > 0:
                return VideoMetadata(
                    duration=duration,
                    fps=fps,
                    width=width,
                    height=height,
                    file_path=str(path),
                )
    except Exception as e:
        logger.warning("cv2 probe failed: %s", e)

    raise DownloadError(f"Could not read video metadata for file: {path}")


def is_bot_check_error(error_message: str) -> bool:
    """Check if the error string indicates YouTube bot verification or rate-limiting."""
    msg = error_message.lower()
    return any(re.search(pat, msg) for pat in BOT_CHECK_PATTERNS)


def format_user_error(error_message: str) -> str:
    """Format yt-dlp error message into a clean, actionable user message."""
    raw = str(error_message)
    if "ERROR:" in raw:
        clean = raw.split("ERROR:", 1)[1].strip()
    else:
        clean = raw.strip()

    if is_bot_check_error(clean):
        return (
            "YouTube Bot Detection / Verification Triggered: "
            "YouTube requested bot verification or sign-in for this video. "
            "Please export your YouTube browser cookies to a 'cookies.txt' file "
            "and set YOUTUBE_COOKIES_PATH in .env or place cookies.txt in the project directory."
        )

    return f"Video download failed: {clean}"


def _execute_yt_dlp_download(
    url: str,
    target_path: Path,
    cookies_path: Path | str | None = None,
) -> None:
    """Execute yt-dlp download with specified options.

    Args:
        url: The video URL.
        target_path: Desired output video path.
        cookies_path: Optional path to cookies file.
    """
    import yt_dlp

    ydl_opts: dict[str, Any] = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best[height<=1080]/best",
        "outtmpl": str(target_path.with_suffix(".%(ext)s")),
        "merge_output_format": "mp4",
        "noplaylist": True,  # Enforce --no-playlist
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
    }

    if cookies_path:
        c_path = Path(cookies_path).resolve()
        if c_path.is_file():
            ydl_opts["cookiefile"] = str(c_path)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_or_copy_video(
    source: str | Path,
    output_dir: Path | str,
    target_filename: str = "source.mp4",
    user_cookies_path: Path | str | None = None,
) -> VideoMetadata:
    """Download a video via yt-dlp or copy a local video to output_dir/source.mp4.

    Strategy for URLs:
    1. If user cookies are provided or configured in .env, try downloading with cookies first.
    2. If cookies attempt fails (or no cookies provided), try one anonymous download.
    3. If bot-check or sign-in is required, report a clear informative error without crashing.
    4. Loader.to fallback remains default OFF unless explicitly enabled.

    Args:
        source: URL string or local file path to the video.
        output_dir: Job directory where source.mp4 will be stored.
        target_filename: Desired target filename (defaults to "source.mp4").
        user_cookies_path: Optional explicit cookies path override.

    Returns:
        VideoMetadata with duration, fps, width, height, and file_path.

    Raises:
        DownloadError: If yt-dlp fails or local file is missing/invalid.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / target_filename

    source_str = str(source).strip()

    # Case 1: Local file path
    local_path = Path(source_str)
    if local_path.is_file():
        try:
            if local_path.resolve() != target_path.resolve():
                shutil.copy2(local_path, target_path)
        except Exception as e:
            raise DownloadError(f"Failed to copy local video file: {e}") from e

        return get_video_metadata(target_path)

    # Case 2: URL download
    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError("yt-dlp is not installed in the Python environment.") from e

    config = get_config()
    # Determine user cookies path
    cookies_to_try = user_cookies_path or config.cookies_path

    attempt_errors: list[str] = []

    # Attempt 1: Try with user's own cookies if configured
    if cookies_to_try and Path(cookies_to_try).is_file():
        logger.info("Attempting video download using user cookies from: %s", cookies_to_try)
        try:
            _execute_yt_dlp_download(source_str, target_path, cookies_path=cookies_to_try)
        except yt_dlp.utils.DownloadError as e:
            logger.warning("Download with user cookies failed: %s. Trying anonymous...", e)
            attempt_errors.append(str(e))
        except Exception as e:
            logger.warning("Unexpected error with cookies: %s. Trying anonymous...", e)
            attempt_errors.append(str(e))

    # Attempt 2: Anonymous try (if not already downloaded)
    if not (target_path.exists() or any(out_dir.glob("source.*"))):
        logger.info("Attempting anonymous video download (no cookies)...")
        try:
            _execute_yt_dlp_download(source_str, target_path, cookies_path=None)
        except yt_dlp.utils.DownloadError as e:
            attempt_errors.append(str(e))
        except Exception as e:
            attempt_errors.append(str(e))

    # Check if download succeeded and resolve file
    if not target_path.exists():
        candidates = list(out_dir.glob("source.*"))
        video_candidates = [
            c for c in candidates if c.suffix.lower() in (".mp4", ".mkv", ".webm")
        ]
        if video_candidates:
            shutil.move(str(video_candidates[0]), str(target_path))
        else:
            # All attempts failed - inspect the last error
            last_err = attempt_errors[-1] if attempt_errors else "Unknown download error"
            raise DownloadError(format_user_error(last_err))

    return get_video_metadata(target_path)
