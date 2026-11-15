"""Configuration management for ClipForge.

Loads settings from .env file or environment variables with strict validation.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv(dotenv_path=None, override=False):
        pass



class ConfigurationError(Exception):
    """Raised when environment or runtime configuration is invalid."""


def check_ffmpeg() -> str:
    """Verify that ffmpeg executable is available on the system PATH.

    Returns:
        The resolved path to the ffmpeg executable.

    Raises:
        ConfigurationError: If ffmpeg is not found on PATH.
    """
    path = shutil.which("ffmpeg")
    if not path:
        raise ConfigurationError(
            "ffmpeg is not found on PATH. Please install ffmpeg and make sure it is accessible in your environment PATH."
        )
    return path


def check_ffprobe() -> str | None:
    """Verify if ffprobe is available on PATH.

    Returns:
        The resolved path to ffprobe if found, else None.
    """
    return shutil.which("ffprobe")


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration."""

    llm_provider: Literal["anthropic", "gemini", "ollama"]
    llm_model: str
    llm_api_key: str
    whisper_model: str
    device: Literal["auto", "cpu", "cuda"]
    smoothing_alpha: float
    ffmpeg_path: str
    work_dir: Path
    cookies_path: Path | None = None
    loader_to_enabled: bool = False

    @classmethod
    def load(cls, env_path: str | Path | None = None) -> AppConfig:
        """Load and validate settings from environment variables or a specific .env file.

        Args:
            env_path: Optional path to an environment file to load.

        Returns:
            AppConfig instance with validated attributes.

        Raises:
            ConfigurationError: If any configuration value is missing or invalid.
        """
        if env_path:
            load_dotenv(dotenv_path=env_path, override=True)

        ffmpeg_path = check_ffmpeg()

        raw_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if raw_provider not in ("anthropic", "gemini", "ollama"):
            raise ConfigurationError(
                f"Invalid LLM_PROVIDER '{raw_provider}'. Must be one of: anthropic, gemini, ollama."
            )

        llm_model = os.getenv("LLM_MODEL", "").strip()
        if not llm_model:
            # Provide sensible defaults if not explicitly set
            if raw_provider == "anthropic":
                llm_model = "claude-3-5-sonnet-latest"
            elif raw_provider == "gemini":
                llm_model = "gemini-1.5-flash"
            else:
                llm_model = "llama3"

        llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        if raw_provider in ("anthropic", "gemini") and not llm_api_key:
            # We don't fail immediately on import to allow --help or checks,
            # but we require it when loading config for pipeline runs.
            pass

        whisper_model = os.getenv("WHISPER_MODEL", "small").strip()

        raw_device = os.getenv("DEVICE", "auto").strip().lower()
        if raw_device not in ("auto", "cpu", "cuda"):
            raise ConfigurationError(
                f"Invalid DEVICE '{raw_device}'. Must be one of: auto, cpu, cuda."
            )

        raw_alpha = os.getenv("SMOOTHING_ALPHA", "0.15").strip()
        try:
            smoothing_alpha = float(raw_alpha)
            if not (0.0 < smoothing_alpha <= 1.0):
                raise ValueError()
        except ValueError:
            raise ConfigurationError(
                f"Invalid SMOOTHING_ALPHA '{raw_alpha}'. Must be a float between 0.0 (exclusive) and 1.0 (inclusive)."
            )

        work_dir = Path(os.getenv("CLIPFORGE_WORK_DIR", "work")).resolve()

        cookies_env = os.getenv("YOUTUBE_COOKIES_PATH", "").strip()
        cookies_path: Path | None = None
        if cookies_env:
            candidate = Path(cookies_env).resolve()
            if candidate.is_file():
                cookies_path = candidate
        elif Path("cookies.txt").is_file():
            cookies_path = Path("cookies.txt").resolve()

        raw_loader_to = os.getenv("LOADER_TO_ENABLED", "false").strip().lower()
        loader_to_enabled = raw_loader_to in ("1", "true", "yes", "on")

        return cls(
            llm_provider=raw_provider,  # type: ignore[arg-type]
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            whisper_model=whisper_model,
            device=raw_device,  # type: ignore[arg-type]
            smoothing_alpha=smoothing_alpha,
            ffmpeg_path=ffmpeg_path,
            work_dir=work_dir,
            cookies_path=cookies_path,
            loader_to_enabled=loader_to_enabled,
        )


# Global singleton instance loaded once
_CONFIG: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the cached configuration or load a new one."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AppConfig.load()
    return _CONFIG
