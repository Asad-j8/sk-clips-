"""Audio transcription stage for ClipForge.

Uses standard library (urllib.request) for cloud transcription via Gemini API / Groq API.
Zero heavy local torch or C++ dependencies. Lightweight (~80MB bundle).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Individual word timing in seconds."""

    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech with start/end timestamps."""

    id: int
    start: float
    end: float
    text: str
    words: list[WordTiming]


def extract_audio_for_transcription(video_path: str | Path, output_audio_path: str | Path) -> str:
    """Extract a lightweight mono 16kHz MP3 or WAV for fast transcription."""
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(output_audio_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output_audio_path)


def transcribe_cloud(
    audio_path: str | Path,
    api_key: str | None = None,
    provider: str = "gemini",
) -> list[TranscriptSegment]:
    """Transcribe audio using cloud API (Gemini or Groq Whisper).

    Zero local GPU/PyTorch footprint. Uses standard library urllib.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    if not api_key:
        logger.warning("No API key set for cloud transcription. Returning empty segments.")
        return []

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if provider == "groq" or os.getenv("GROQ_API_KEY"):
        return _transcribe_groq_whisper(audio_path, api_key)

    return _transcribe_gemini(audio_path, api_key)


def _transcribe_groq_whisper(audio_path: Path, api_key: str) -> list[TranscriptSegment]:
    """Call Groq Whisper API using standard library multipart POST."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

    with open(audio_path, "rb") as f:
        file_bytes = f.read()

    body = bytearray()
    # file part
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'.encode())
    body.extend(b"Content-Type: audio/mpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")

    # model part
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body.extend(b"whisper-large-v3-turbo\r\n")

    # response_format part
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
    body.extend(b"verbose_json\r\n")

    # closing boundary
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    segments: list[TranscriptSegment] = []
    raw_segments = result.get("segments", [])
    for idx, seg in enumerate(raw_segments):
        words = [
            WordTiming(word=w.get("word", ""), start=float(w.get("start", 0)), end=float(w.get("end", 0)))
            for w in seg.get("words", [])
        ]
        segments.append(
            TranscriptSegment(
                id=idx,
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                text=seg.get("text", "").strip(),
                words=words,
            )
        )
    return segments


def _transcribe_gemini(audio_path: Path, api_key: str) -> list[TranscriptSegment]:
    """Cloud transcribe via Gemini generateContent."""
    import base64

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/mp3",
                            "data": audio_b64,
                        }
                    },
                    {
                        "text": (
                            "Transcribe this audio with timestamps. Return ONLY a valid JSON array of objects:\n"
                            '[{"start": 0.0, "end": 2.5, "text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.5}]}]'
                        )
                    },
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    candidates = result.get("candidates", [])
    if not candidates:
        return []

    text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "[]")
    try:
        data = json.loads(text_content)
    except Exception:
        return []

    segments: list[TranscriptSegment] = []
    for idx, item in enumerate(data):
        words = [
            WordTiming(
                word=w.get("word", ""),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
            )
            for w in item.get("words", [])
        ]
        segments.append(
            TranscriptSegment(
                id=idx,
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=item.get("text", "").strip(),
                words=words,
            )
        )
    return segments
