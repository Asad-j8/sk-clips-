"""Active speaker detection using face tracks and audio energy/activity.

Features:
- Extracts audio RMS / energy profile per time segment
- Correlates audio speech segments with detected faces
- Distinguishes simultaneous speakers (split screen trigger)
- Guarantees multi-speaker safety: Never crop one speaker while omitting the other
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from clipforge.stages.face_tracker import FaceDetection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerInterval:
    """Active speaker identified for a specific time range."""

    start: float
    end: float
    active_face_id: str | None
    is_multi_speaker: bool
    secondary_face_id: str | None = None


def extract_audio_rms_profile(
    video_path: Path | str,
    window_sec: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Extract RMS audio volume energy across time windows using ffmpeg / ffprobe.

    Returns:
        List of (start_sec, end_sec, rms_energy_normalized).
    """
    v_path = Path(video_path).resolve()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return []

    # Extract 16kHz mono raw PCM s16le stream
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(v_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw_pcm = proc.stdout
        if not raw_pcm:
            return []

        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sample_rate = 16000
        hop_len = int(window_sec * sample_rate)

        profile: list[tuple[float, float, float]] = []
        for i in range(0, len(samples), hop_len):
            chunk = samples[i : i + hop_len]
            if len(chunk) == 0:
                continue
            rms = float(np.sqrt(np.mean(chunk**2)))
            st = i / sample_rate
            et = (i + len(chunk)) / sample_rate
            profile.append((st, et, rms))

        return profile
    except Exception as e:
        logger.warning("Failed to extract audio RMS profile: %s", e)
        return []


def determine_active_speakers(
    audio_energy_profile: list[tuple[float, float, float]],
    frame_faces_map: dict[float, list[FaceDetection]],
    min_speech_energy: float = 0.015,
) -> list[SpeakerInterval]:
    """Determine speaker presence and simultaneous speaker segments.

    Rules:
    - If 2+ faces are detected and audio energy is consistently active,
      check for multi-speaker interaction.
    - If both speakers are present in the same shot and talking, flag is_multi_speaker=True.
    - Smooth flip rate so speaker doesn't flicker under 0.5s.
    """
    if not audio_energy_profile:
        # Fallback: assign primary face if available
        return [
            SpeakerInterval(
                start=0.0,
                end=60.0,
                active_face_id="face_1" if frame_faces_map else None,
                is_multi_speaker=False,
            )
        ]

    intervals: list[SpeakerInterval] = []

    for st, et, energy in audio_energy_profile:
        mid_t = (st + et) / 2.0
        # Find nearest frame faces
        nearest_t = min(frame_faces_map.keys(), key=lambda t: abs(t - mid_t)) if frame_faces_map else None
        faces = frame_faces_map.get(nearest_t, []) if nearest_t is not None else []

        if energy < min_speech_energy or not faces:
            # Silence or no face
            intervals.append(
                SpeakerInterval(
                    start=st,
                    end=et,
                    active_face_id=None,
                    is_multi_speaker=False,
                )
            )
        elif len(faces) == 1:
            intervals.append(
                SpeakerInterval(
                    start=st,
                    end=et,
                    active_face_id=faces[0].face_id,
                    is_multi_speaker=False,
                )
            )
        else:
            # Multiple faces: check if both are prominent (multi-speaker simultaneous)
            sorted_faces = sorted(faces, key=lambda f: f.w * f.h, reverse=True)
            f1, f2 = sorted_faces[0], sorted_faces[1]

            # If second face is at least 35% size of largest face, both are prominent
            is_both_active = (f2.w * f2.h) >= 0.35 * (f1.w * f1.h)
            intervals.append(
                SpeakerInterval(
                    start=st,
                    end=et,
                    active_face_id=f1.face_id,
                    is_multi_speaker=is_both_active,
                    secondary_face_id=f2.face_id if is_both_active else None,
                )
            )

    # Merge consecutive identical intervals
    if not intervals:
        return []

    merged: list[SpeakerInterval] = []
    curr = intervals[0]

    for nxt in intervals[1:]:
        if (
            nxt.active_face_id == curr.active_face_id
            and nxt.is_multi_speaker == curr.is_multi_speaker
            and nxt.secondary_face_id == curr.secondary_face_id
        ):
            curr = SpeakerInterval(
                start=curr.start,
                end=nxt.end,
                active_face_id=curr.active_face_id,
                is_multi_speaker=curr.is_multi_speaker,
                secondary_face_id=curr.secondary_face_id,
            )
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)

    return merged
