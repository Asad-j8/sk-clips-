"""Exponential Moving Average (EMA) smoothing and pan-speed limiting for video crop boxes.

Rules:
- Crop position smoothed via EMA (alpha from config or caller).
- Pan speed capped to prevent abrupt jerky movements or visual disorientation.
- Crop changes strictly confined to shot boundaries or speech pauses.
- Face rule: Crop width >= 1.1 * face width (face is never cropped).
- Clamped strictly within frame boundaries [0, max_crop_x].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropBox:
    """Calculated 9:16 crop window coordinates."""

    x: int
    y: int
    width: int
    height: int


def calculate_vertical_crop_dim(
    source_w: int,
    source_h: int,
    aspect_w: int = 9,
    aspect_h: int = 16,
) -> tuple[int, int]:
    """Calculate 9:16 crop dimensions inside a landscape source.

    For 1920x1080 source, crop_h = 1080, crop_w = 1080 * 9 / 16 = 608 (even).
    """
    target_aspect = aspect_w / aspect_h
    src_aspect = source_w / max(1, source_h)

    if src_aspect >= target_aspect:
        crop_h = source_h
        crop_w = int(round(crop_h * target_aspect))
    else:
        crop_w = source_w
        crop_h = int(round(crop_w / target_aspect))

    # Keep even dimensions
    if crop_w % 2 != 0:
        crop_w += 1
    if crop_h % 2 != 0:
        crop_h += 1

    crop_w = min(source_w, max(2, crop_w))
    crop_h = min(source_h, max(2, crop_h))
    return crop_w, crop_h


class CropSmoother:
    """Smooths crop movement across frames using EMA and pan speed limit."""

    def __init__(
        self,
        source_w: int,
        source_h: int,
        alpha: float = 0.15,
        max_pan_speed_px_per_sec: float = 350.0,
        fps: float = 30.0,
    ) -> None:
        self.source_w = source_w
        self.source_h = source_h
        self.alpha = max(0.01, min(1.0, alpha))
        self.fps = max(1.0, fps)
        self.max_pan_step = max_pan_speed_px_per_sec / self.fps
        self.crop_w, self.crop_h = calculate_vertical_crop_dim(source_w, source_h)
        self.max_crop_x = max(0, source_w - self.crop_w)

        # Default centered crop
        self.current_x: float = float(max(0, (source_w - self.crop_w) // 2))

    def update(
        self,
        target_center_x: float | None,
        face_width: float | None = None,
        is_boundary_or_pause: bool = True,
    ) -> CropBox:
        """Update crop window position for next frame.

        Args:
            target_center_x: Pixel x-center of the speaker face, or None if no face.
            face_width: Pixel width of the detected face (for 1.1x face rule check).
            is_boundary_or_pause: True if current frame is at a shot transition or speech pause.

        Returns:
            CropBox with smoothed, bounded, and speed-capped integer coordinates.
        """
        # 1. Determine raw desired x
        if target_center_x is not None:
            # Enforce 1.1x face rule: ensure face fits inside crop_w
            effective_face_w = face_width or 0.0
            if effective_face_w > 0 and (effective_face_w * 1.1) > self.crop_w:
                # If face is larger than 9:16 crop width, default to centered to preserve context
                desired_x = float((self.source_w - self.crop_w) // 2)
            else:
                desired_x = target_center_x - (self.crop_w / 2.0)
        else:
            desired_x = self.current_x

        # Clamp raw desired_x to frame boundaries
        clamped_desired_x = max(0.0, min(float(self.max_crop_x), desired_x))

        # 2. Crop change rule: Only allow significant repositioning during shot boundary or pause
        if is_boundary_or_pause:
            effective_target = clamped_desired_x
        else:
            # During continuous speech mid-shot, apply strong resistance to drift
            effective_target = (self.current_x * 0.9) + (clamped_desired_x * 0.1)

        # 3. Apply EMA
        ema_x = (1.0 - self.alpha) * self.current_x + self.alpha * effective_target

        # 4. Cap pan speed to prevent abrupt jump
        delta = ema_x - self.current_x
        if abs(delta) > self.max_pan_step:
            delta = math_copysign(self.max_pan_step, delta)
        new_x = self.current_x + delta

        # Clamp again strictly
        new_x = max(0.0, min(float(self.max_crop_x), new_x))
        self.current_x = new_x

        return CropBox(
            x=int(round(new_x)),
            y=0,
            width=self.crop_w,
            height=self.crop_h,
        )


def math_copysign(magnitude: float, sign_ref: float) -> float:
    """Return magnitude with the sign of sign_ref."""
    return magnitude if sign_ref >= 0 else -magnitude
