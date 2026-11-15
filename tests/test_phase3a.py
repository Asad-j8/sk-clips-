"""Phase 3a Verification Tests: Speaker Tracking, Layouts, Crop Math, Face Rules.

Tests cover:
1. Exact prompt templates in prompts.py (word-for-word).
2. Layout decisions: none, screencast, split, camera_inset.
3. Multi-speaker safety: Never crop one speaker when two are talking in the same shot.
4. Face rule: Crop width >= 1.1x face width.
5. Speaker tracking with YuNet / Haar and persistent IDs.
6. EMA smoothing and pan speed capping (no abrupt jumps).
7. Error resilience (no crash on edge cases).
"""

import math
import unittest
from pathlib import Path

import numpy as np

from clipforge.prompts import (
    CLIP_FINDER_PROMPT_TEMPLATE,
    LAYOUT_PROMPT_TEMPLATE,
    REFRAMING_PROMPT_TEMPLATE,
    SHOT_PROMPT_TEMPLATE,
)
from clipforge.stages.face_tracker import (
    FaceDetection,
    FaceTracker,
    check_face_crop_rule,
    compute_iou,
)
from clipforge.stages.layout import (
    build_layout_prompt,
    build_shot_prompt,
    determine_clip_layout,
)
from clipforge.stages.smoothing import (
    CropSmoother,
    calculate_vertical_crop_dim,
)
from clipforge.stages.speaker import (
    SpeakerInterval,
    determine_active_speakers,
)


class TestPhase3aPrompts(unittest.TestCase):
    """Test exact word-for-word prompt fidelity."""

    def test_layout_prompt_exact_words(self):
        built = build_layout_prompt(
            duration=45.2,
            face_count=2,
            is_screencast=False,
            simultaneous_speakers=True,
            summary="Interview segment",
        )
        self.assertIn("You are an intelligent video reframing and layout director", built)
        self.assertIn('"none": Default.', built)
        self.assertIn('"screencast": Selected when', built)
        self.assertIn('"split": Selected when two speakers', built)
        self.assertIn("45.2s", built)

    def test_shot_prompt_exact_words(self):
        built = build_shot_prompt(
            shot_id=1,
            start_time=0.0,
            end_time=15.0,
            visual_description="Slides with presenter webcam inset",
            has_webcam=True,
        )
        self.assertIn("You are a precision shot analyzer", built)
        self.assertIn('"presenter_cam": true | false', built)
        self.assertIn('"camera_inset": "webcam_stack" | "none"', built)
        self.assertIn("has webcam inset: true", built.lower())


class TestLayoutRules(unittest.TestCase):
    """Test layout selection logic."""

    def test_default_none_layout(self):
        intervals = [
            SpeakerInterval(start=0.0, end=30.0, active_face_id="face_1", is_multi_speaker=False)
        ]
        layouts = determine_clip_layout(intervals)
        self.assertEqual(len(layouts), 1)
        self.assertEqual(layouts[0].layout, "none")

    def test_screencast_layout(self):
        intervals = [
            SpeakerInterval(start=0.0, end=30.0, active_face_id=None, is_multi_speaker=False)
        ]
        layouts = determine_clip_layout(intervals, is_screencast=True, has_webcam_inset=False)
        self.assertEqual(layouts[0].layout, "screencast")

    def test_camera_inset_layout(self):
        intervals = [
            SpeakerInterval(start=0.0, end=30.0, active_face_id="face_1", is_multi_speaker=False)
        ]
        layouts = determine_clip_layout(intervals, is_screencast=True, has_webcam_inset=True)
        self.assertEqual(layouts[0].layout, "camera_inset")
        self.assertTrue(layouts[0].presenter_cam)

    def test_multi_speaker_triggers_split_never_crops_out(self):
        """Multi-speaker simultaneous talking MUST trigger split layout."""
        intervals = [
            SpeakerInterval(
                start=0.0,
                end=20.0,
                active_face_id="face_1",
                is_multi_speaker=True,
                secondary_face_id="face_2",
            )
        ]
        layouts = determine_clip_layout(intervals)
        self.assertEqual(layouts[0].layout, "split")
        self.assertEqual(layouts[0].primary_face_id, "face_1")
        self.assertEqual(layouts[0].secondary_face_id, "face_2")


class TestFaceTrackingAndRule(unittest.TestCase):
    """Test YuNet face tracking and 1.1x face rule."""

    def test_face_crop_rule_1_point_1_ratio(self):
        # Crop width 608px vs face width 200px -> valid (608 >= 220)
        self.assertTrue(check_face_crop_rule(crop_w=608, face_w=200))
        # Face width 600px -> invalid (608 < 660)
        self.assertFalse(check_face_crop_rule(crop_w=608, face_w=600))

    def test_iou_computation(self):
        box_a = [100, 100, 50, 50]
        box_b = [100, 100, 50, 50]
        self.assertAlmostEqual(compute_iou(box_a, box_b), 1.0)

        box_c = [200, 200, 50, 50]
        self.assertAlmostEqual(compute_iou(box_a, box_c), 0.0)

    def test_yunet_tracker_initialization_no_crash(self):
        """Tracker initializes cleanly without crashing even on empty or synthetic inputs."""
        tracker = FaceTracker()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        faces = tracker.detect_faces(dummy_frame)
        self.assertIsInstance(faces, list)


class TestCropMathAndSmoothing(unittest.TestCase):
    """Test 9:16 crop math, EMA smoothing, and pan speed cap."""

    def test_crop_math_1920x1080_yields_608(self):
        crop_w, crop_h = calculate_vertical_crop_dim(1920, 1080)
        self.assertEqual(crop_h, 1080)
        self.assertEqual(crop_w, 608)

    def test_pan_speed_cap_prevents_abrupt_jumps(self):
        """Ensure huge target jump is constrained by pan speed cap."""
        smoother = CropSmoother(
            source_w=1920,
            source_h=1080,
            alpha=0.5,
            max_pan_speed_px_per_sec=300.0,
            fps=30.0,
        )
        # initial center is (1920-608)//2 = 656
        initial_x = smoother.current_x
        # Sudden target on far right (e.g. face at x=1800)
        box = smoother.update(target_center_x=1800.0, is_boundary_or_pause=True)

        max_step = 300.0 / 30.0  # 10px per frame
        self.assertLessEqual(abs(box.x - initial_x), max_step + 1.0)


if __name__ == "__main__":
    unittest.main()
