"""Exact word-for-word prompt templates for ClipForge.

Contains Section 7.4 (LAYOUT prompt) and Section 7.5 (SHOT prompt)
along with Section 6.9 clip finder prompt. Only runtime placeholders are formatted.
"""

from __future__ import annotations

# ==============================================================================
# SECTION 7.4: LAYOUT PROMPT (Word-for-word)
# ==============================================================================
LAYOUT_PROMPT_TEMPLATE = """You are an intelligent video reframing and layout director for short-form 9:16 vertical video production.

Analyze the video metadata, visual features, and clip context below. Determine the single best layout choice for this video clip.

Layout Choices:
- "none": Default. Suitable for standard single-person talking head, interviews with natural camera panning/switching, b-roll, action, or monologue clips.
- "screencast": Selected when the video primarily displays slides, software tutorials, code editors, or screen recordings.
- "split": Selected when two speakers are in the same shot speaking simultaneously or having a rapid dialogue exchange where cutting back and forth would harm conversational flow.

Clip Information:
- Duration: {duration}s
- Face Count Detected: {face_count}
- Screencast Content Detected: {is_screencast}
- Simultaneous Speakers Detected: {simultaneous_speakers}
- Summary: {summary}

Return ONLY valid JSON matching this exact structure:
{{
  "layout": "none" | "screencast" | "split",
  "reason": "Brief explanation of the layout choice"
}}
"""


# ==============================================================================
# SECTION 7.5: SHOT PROMPT (Word-for-word)
# ==============================================================================
SHOT_PROMPT_TEMPLATE = """You are a precision shot analyzer for short-form video reframing.

For each shot or time interval within the clip, inspect the speaker activity and visual contents.
Determine whether a webcam or presenter inset camera is present over a screen recording.

Shot Information:
- Start Time: {start_time}s
- End Time: {end_time}s
- Visual Description: {visual_description}
- Has Webcam Inset: {has_webcam}

Return ONLY valid JSON matching this exact structure:
{{
  "shot_id": {shot_id},
  "presenter_cam": true | false,
  "camera_inset": "webcam_stack" | "none",
  "reason": "Brief shot reasoning"
}}
"""


# ==============================================================================
# SECTION 6.9: CLIP FINDER PROMPT (Word-for-word from spec)
# ==============================================================================
CLIP_FINDER_PROMPT_TEMPLATE = """You are an expert short-form video editor and social media strategist. Analyze the transcript below from a long-form video. Identify the top 3 to 5 most engaging, stand-alone clip ideas suitable for TikTok, Instagram Reels and YouTube Shorts, each 15 to 60 seconds long.
Rules:
- Each clip must make sense without the rest of the video.
- The quote must be copied exactly from the transcript.
- Start and end times must come from the timestamps shown.
- Clips must not overlap.
Virality score (1 to 100) rubric: hook strength in first 3 seconds (0-30), emotional or surprising payoff (0-25), single clear idea (0-25), shareability or debate potential (0-20).
Return ONLY a JSON array, no preamble, no markdown fences. Each object has keys: title, start_time, end_time, quote, virality_score, reason.
Transcript:
{transcript}"""


# ==============================================================================
# SMART REFRAMING CONTROLLER PROMPT (Word-for-word from spec)
# ==============================================================================
REFRAMING_PROMPT_TEMPLATE = """You are a Smart Video Reframing Controller for a short-form video engine. Input: target aspect ratio 9:16 (1080x1920), face tracking metadata with faces, normalized bounding boxes and active speaker per time segment. Choose one layout per segment from: SINGLE_SPEAKER, SPLIT_SCREEN, GRID_2X2, PAN_TO_SPEAKER. Return ONLY JSON: {{"segments":[{{"start_sec":float,"end_sec":float,"layout_mode":string,"primary_face_id":string or null,"secondary_face_id":string or null}}]}}
Metadata:
{metadata_json}"""
