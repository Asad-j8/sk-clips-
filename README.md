# ClipForge

Free, local-first Python web app that turns long podcast or interview videos into short vertical (9:16) clips with speaker-aware face-tracked reframing and captions.

## Tech Stack
- **UI**: Gradio
- **Video Download**: yt-dlp
- **Transcription**: faster-whisper with word-level timestamps
- **Face Detection & Tracking**: MediaPipe Face Detection & Mesh (fallback OpenCV YuNet) + IoU tracker
- **Active Speaker Detection**: Silero VAD + Lip Activity analysis
- **Reframing & Layout**: Deterministic rules + LLM fallback for ambiguous segments
- **Rendering**: OpenCV crop pipeline piped into ffmpeg with ASS caption burn-in

## Setup Instructions

### 1. Requirements & Prerequisites
- Python 3.11+
- `ffmpeg` and `ffprobe` installed and available on PATH.

### 2. Installation
```bash
# Clone the repository
git clone <repo_url>
cd sk-clips-

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Open `.env` and configure:
- `LLM_PROVIDER`: `gemini`, `anthropic`, or `ollama` (for free local inference)
- `LLM_MODEL`: e.g. `gemini-1.5-flash`, `claude-3-5-sonnet-latest`, or `llama3`
- `LLM_API_KEY`: API key for Gemini or Anthropic (leave empty if using Ollama)
- `WHISPER_MODEL`: `small` (or `tiny`, `base`, `medium`, `large-v3`)
- `DEVICE`: `auto`, `cpu`, or `cuda`
- `SMOOTHING_ALPHA`: `0.15`

### 4. Running the Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
