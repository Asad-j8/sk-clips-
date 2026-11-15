"""Tests for lightweight cloud transcriber."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from clipforge.stages.transcriber import TranscriptSegment, WordTiming, transcribe_cloud


class TestTranscriber(unittest.TestCase):
    """Test cloud transcription with mock HTTP responses."""

    def test_transcribe_empty_key(self):
        """Transcribe should safely return empty list if no API key is set."""
        with patch.dict("os.environ", {}, clear=True):
            res = transcribe_cloud("dummy.mp3", api_key="")
            self.assertEqual(res, [])

    @patch("clipforge.stages.transcriber.urllib.request.urlopen")
    def test_transcribe_groq_whisper(self, mock_urlopen):
        """Test Groq Whisper transcription parsing."""
        payload = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0},
                        {"word": "world", "start": 1.0, "end": 2.0},
                    ],
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", unittest.mock.mock_open(read_data=b"fake audio")):
            segments = transcribe_cloud("dummy.mp3", api_key="test_groq_key", provider="groq")
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].text, "Hello world")
            self.assertEqual(len(segments[0].words), 2)
            self.assertEqual(segments[0].words[0].word, "Hello")


if __name__ == "__main__":
    unittest.main()
