import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Asegurar que los módulos del proyecto se carguen
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient


class FakeResult:
    def __init__(self, text, language):
        self.text = text
        self.language = language
        self.time_stamps = None


class FakeDiarizationSegment:
    def __init__(self, speaker, start, end):
        self.speaker = speaker
        self.start = start
        self.end = end


class TestASRServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock Qwen3ASRModel antes de importar main
        cls.model_patcher = patch("asr_model.Qwen3ASRModel")
        cls.mock_model_cls = cls.model_patcher.start()

        mock_instance = MagicMock()
        cls.mock_model_cls.from_pretrained.return_value = mock_instance
        mock_instance.transcribe.return_value = [
            FakeResult(text="Hola mundo de prueba", language="Spanish")
        ]

        # Mock diarization pipeline
        cls.diarization_patcher = patch("diarization.Pipeline")
        cls.mock_diarization_cls = cls.diarization_patcher.start()
        mock_diarization_instance = MagicMock()
        cls.mock_diarization_cls.from_pretrained.return_value = mock_diarization_instance

        # Simular salida de diarización
        mock_output = MagicMock()
        mock_turn_0 = MagicMock()
        mock_turn_0.start = 0.0
        mock_turn_0.end = 2.0
        mock_turn_1 = MagicMock()
        mock_turn_1.start = 2.5
        mock_turn_1.end = 5.0
        mock_output.speaker_diarization = [
            (mock_turn_0, "SPEAKER_00"),
            (mock_turn_1, "SPEAKER_01"),
        ]
        mock_output.exclusive_speaker_diarization = mock_output.speaker_diarization
        mock_diarization_instance.return_value = mock_output

        # Importar main después de los mocks
        import main
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.model_patcher.stop()
        cls.diarization_patcher.stop()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["device"], "cpu")
        self.assertEqual(data["model_name"], "Qwen/Qwen3-ASR-0.6B")
        self.assertIn("diarization_loaded", data)

    def test_list_models(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "Qwen3-ASR-0.6B")

    def test_transcribe_short_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=1000:duration=2",
                    "-ar", "16000", "-ac", "1", tmp.name,
                ],
                capture_output=True, check=True,
            )
            audio_path = tmp.name

        with open(audio_path, "rb") as f:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "es", "prompt": "contexto de prueba"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertIn("language", data)
        self.assertIn("duration", data)
        self.assertIn("processing_time", data)
        self.assertIsInstance(data["duration"], float)
        self.assertIsInstance(data["processing_time"], float)
        Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_with_diarization(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=1000:duration=5",
                    "-ar", "16000", "-ac", "1", tmp.name,
                ],
                capture_output=True, check=True,
            )
            audio_path = tmp.name

        with open(audio_path, "rb") as f:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "es", "diarize": "true"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertIn("segments", data)
        self.assertIsInstance(data["segments"], list)
        self.assertTrue(len(data["segments"]) > 0)
        # Verificar formato de segmentos
        seg = data["segments"][0]
        self.assertIn("speaker", seg)
        self.assertIn("start", seg)
        self.assertIn("end", seg)
        self.assertIn("text", seg)
        Path(audio_path).unlink(missing_ok=True)

    def test_diarization_endpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=1000:duration=5",
                    "-ar", "16000", "-ac", "1", tmp.name,
                ],
                capture_output=True, check=True,
            )
            audio_path = tmp.name

        with open(audio_path, "rb") as f:
            response = self.client.post(
                "/v1/audio/diarization",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"min_speakers": "1", "max_speakers": "3"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("segments", data)
        self.assertIn("num_speakers", data)
        self.assertIsInstance(data["segments"], list)
        self.assertTrue(len(data["segments"]) > 0)
        seg = data["segments"][0]
        self.assertIn("speaker", seg)
        self.assertIn("start", seg)
        self.assertIn("end", seg)
        Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_chunked_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=1000:duration=5",
                    "-ar", "16000", "-ac", "1", tmp.name,
                ],
                capture_output=True, check=True,
            )
            audio_path = tmp.name

        with patch("asr_model.settings.CHUNK_THRESHOLD_MINUTES", 0):
            with patch("asr_model.settings.CHUNK_DURATION", 2):
                with open(audio_path, "rb") as f:
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("long.wav", f, "audio/wav")},
                        data={"language": "auto"},
                    )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_empty_file(self):
        empty = io.BytesIO(b"")
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("empty.wav", empty, "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
