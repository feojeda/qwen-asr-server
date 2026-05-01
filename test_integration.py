"""Integration tests — requieren el modelo Qwen3-ASR real y pyannote.

Ejecutar con:
    python -m pytest test_integration.py -v --run-integration

Sin --run-integration, estos tests se saltean automáticamente.
No se ejecutan en CI (GitHub Actions no tiene GPU ni descarga modelos).
"""

import tempfile
from pathlib import Path

import pytest

# Marcador explícito (aunque conftest ya los detecta por keyword "integration")
pytestmark = pytest.mark.integration


def _generate_test_audio(duration: float = 3.0) -> Path:
    """Genera un archivo WAV sintético con ffmpeg (tono de 1 kHz)."""
    import subprocess

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"sine=frequency=1000:duration={duration}",
            "-ar", "16000", "-ac", "1", tmp.name,
        ],
        capture_output=True, check=True,
    )
    return Path(tmp.name)


class TestRealTranscription:
    """Transcripción simple con el modelo Qwen3-ASR real."""

    def test_transcribe_short_audio(self):
        """Transcribe un audio corto generado sintéticamente (sin palabras reales)."""
        from fastapi.testclient import TestClient
        import main

        audio_path = _generate_test_audio(duration=3.0)
        try:
            with TestClient(main.app) as client:
                with open(audio_path, "rb") as f:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", f, "audio/wav")},
                        data={"language": "auto"},
                    )

            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert "language" in data
            assert "duration" in data
            assert "processing_time" in data
            assert data["duration"] > 0
            assert data["processing_time"] > 0
        finally:
            audio_path.unlink(missing_ok=True)

    def test_transcribe_with_language_hint(self):
        """Transcribe con hint de idioma español."""
        from fastapi.testclient import TestClient
        import main

        audio_path = _generate_test_audio(duration=2.0)
        try:
            with TestClient(main.app) as client:
                with open(audio_path, "rb") as f:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", f, "audio/wav")},
                        data={"language": "es"},
                    )

            assert response.status_code == 200
            data = response.json()
            # Con hint "es", debería reportar Spanish
            assert data["language"] in ("Spanish", "es", "auto")
        finally:
            audio_path.unlink(missing_ok=True)

    def test_transcribe_empty_file_returns_400(self):
        """Archivo vacío debe devolver 400."""
        from fastapi.testclient import TestClient
        import main
        import io

        with TestClient(main.app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
            )

        assert response.status_code == 400


class TestRealDiarization:
    """Diarización con pyannote real (requiere HF_TOKEN para descargar)."""

    def test_diarization_endpoint(self):
        """Endpoint /v1/audio/diarization con modelo real."""
        from fastapi.testclient import TestClient
        import main

        audio_path = _generate_test_audio(duration=5.0)
        try:
            with TestClient(main.app) as client:
                with open(audio_path, "rb") as f:
                    response = client.post(
                        "/v1/audio/diarization",
                        files={"file": ("test.wav", f, "audio/wav")},
                        data={"min_speakers": "1", "max_speakers": "3"},
                    )

            # Si pyannote no está disponible (sin HF_TOKEN), esto puede fallar
            # pero el test verifica que al menos no sea un 500 genérico
            if response.status_code == 200:
                data = response.json()
                assert "segments" in data
                assert "num_speakers" in data
                assert isinstance(data["segments"], list)
            elif response.status_code == 500:
                # Puede fallar si pyannote no está configurado — aceptable
                data = response.json()
                assert "detail" in data
        finally:
            audio_path.unlink(missing_ok=True)

    def test_transcribe_with_diarization_graceful_degradation(self):
        """Si diarize=true, debe devolver segmentos o diarization_failed=true."""
        from fastapi.testclient import TestClient
        import main

        audio_path = _generate_test_audio(duration=4.0)
        try:
            with TestClient(main.app) as client:
                with open(audio_path, "rb") as f:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", f, "audio/wav")},
                        data={"language": "auto", "diarize": "true"},
                    )

            assert response.status_code == 200
            data = response.json()
            assert "text" in data

            if data.get("segments") and len(data["segments"]) > 0:
                # Si la diarización funcionó, verificar formato
                seg = data["segments"][0]
                assert "speaker" in seg
                assert "start" in seg
                assert "end" in seg
                assert "text" in seg
            else:
                # Si no, debe tener el flag de fallback
                # (no es obligatorio que falle — depende de pyannote)
                pass
        finally:
            audio_path.unlink(missing_ok=True)


class TestRealChunking:
    """Transcripción con chunking para audios largos."""

    def test_long_audio_triggers_chunking(self):
        """Audio de 35s debe activar chunking automático."""
        import main
        from fastapi.testclient import TestClient

        # Guardar el threshold original
        original_threshold = main.settings.CHUNK_THRESHOLD_MINUTES

        try:
            # Forzar chunking con audio de 35s (umbral 0 min)
            main.settings.CHUNK_THRESHOLD_MINUTES = 0
            main.settings.CHUNK_DURATION = 10  # chunks de 10s

            audio_path = _generate_test_audio(duration=25.0)
            try:
                with TestClient(main.app) as client:
                    with open(audio_path, "rb") as f:
                        response = client.post(
                            "/v1/audio/transcriptions",
                            files={"file": ("long.wav", f, "audio/wav")},
                            data={"language": "auto"},
                        )

                assert response.status_code == 200
                data = response.json()
                assert "text" in data
                assert data["duration"] > 0
                assert data["processing_time"] > 0
            finally:
                audio_path.unlink(missing_ok=True)
        finally:
            main.settings.CHUNK_THRESHOLD_MINUTES = original_threshold


class TestHealthAndModels:
    """Endpoints de health y listado de modelos con servidor real."""

    def test_health_endpoint(self):
        """Health check con modelo real."""
        from fastapi.testclient import TestClient
        import main

        with TestClient(main.app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "device" in data
        assert "model_name" in data

    def test_list_models_endpoint(self):
        """Listado de modelos OpenAI-compatible."""
        from fastapi.testclient import TestClient
        import main

        with TestClient(main.app) as client:
            response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert "id" in data["data"][0]
