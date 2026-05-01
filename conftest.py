"""Pytest conftest: fixtures compartidos y manejo de integration tests.

Patrón:
  - pytest (sin flags) → solo tests unitarios (mocks, sin GPU)
  - pytest --run-integration → tests de integración con modelos reales

Los mocks a nivel de módulo (sys.modules) se aplican ANTES de importar
los módulos de la aplicación, permitiendo que los tests unitarios corran
sin qwen-asr ni pyannote instalados.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Asegurar que el directorio raíz del proyecto está en sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── CLI flag para integration tests ──────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that load real AI models (slow, requires model downloads)",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(
        reason="Integration test: run with --run-integration (requires model downloads)"
    )
    for item in items:
        if "integration" in item.keywords:
            if not config.getoption("--run-integration"):
                item.add_marker(skip_integration)


# ── Mock global de dependencias pesadas (solo modo unitario) ────────

_integration_mode = False
for arg in sys.argv:
    if arg == "--run-integration":
        _integration_mode = True
        break

if not _integration_mode:
    # Mock qwen_asr antes de que asr_model.py intente importarlo
    _mock_qwen_cls = MagicMock()
    _mock_qwen_instance = MagicMock()

    class FakeASRResult:
        def __init__(self, text="Hola mundo de prueba", language="Spanish"):
            self.text = text
            self.language = language
            self.time_stamps = None

    _mock_qwen_instance.transcribe.return_value = [FakeASRResult()]
    _mock_qwen_cls.from_pretrained.return_value = _mock_qwen_instance
    sys.modules["qwen_asr"] = MagicMock(Qwen3ASRModel=_mock_qwen_cls)

    # Mock pyannote.audio antes de que diarization.py intente importarlo
    _mock_pipeline_cls = MagicMock()
    _mock_pipeline_instance = MagicMock()

    class FakeTurn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    _mock_output = MagicMock()
    _mock_output.speaker_diarization = [
        (FakeTurn(0.0, 2.0), "SPEAKER_00"),
        (FakeTurn(2.5, 5.0), "SPEAKER_01"),
    ]
    _mock_output.exclusive_speaker_diarization = _mock_output.speaker_diarization
    _mock_pipeline_instance.return_value = _mock_output
    _mock_pipeline_cls.from_pretrained.return_value = _mock_pipeline_instance

    # pyannote.audio importa internamente Pipeline; lo mockeamos completo
    _mock_pyannote = MagicMock()
    _mock_pyannote.Pipeline = _mock_pipeline_cls
    sys.modules["pyannote.audio"] = _mock_pyannote
