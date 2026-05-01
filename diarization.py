import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from pyannote.audio import Pipeline

from audio_utils import convert_to_wav
from config import settings

logger = logging.getLogger(__name__)

# HF_TOKEN puede ser None si el modelo ya está en caché o si se usa local
HF_TOKEN = settings.__dict__.get("HF_TOKEN", None) or None


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    text: str = ""


class DiarizationManager:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarization")
        self.device = settings.DEVICE
        # pyannote funciona mejor en CPU para modelos pequeños, pero podemos usar MPS en Mac si está disponible
        if self.device == "cpu" and torch.backends.mps.is_available():
            logger.info("MPS disponible en Mac, pero pyannote community-1 se mantiene en CPU por estabilidad")

    def _load_pipeline_sync(self) -> Pipeline:
        logger.info("Cargando pipeline de diarización pyannote...")
        start = time.time()

        # Intentar cargar con token si está disponible
        kwargs = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                **kwargs,
            )
        except Exception as e:
            logger.warning("Error cargando community-1: %s", e)
            # Fallback al pipeline legacy 3.1
            logger.info("Intentando pipeline legacy 3.1...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                **kwargs,
            )

        if pipeline is None:
            raise RuntimeError("No se pudo cargar ningún pipeline de diarización.")

        pipeline.to(torch.device(self.device))
        elapsed = time.time() - start
        logger.info("Pipeline de diarización cargado en %.2f segundos", elapsed)
        return pipeline

    async def load(self) -> None:
        if self.pipeline is not None:
            return
        async with self._lock:
            if self.pipeline is not None:
                return
            loop = asyncio.get_running_loop()
            self.pipeline = await loop.run_in_executor(None, self._load_pipeline_sync)

    def unload(self) -> None:
        if self.pipeline is not None:
            logger.info("Descargando pipeline de diarización...")
            del self.pipeline
            self.pipeline = None

    async def diarize(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Ejecuta diarización sobre un archivo de audio y retorna segmentos con hablante."""
        await self.load()

        # pyannote necesita WAV 16kHz mono
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            await asyncio.get_running_loop().run_in_executor(
                None, convert_to_wav, audio_path, wav_path
            )

            loop = asyncio.get_running_loop()
            segments = await loop.run_in_executor(
                self._executor,
                self._diarize_sync,
                str(wav_path),
                num_speakers,
                min_speakers,
                max_speakers,
            )

        return segments

    def _diarize_sync(
        self,
        audio_path: str,
        num_speakers: Optional[int],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> List[SpeakerSegment]:
        start = time.time()

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        output = self.pipeline(audio_path, **kwargs)

        segments: List[SpeakerSegment] = []
        # Usar exclusive_speaker_diarization si está disponible (community-1)
        diarization = getattr(output, "exclusive_speaker_diarization", output.speaker_diarization)

        for turn, speaker in diarization:
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=round(turn.start, 2),
                end=round(turn.end, 2),
            ))

        elapsed = time.time() - start
        logger.info("Diarización completada en %.2f s: %d segmentos, %d hablantes",
                    elapsed, len(segments), len({s.speaker for s in segments}))
        return segments

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None


# Singleton global
diarization_manager = DiarizationManager()
