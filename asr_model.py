import asyncio
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import torch
from qwen_asr import Qwen3ASRModel

from audio_utils import convert_to_wav, extract_segment, get_audio_duration, split_audio_into_chunks
from config import settings

logger = logging.getLogger(__name__)

# Mapeo simple de códigos ISO a nombres de idioma que acepta Qwen3-ASR
LANGUAGE_MAP = {
    "auto": None,
    "es": "Spanish",
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}


def _resolve_language(language: Optional[str]) -> Optional[str]:
    if language is None or language.lower() == "auto":
        return None
    mapped = LANGUAGE_MAP.get(language.lower())
    if mapped:
        return mapped
    # Si no está mapeado, pasar tal cual (el modelo puede aceptarlo)
    return language


def _get_optimal_dtype():
    """Selecciona el mejor dtype disponible para CPU."""
    try:
        # bfloat16 es el recomendado por Qwen y reduce a la mitad el uso de RAM
        torch.tensor([1.0], dtype=torch.bfloat16)
        logger.info("Usando torch.bfloat16 para inferencia en CPU")
        return torch.bfloat16
    except Exception:
        logger.info("Usando torch.float32 para inferencia en CPU")
        return torch.float32


class ASRModelManager:
    def __init__(self):
        self.model: Optional[Qwen3ASRModel] = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-inference")
        self.device = settings.DEVICE
        self.dtype = _get_optimal_dtype()
        # Optimizar número de threads de torch para CPU
        if self.device == "cpu":
            num_threads = torch.get_num_threads()
            logger.info("Torch threads disponibles: %d", num_threads)
            # Dejar que torch use todos los cores disponibles por defecto

    def _load_model_sync(self) -> Qwen3ASRModel:
        logger.info("Cargando modelo %s en %s con dtype %s ...", settings.MODEL_NAME, self.device, self.dtype)
        start = time.time()
        model = Qwen3ASRModel.from_pretrained(
            settings.MODEL_NAME,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        elapsed = time.time() - start
        logger.info("Modelo cargado en %.2f segundos", elapsed)
        return model

    async def load(self) -> None:
        """Carga el modelo de forma lazy (thread-safe)."""
        if self.model is not None:
            return
        async with self._lock:
            if self.model is not None:
                return
            loop = asyncio.get_running_loop()
            self.model = await loop.run_in_executor(None, self._load_model_sync)

    def unload(self) -> None:
        """Libera memoria del modelo."""
        if self.model is not None:
            logger.info("Descargando modelo de memoria...")
            del self.model
            self.model = None
            if self.device == "cpu":
                import gc
                gc.collect()
            logger.info("Modelo descargado.")

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """Transcribe un archivo de audio. Retorna dict con text, language, duration, processing_time."""
        await self.load()

        total_start = time.time()

        # Convertir a WAV 16kHz mono en un archivo temporal
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            duration = await asyncio.get_running_loop().run_in_executor(
                None, convert_to_wav, audio_path, wav_path
            )

            if duration > settings.MAX_AUDIO_DURATION:
                raise ValueError(
                    f"Audio demasiado largo: {duration:.1f}s. "
                    f"Máximo permitido: {settings.MAX_AUDIO_DURATION}s"
                )

            resolved_lang = _resolve_language(language)
            context = prompt or ""

            # Determinar si hacer chunking
            chunk_threshold_seconds = settings.CHUNK_THRESHOLD_MINUTES * 60
            if duration > chunk_threshold_seconds:
                logger.info(
                    "Audio de %.1f s supera umbral de %d min. Dividiendo en chunks de %d s...",
                    duration, settings.CHUNK_THRESHOLD_MINUTES, settings.CHUNK_DURATION
                )
                chunks = await asyncio.get_running_loop().run_in_executor(
                    None, split_audio_into_chunks, wav_path, settings.CHUNK_DURATION, Path(tmpdir)
                )
                texts: List[str] = []
                langs: List[str] = []

                for idx, chunk_path in enumerate(chunks):
                    logger.info("Transcribiendo chunk %d/%d: %s", idx + 1, len(chunks), chunk_path)
                    chunk_result = await self._transcribe_single(chunk_path, resolved_lang, context)
                    texts.append(chunk_result["text"])
                    langs.append(chunk_result["language"])

                full_text = " ".join(texts)
                # Idioma mayoritario
                most_common_lang = Counter(langs).most_common(1)[0][0]
            else:
                result = await self._transcribe_single(wav_path, resolved_lang, context)
                full_text = result["text"]
                most_common_lang = result["language"]

        processing_time = time.time() - total_start
        logger.info(
            "Transcripción completada en %.2f s (audio: %.1f s)",
            processing_time, duration
        )

        return {
            "text": full_text.strip(),
            "language": most_common_lang,
            "duration": round(duration, 2),
            "processing_time": round(processing_time, 2),
        }

    async def _transcribe_single(
        self,
        audio_path: Path,
        language: Optional[str],
        context: str,
    ) -> dict:
        """Transcribe un único archivo de audio (sin chunking)."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._infer_sync,
            str(audio_path),
            language,
            context,
        )
        return result

    def _infer_sync(
        self,
        audio_path: str,
        language: Optional[str],
        context: str,
    ) -> dict:
        """Llamada síncrona al modelo (ejecutada en thread pool)."""
        start = time.time()
        results = self.model.transcribe(
            audio=audio_path,
            language=language,
            context=context or "",
            return_time_stamps=False,
        )
        infer_time = time.time() - start
        r = results[0]
        logger.debug("Inferencia chunk en %.2f s -> lang=%s text=%s...", infer_time, r.language, r.text[:60])
        return {
            "text": r.text,
            "language": r.language,
        }

    async def transcribe_segment(
        self,
        audio_path: Path,
        start: float,
        end: float,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """Transcribe un segmento de audio [start, end] en segundos."""
        await self.load()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            segment_path = Path(tmp.name)

        try:
            await asyncio.get_running_loop().run_in_executor(
                None, extract_segment, audio_path, segment_path, start, end
            )
            result = await self._transcribe_single(
                segment_path,
                _resolve_language(language),
                prompt or "",
            )
            return {
                "text": result["text"],
                "language": result["language"],
            }
        finally:
            try:
                segment_path.unlink(missing_ok=True)
            except Exception:
                pass

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Singleton global
model_manager = ASRModelManager()
