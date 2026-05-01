import asyncio
import logging
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asr_model import model_manager
from audio_utils import convert_to_wav as _convert_to_wav
from config import settings
from diarization import diarization_manager
from schemas import (
    DiarizationResponse,
    DiarizationSegment,
    HealthResponse,
    ModelInfo,
    ModelList,
    TranscriptionResponse,
    TranscriptionSegment,
)

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("qwen-asr-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Qwen3-ASR Server")
    logger.info("Config: model=%s device=%s port=%d", settings.MODEL_NAME, settings.DEVICE, settings.PORT)
    yield
    logger.info("Shutting down Qwen3-ASR Server")
    model_manager.unload()
    diarization_manager.unload()


app = FastAPI(
    title="Qwen3-ASR Server",
    description="Servidor ASR basado en Qwen3-ASR para ttsQwen",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        diarization_loaded=diarization_manager.is_loaded,
        device=settings.DEVICE,
        model_name=settings.MODEL_NAME,
    )


@app.get("/v1/models")
async def list_models():
    model_id = settings.MODEL_NAME.split("/")[-1]
    return ModelList(
        object="list",
        data=[
            ModelInfo(
                id=model_id,
                object="model",
                owned_by="Qwen",
            )
        ],
    )


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(..., description="Archivo de audio (mp3, wav, webm, ogg, m4a)"),
    language: str = Form(default="auto", description="Idioma: auto, es, en, ..."),
    prompt: str = Form(default="", description="Contexto/prompt opcional para el modelo"),
    diarize: bool = Form(default=False, description="Activar diarización de hablantes (requiere HF_TOKEN)"),
    num_speakers: int = Form(default=None, description="Número exacto de hablantes (opcional, para diarización)"),
    min_speakers: int = Form(default=None, description="Mínimo de hablantes (opcional, para diarización)"),
    max_speakers: int = Form(default=None, description="Máximo de hablantes (opcional, para diarización)"),
):
    logger.info(
        "Request de transcripción: filename=%s language=%s prompt=%s diarize=%s",
        file.filename, language, prompt, diarize
    )

    suffix = Path(file.filename or "").suffix.lower()
    tmp_path = None
    try:
        # Guardar archivo subido a disco temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".audio") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Archivo de audio vacío")
            tmp.write(content)
            tmp_path = Path(tmp.name)

        if not diarize:
            # Transcripción simple sin diarización
            result = await model_manager.transcribe(
                audio_path=tmp_path,
                language=language if language else "auto",
                prompt=prompt,
            )
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=result["duration"],
                processing_time=result["processing_time"],
            )

        # --- Pipeline con diarización ---
        total_start = time.time()
        diarization_failed = False

        # 1. Diarización (con graceful degradation)
        speaker_segments = []
        try:
            speaker_segments = await diarization_manager.diarize(
                audio_path=tmp_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as e:
            logger.warning("Diarización falló, degradando a transcripción simple: %s", e)
            diarization_failed = True

        if diarization_failed or not speaker_segments:
            # Fallback a transcripción simple si diarización falló o no detectó hablantes
            result = await model_manager.transcribe(
                audio_path=tmp_path,
                language=language if language else "auto",
                prompt=prompt,
            )
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=result["duration"],
                processing_time=result["processing_time"],
                segments=[],
                diarization_failed=diarization_failed or None,
            )

        # 2. Convertir audio original a WAV para segmentos eficientes
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _convert_to_wav(tmp_path, wav_path)
            )

            # 3. Transcribir cada segmento de hablante
            segments_output: list[TranscriptionSegment] = []
            for seg in speaker_segments:
                seg_result = await model_manager.transcribe_segment(
                    audio_path=wav_path,
                    start=seg.start,
                    end=seg.end,
                    language=language if language else "auto",
                    prompt=prompt,
                )
                segments_output.append(TranscriptionSegment(
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                    text=seg_result["text"],
                    language=seg_result.get("language"),
                ))

        # 4. Construir texto completo concatenado
        full_text = " ".join(s.text for s in segments_output)
        # Idioma detectado (del primer segmento, o fallback al hint del usuario)
        if segments_output and segments_output[0].language:
            detected_lang = segments_output[0].language
        else:
            detected_lang = language

        total_time = time.time() - total_start

        return TranscriptionResponse(
            text=full_text.strip(),
            language=detected_lang,
            duration=round(segments_output[-1].end if segments_output else 0, 2),
            processing_time=round(total_time, 2),
            segments=segments_output,
        )

    except ValueError as e:
        logger.warning("Error de validación: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error interno durante transcripción")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


@app.post("/v1/audio/diarization")
async def diarize_audio(
    file: UploadFile = File(..., description="Archivo de audio (mp3, wav, webm, ogg, m4a)"),
    num_speakers: int = Form(default=None, description="Número exacto de hablantes (opcional)"),
    min_speakers: int = Form(default=None, description="Mínimo de hablantes (opcional)"),
    max_speakers: int = Form(default=None, description="Máximo de hablantes (opcional)"),
):
    logger.info("Request de diarización: filename=%s", file.filename)

    suffix = Path(file.filename or "").suffix.lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".audio") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Archivo de audio vacío")
            tmp.write(content)
            tmp_path = Path(tmp.name)

        segments = await diarization_manager.diarize(
            audio_path=tmp_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        return DiarizationResponse(
            segments=[
                DiarizationSegment(speaker=s.speaker, start=s.start, end=s.end)
                for s in segments
            ],
            num_speakers=len({s.speaker for s in segments}),
        )

    except ValueError as e:
        logger.warning("Error de validación: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error interno durante diarización")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
