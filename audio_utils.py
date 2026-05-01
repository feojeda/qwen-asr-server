import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está instalado en el sistema. Por favor instálalo: brew install ffmpeg")


def get_audio_duration(audio_path: Path) -> float:
    """Obtiene la duración en segundos de un archivo de audio usando ffprobe/ffmpeg."""
    _check_ffmpeg()

    # Intentar con ffprobe primero
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            pass

    # Fallback: usar ffmpeg -i y parsear Duration de stderr
    result = subprocess.run(
        ["ffmpeg", "-i", str(audio_path)],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", result.stderr)
    if not match:
        raise ValueError(f"No se pudo obtener la duración del audio: {audio_path}")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def convert_to_wav(input_path: Path, output_path: Path) -> float:
    """Convierte cualquier audio a WAV 16kHz mono. Retorna la duración."""
    _check_ffmpeg()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    logger.info("Convirtiendo audio a WAV 16kHz mono: %s -> %s", input_path, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg error: %s", result.stderr)
        raise ValueError(f"Error al convertir audio con ffmpeg: {result.stderr[:500]}")

    duration = get_audio_duration(output_path)
    logger.info("Conversión exitosa. Duración: %.2f s", duration)
    return duration


def split_audio_into_chunks(
    audio_path: Path,
    chunk_duration: int,
    output_dir: Path,
) -> List[Path]:
    """Divide un archivo WAV en chunks de chunk_duration segundos usando ffmpeg segment."""
    _check_ffmpeg()

    output_pattern = str(output_dir / "chunk_%03d.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-c", "copy",
        output_pattern,
    ]

    logger.info("Dividiendo audio en chunks de %d segundos...", chunk_duration)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg segment error: %s", result.stderr)
        raise ValueError(f"Error al dividir audio: {result.stderr[:500]}")

    chunks = sorted(output_dir.glob("chunk_*.wav"))
    logger.info("Audio dividido en %d chunks", len(chunks))
    return chunks


def extract_segment(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> None:
    """Extrae un segmento de audio [start, end] en segundos y guarda como WAV 16kHz mono."""
    _check_ffmpeg()

    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    logger.debug("Extrayendo segmento %.2f-%.2f s -> %s", start, end, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg segment extraction error: %s", result.stderr)
        raise ValueError(f"Error al extraer segmento: {result.stderr[:500]}")
