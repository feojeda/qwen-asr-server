# Qwen3-ASR Server

**Audio → Texto con diarización de hablantes. El primer eslabón de una cadena que termina con voces que suenan como personas reales.**

[English](README.md) | **Español** | [中文](README.zh.md) | [日本語](README.ja.md)

---

## ¿Por qué existe esto?

El doblaje automático de YouTube me saca de quicio. Una mujer hablando en japonés, y el doblaje al español le pone voz de hombre genérica. O peor: voces robóticas que te expulsan del video en 10 segundos.

Ya tenía un servidor que convierte texto en audio con voces clonadas ([qwen-tts-server](https://github.com/feojeda/qwen-tts-server)). El paso lógico era cerrar el círculo:

```
Audio original → Texto (ASR) → Traducción → Voz clonada (TTS)
```

Este proyecto es el primer eslabón: **pasar de audio a texto, sabiendo quién dijo qué y cuándo**. El resto de la cadena — traducir, clonar voces, generar audio final — lo manejan otros servicios. Cada uno con una responsabilidad única.

---

## Dónde encaja

```
qwen-asr-server  (este proyecto)     qwen-tts-server
  :8001                                :8000
  Audio → Texto + Diarización         Texto → Audio con voz clonada
       │                                     │
       └──────────────┬──────────────────────┘
                      │
              ttsQwen (frontend)
              Orquesta la cadena completa:
              Grabar → Transcribir → Traducir → Voice Clone → Generar audio
```

Cada servidor hace una sola cosa y la hace bien. El frontend los consume vía REST. Si mañana cambio el motor ASR por otro, el frontend ni se entera.

---

## Qué hace ahora

- **Transcripción de audio a texto** con modelos [`Qwen3-ASR`](https://huggingface.co/collections/Qwen/qwen3-asr) (0.6B y 1.7B)
- **Diarización de hablantes** vía [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) — quién habló y en qué momento
- **API OpenAI-compatible** (`/v1/audio/transcriptions`, `/v1/models`)
- **Chunking automático** para audios largos (>30 min)
- **Graceful degradation**: si pyannote falla, transcribe igual y te avisa
- **CPU-optimizado**: `bfloat16` en Apple Silicon M4, ~4.5x real-time con el modelo 0.6B

---

## Quick Start

### Requisitos

- Python 3.12+
- ffmpeg (`brew install ffmpeg` en macOS)
- ~4 GB RAM libre (modelo 0.6B)

### Instalación

```bash
git clone https://github.com/feojeda/qwen-asr-server
cd qwen-asr-server
python3 -m venv .venv && source .venv/bin/activate

# Torch (CPU, Apple Silicon)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Arrancar

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Primera request: ~60s (descarga el modelo). Siguientes: instantáneas.

### Probar

```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "language=es"
```

```json
{
  "text": "Hola, esto es una prueba de transcripción.",
  "language": "Spanish",
  "duration": 4.2,
  "processing_time": 0.9
}
```

---

## API

### `POST /v1/audio/transcriptions`

Transcribe audio a texto. Con diarización opcional.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `file` | file | *requerido* | Audio (mp3, wav, webm, ogg, m4a, flac) |
| `language` | string | `auto` | Código de idioma: `es`, `en`, `zh`, `fr`, `de`, `pt`, etc. |
| `prompt` | string | `""` | Contexto para guiar al modelo |
| `diarize` | bool | `false` | Activar diarización de hablantes |
| `num_speakers` | int | — | Número exacto de hablantes |
| `min_speakers` | int | — | Mínimo de hablantes |
| `max_speakers` | int | — | Máximo de hablantes |

**Respuesta simple:**

```json
{
  "text": "Buenos días a todos los presentes.",
  "language": "Spanish",
  "duration": 5.3,
  "processing_time": 2.1
}
```

**Respuesta con diarización:**

```json
{
  "text": "Hola María. Hola Juan.",
  "language": "Spanish",
  "duration": 8.5,
  "processing_time": 4.2,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 2.1,
      "text": "Hola María.",
      "language": "Spanish"
    },
    {
      "speaker": "SPEAKER_01",
      "start": 2.5,
      "end": 4.0,
      "text": "Hola Juan.",
      "language": "Spanish"
    }
  ]
}
```

Si pyannote falla, la respuesta incluye `"diarization_failed": true` en lugar de un error 500.

### `POST /v1/audio/diarization`

Solo diarización, sin transcribir:

```bash
curl -X POST "http://localhost:8001/v1/audio/diarization" \
  -F "file=@reunion.mp3" \
  -F "min_speakers=2"
```

### `GET /health` · `GET /v1/models`

Endpoints estándar de salud y listado de modelos (OpenAI-compatible).

---

## Configuración

Variables de entorno (`.env` o directas):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-0.6B` | Modelo ASR (0.6B o 1.7B) |
| `DEVICE` | `cpu` | Dispositivo (`cpu`, `cuda`) |
| `PORT` | `8001` | Puerto del servidor |
| `MAX_AUDIO_DURATION` | `3600` | Máximo en segundos |
| `CHUNK_DURATION` | `30` | Segundos por chunk |
| `CHUNK_THRESHOLD_MINUTES` | `30` | Activar chunking a partir de N minutos |
| `HF_TOKEN` | — | Token HuggingFace para pyannote (diarización) |

### Diarización (opcional)

Requiere aceptar los términos de pyannote y un token:

1. Crear cuenta en [huggingface.co](https://huggingface.co/join)
2. Aceptar términos: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Generar token en [Settings → Tokens](https://huggingface.co/settings/tokens)
4. `export HF_TOKEN=hf_xxx` o agregarlo al `.env`

---

## Rendimiento

Mediciones en **Mac Mini M4 (10 cores, 16 GB RAM)** con `torch.bfloat16`:

| Modelo | RAM | Carga inicial | 4s de audio | Factor real-time |
|--------|-----|---------------|-------------|------------------|
| **0.6B** | ~4 GB | ~67s | ~18s | **~4.5x** |
| 1.7B | ~8 GB | ~147s | >10 min | >150x |

**Para CPU, usa 0.6B.** El modelo 1.7B es para GPU.

---

## Lo que falta (roadmap)

- **Extraer audio + texto por segmento**: Nuevo endpoint que devuelva el audio recortado de un hablante específico junto con su transcripción. Necesario para automatizar el voice cloning de cada hablante real.
- **Soporte GPU (NVIDIA)**: Actualmente optimizado para Apple Silicon M4. Adaptar para CUDA sin perder la simplicidad del deploy actual.
- **Auto-speaker-labeling**: Mapear `SPEAKER_00` → nombre real usando embeddings de voz o metadata.
- **Traducción integrada**: Endpoint opcional que devuelva texto traducido además del original.

---

## Tests

```bash
# Unitarios (rápidos, sin GPU, CI)
python -m pytest test_server.py test_schemas.py -v

# Integración (modelo real, ~40s, solo local)
python -m pytest test_integration.py -v --run-integration
```

---

## Stack

- **FastAPI** + **uvicorn** — API REST
- **qwen-asr** — Wrapper oficial de Qwen3-ASR
- **pyannote.audio** — Diarización de hablantes
- **PyTorch** — Backend de inferencia
- **ffmpeg** — Conversión y segmentación de audio

---

## Licencia

Código: MIT. Modelo Qwen3-ASR: Apache 2.0. pyannote: CC-BY-4.0.
