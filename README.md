# Qwen3-ASR Server

Servidor REST de **Audio-to-Text (ASR)** con soporte opcional de **diarización de hablantes** basado en los modelos [`Qwen3-ASR`](https://huggingface.co/collections/Qwen/qwen3-asr) de HuggingFace, optimizado para ejecutarse en **CPU** (incluyendo Apple Silicon / ARM64).

Este servidor está diseñado para ser consumido por `ttsQwen` u otras aplicaciones mediante una API REST compatible con OpenAI-style.

> **⚠️ Recomendación para CPU:** En Mac Mini / CPU el modelo **1.7B es extremadamente lento** (~3-5 min por minuto de audio). Se recomienda usar el modelo **0.6B** como default para uso práctico en CPU. Ver [sección de rendimiento](#rendimiento-en-cpu) abajo.

---

## Stack

- **FastAPI** — API REST
- **uvicorn** — Servidor ASGI
- **qwen-asr** — Wrapper oficial del modelo Qwen3-ASR
- **pyannote.audio** — Diarización de hablantes (state-of-the-art)
- **torch** — Backend de inferencia
- **ffmpeg** — Conversión y segmentación de audio

---

## Requisitos

- Python **3.10 - 3.13** (recomendado 3.13)
- **ffmpeg** instalado en el sistema
- ~**4 GB de RAM libre** para el modelo 0.6B con `bfloat16`
- ~**6 GB de RAM libre** para el modelo 1.7B con `bfloat16`
- ~**5 GB de espacio en disco** para descargar el modelo desde HuggingFace

### Instalar ffmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

---

## Instalación

1. Clonar o crear el directorio del proyecto:
```bash
cd qwen-asr-server
```

2. Crear un entorno virtual (recomendado):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar PyTorch CPU (ARM64 / x86_64):

**macOS ARM64 (Apple Silicon):**
```bash
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

**Linux CPU:**
```bash
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

4. Instalar el resto de dependencias:
```bash
pip install -r requirements.txt
```

> Nota: La primera vez que ejecutes el servidor descargará automáticamente el modelo de HuggingFace (~1.5 GB para 0.6B, ~4.2 GB para 1.7B). Asegúrate de tener espacio y conexión estable.

---

## Configuración de Diarización (opcional)

La diarización usa **pyannote/speaker-diarization-community-1** que requiere:

1. **Cuenta en HuggingFace** (gratis): https://huggingface.co/join
2. **Aceptar los términos** del modelo: https://huggingface.co/pyannote/speaker-diarization-community-1
3. **Crear un token de acceso** (read): https://huggingface.co/settings/tokens
4. **Exportar el token** como variable de entorno:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Sin `HF_TOKEN`, la diarización no funcionará en la primera ejecución (necesita descargar el modelo). Si ya descargaste los modelos manualmente, puede funcionar sin token.

---

## Uso

### Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-0.6B` | Modelo ASR de HuggingFace |
| `DEVICE` | `cpu` | Dispositivo de inferencia (`cpu` o `cuda`) |
| `PORT` | `8001` | Puerto del servidor |
| `HOST` | `0.0.0.0` | Host de binding |
| `MAX_AUDIO_DURATION` | `3600` | Duración máxima de audio en segundos |
| `CHUNK_DURATION` | `30` | Duración de cada chunk para audios largos (segundos) |
| `CHUNK_THRESHOLD_MINUTES` | `30` | Umbral en minutos para activar chunking |
| `LOG_LEVEL` | `INFO` | Nivel de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `HF_TOKEN` | `None` | Token de HuggingFace para modelos pyannote (diarización) |

### Correr el servidor

```bash
# Recomendado para CPU (modelo 0.6B - más rápido)
uvicorn main:app --host 0.0.0.0 --port 8001

# Con diarización (requiere HF_TOKEN)
HF_TOKEN=hf_xxx uvicorn main:app --host 0.0.0.0 --port 8001
```

---

## Endpoints

### `GET /health`
```bash
curl http://localhost:8001/health
```
```json
{
  "status": "healthy",
  "model_loaded": false,
  "diarization_loaded": false,
  "device": "cpu",
  "model_name": "Qwen/Qwen3-ASR-0.6B"
}
```

### `GET /v1/models`
```bash
curl http://localhost:8001/v1/models
```

### `POST /v1/audio/transcriptions`

#### Transcripción simple
```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "language=es" \
  -F "prompt=Contexto sobre tecnología"
```

**Response:**
```json
{
  "text": "Texto transcrito completo...",
  "language": "es",
  "duration": 12.5,
  "processing_time": 4.2
}
```

#### Transcripción con diarización
```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@meeting.mp3" \
  -F "language=es" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

**Response:**
```json
{
  "text": "Hola buenos días. Buenos días María...",
  "language": "es",
  "duration": 120.5,
  "processing_time": 45.3,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 3.2,
      "text": "Hola buenos días."
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.5,
      "end": 8.1,
      "text": "Buenos días María."
    }
  ]
}
```

**Parámetros:**
- `file`: archivo de audio (`mp3`, `wav`, `webm`, `ogg`, `m4a`, `flac`, `aac`, etc.)
- `language`: `auto` (default), `es`, `en`, `zh`, `fr`, `de`, `pt`, etc.
- `prompt`: contexto opcional para guiar al modelo
- `diarize`: `true` o `false` (default) — activa diarización de hablantes
- `num_speakers`: número exacto de hablantes (opcional, solo si `diarize=true`)
- `min_speakers`: mínimo de hablantes (opcional, solo si `diarize=true`)
- `max_speakers`: máximo de hablantes (opcional, solo si `diarize=true`)

### `POST /v1/audio/diarization`

Diarización pura (sin transcripción):

```bash
curl -X POST "http://localhost:8001/v1/audio/diarization" \
  -F "file=@meeting.mp3" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

**Response:**
```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 12.5},
    {"speaker": "SPEAKER_01", "start": 13.1, "end": 25.8}
  ],
  "num_speakers": 2
}
```

---

## Consumo desde Python (httpx)

### Transcripción simple
```python
import httpx

async with httpx.AsyncClient() as client:
    with open("audio.mp3", "rb") as f:
        response = await client.post(
            "http://localhost:8001/v1/audio/transcriptions",
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={"language": "es", "prompt": ""},
        )
    result = response.json()
    print(result["text"])
```

### Transcripción con diarización
```python
import httpx

async with httpx.AsyncClient() as client:
    with open("meeting.mp3", "rb") as f:
        response = await client.post(
            "http://localhost:8001/v1/audio/transcriptions",
            files={"file": ("meeting.mp3", f, "audio/mpeg")},
            data={
                "language": "es",
                "diarize": "true",
                "min_speakers": "2",
                "max_speakers": "4",
            },
        )
    result = response.json()
    for seg in result["segments"]:
        print(f"[{seg['speaker']}] {seg['start']:.1f}s - {seg['end']:.1f}s: {seg['text']}")
```

---

## Rendimiento en CPU

Tests realizados en **Mac Mini M4 (10 cores, 16 GB RAM)** con `torch.bfloat16`:

| Modelo | Tamaño | Carga | Inferencia 4s audio | Factor real-time |
|--------|--------|-------|---------------------|------------------|
| **Qwen3-ASR-0.6B** | ~1.5 GB | ~67s | ~18s | **~4.5x** |
| **Qwen3-ASR-1.7B** | ~4.2 GB | ~147s | **>10 min** | **>150x** |

**Conclusión:** El modelo 1.7B es demasiado lento para uso práctico en CPU. El modelo **0.6B es el recomendado** para despliegues sin GPU.

---

## Docker

Construir y correr con Docker:

```bash
docker build -t qwen-asr-server .
docker run -p 8001:8001 --env HF_TOKEN=hf_xxx qwen-asr-server
```

---

## Notas sobre CPU y Apple Silicon

- **Carga lazy**: Los modelos se descargan y cargan en memoria en la **primera request**. Las siguientes son mucho más rápidas.
- **Optimización de memoria**: En Mac ARM64 y CPUs modernas el servidor intenta usar automáticamente **`torch.bfloat16`**, reduciendo el uso de RAM a la mitad. Si no está disponible, cae a `torch.float32`.
- **Threads**: El servidor deja que PyTorch use todos los cores disponibles por defecto. Puedes controlarlo con la variable de entorno `OMP_NUM_THREADS`.
- **No bloquea**: La inferencia síncrona del modelo se ejecuta en un `ThreadPoolExecutor` separado para no bloquear el event loop de FastAPI.

---

## Testing

Ejecutar tests de integración (sin descargar el modelo):

```bash
python test_server.py
```

Los tests verifican:
- Levantamiento de FastAPI y endpoints
- Conversión de audio con ffmpeg
- Chunking para audios largos
- Diarización (mockeada)
- Transcripción con diarización activada
- Manejo de errores (archivos vacíos, formatos inválidos)

---

## Troubleshooting

### `No space left on device` al descargar el modelo
El modelo pesa ~1.5 GB (0.6B) o ~4.2 GB (1.7B). Asegúrate de tener espacio suficiente en disco. Puedes cambiar la ubicación del caché de HuggingFace:
```bash
export HF_HOME=/ruta/con/espacio/.cache/huggingface
```

### `RuntimeError: Data processing error` durante descarga
Esto puede deberse a un caché corrupto. Bórralo:
```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B
```

### `HTTP 403` o `401` al cargar pyannote para diarización
No has aceptado los términos del modelo o el token es inválido. Ve a https://huggingface.co/pyannote/speaker-diarization-community-1 y acepta los términos, luego genera un token en https://huggingface.co/settings/tokens.

### El servidor tarda mucho en la primera request
Es comportamiento esperado (lazy loading). El modelo se está descargando/cargando en memoria. Dependiendo de tu conexión, puede tardar varios minutos la primera vez.

### Out of Memory (OOM) en CPU
Si tienes menos de 8 GB de RAM, usa el modelo más pequeño:
```bash
export MODEL_NAME=Qwen/Qwen3-ASR-0.6B
```

---

## Licencia

El código de este servidor se proporciona bajo licencia MIT. Los modelos `Qwen3-ASR` están bajo licencia Apache-2.0. El pipeline de diarización `pyannote/speaker-diarization-community-1` está bajo CC-BY-4.0.
