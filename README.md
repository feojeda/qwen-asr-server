# Qwen3-ASR Server

**Audio → Text with speaker diarization. The first link in a chain that ends with voices that sound like real people.**

[**English**](README.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md)

---

## Why this exists

YouTube's auto-dubbing drives me crazy. A woman speaking in Japanese, and the Spanish dub gives her a generic male voice. Or worse: robotic voices that push you out of the video in 10 seconds.

I already had a server that converts text to audio using cloned voices ([qwen-tts-server](https://github.com/feojeda/qwen-tts-server)). The logical next step was to close the loop:

```
Original audio → Text (ASR) → Translation → Cloned voice (TTS)
```

This project is the first link: **turning audio into text, knowing who said what and when**. The rest of the chain — translating, cloning voices, generating final audio — is handled by other services. Each with a single responsibility.

---

## Where it fits

```
qwen-asr-server  (this project)     qwen-tts-server
  :8001                                :8000
  Audio → Text + Diarization         Text → Audio with cloned voice
       │                                     │
       └──────────────┬──────────────────────┘
                      │
              ttsQwen (frontend)
              Orchestrates the full pipeline:
              Record → Transcribe → Translate → Voice Clone → Generate audio
```

Each server does one thing and does it well. The frontend consumes them via REST. If I swap the ASR engine tomorrow, the frontend won't even notice.

---

## What it does now

- **Audio-to-text transcription** with [`Qwen3-ASR`](https://huggingface.co/collections/Qwen/qwen3-asr) models (0.6B and 1.7B)
- **Speaker diarization** via [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) — who spoke and when
- **OpenAI-compatible API** (`/v1/audio/transcriptions`, `/v1/models`)
- **Automatic chunking** for long audio (>30 min)
- **Graceful degradation**: if pyannote fails, it still transcribes and lets you know
- **CPU-optimized**: `bfloat16` on Apple Silicon M4, ~4.5x real-time with the 0.6B model

---

## Quick Start

### Requirements

- Python 3.12+
- ffmpeg (`brew install ffmpeg` on macOS)
- ~4 GB free RAM (0.6B model)

### Installation

```bash
git clone https://github.com/feojeda/qwen-asr-server
cd qwen-asr-server
python3 -m venv .venv && source .venv/bin/activate

# Torch (CPU, Apple Silicon)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

First request: ~60s (downloads the model). Subsequent requests: instant.

### Test

```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "language=en"
```

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "English",
  "duration": 4.2,
  "processing_time": 0.9
}
```

---

## API

### `POST /v1/audio/transcriptions`

Transcribes audio to text. Optional diarization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | *required* | Audio file (mp3, wav, webm, ogg, m4a, flac) |
| `language` | string | `auto` | Language code: `en`, `es`, `zh`, `fr`, `de`, `pt`, etc. |
| `prompt` | string | `""` | Context to guide the model |
| `diarize` | bool | `false` | Enable speaker diarization |
| `num_speakers` | int | — | Exact number of speakers |
| `min_speakers` | int | — | Minimum number of speakers |
| `max_speakers` | int | — | Maximum number of speakers |

**Simple response:**

```json
{
  "text": "Good morning everyone.",
  "language": "English",
  "duration": 5.3,
  "processing_time": 2.1
}
```

**Response with diarization:**

```json
{
  "text": "Hi Maria. Hi John.",
  "language": "English",
  "duration": 8.5,
  "processing_time": 4.2,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 2.1,
      "text": "Hi Maria.",
      "language": "English"
    },
    {
      "speaker": "SPEAKER_01",
      "start": 2.5,
      "end": 4.0,
      "text": "Hi John.",
      "language": "English"
    }
  ]
}
```

If pyannote fails, the response includes `"diarization_failed": true` instead of a 500 error.

### `POST /v1/audio/diarization`

Diarization only, without transcription:

```bash
curl -X POST "http://localhost:8001/v1/audio/diarization" \
  -F "file=@meeting.mp3" \
  -F "min_speakers=2"
```

### `GET /health` · `GET /v1/models`

Standard health check and model listing endpoints (OpenAI-compatible).

---

## Configuration

Environment variables (`.env` or inline):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-0.6B` | ASR model (0.6B or 1.7B) |
| `DEVICE` | `cpu` | Device (`cpu`, `cuda`) |
| `PORT` | `8001` | Server port |
| `MAX_AUDIO_DURATION` | `3600` | Max audio duration in seconds |
| `CHUNK_DURATION` | `30` | Seconds per chunk |
| `CHUNK_THRESHOLD_MINUTES` | `30` | Enable chunking for audio longer than N minutes |
| `HF_TOKEN` | — | HuggingFace token for pyannote (diarization) |

### Diarization (optional)

Requires accepting pyannote's terms and a token:

1. Create an account at [huggingface.co](https://huggingface.co/join)
2. Accept terms: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Generate a token at [Settings → Tokens](https://huggingface.co/settings/tokens)
4. `export HF_TOKEN=hf_xxx` or add it to `.env`

---

## Performance

Measurements on **Mac Mini M4 (10 cores, 16 GB RAM)** with `torch.bfloat16`:

| Model | RAM | Initial load | 4s audio | Real-time factor |
|-------|-----|--------------|----------|------------------|
| **0.6B** | ~4 GB | ~67s | ~18s | **~4.5x** |
| 1.7B | ~8 GB | ~147s | >10 min | >150x |

**For CPU, use 0.6B.** The 1.7B model is for GPU.

---

## What's missing (roadmap)

- **Segment audio + text extraction**: New endpoint that returns the trimmed audio of a specific speaker along with its transcript. Needed to automate voice cloning for each real speaker.
- **GPU support (NVIDIA)**: Currently optimized for Apple Silicon M4. Adapt for CUDA without losing the simplicity of the current deployment.
- **Auto-speaker-labeling**: Map `SPEAKER_00` → real name using voice embeddings or metadata.
- **Integrated translation**: Optional endpoint that returns translated text alongside the original.

---

## Tests

```bash
# Unit tests (fast, no GPU, CI)
python -m pytest test_server.py test_schemas.py -v

# Integration tests (real model, ~40s, local only)
python -m pytest test_integration.py -v --run-integration
```

---

## Stack

- **FastAPI** + **uvicorn** — REST API
- **qwen-asr** — Official Qwen3-ASR wrapper
- **pyannote.audio** — Speaker diarization
- **PyTorch** — Inference backend
- **ffmpeg** — Audio conversion and segmentation

---

## License

Code: MIT. Qwen3-ASR model: Apache 2.0. pyannote: CC-BY-4.0.
