# Qwen3-ASR Server

**音频 → 文本，带说话人分离。这是一条链路的第一个环节，链路的终点是听起来像真人的声音。**

[English](README.md) | [Español](README.es.md) | **中文** | [日本語](README.ja.md)

---

## 为什么会有这个项目

YouTube 的自动配音让我抓狂。一个女人在用日语说话，西班牙语配音却给她配了个普通男声。或者更糟：机械的声音让你十秒内就想关掉视频。

我已经有了一个用克隆声音把文本转成音频的服务器（[qwen-tts-server](https://github.com/feojeda/qwen-tts-server)）。逻辑上的下一步就是闭合这条链路：

```
原始音频 → 文本（ASR）→ 翻译 → 克隆声音（TTS）
```

这个项目是第一个环节：**将音频转化为文本，知道谁在什么时候说了什么**。链路的其余部分——翻译、克隆声音、生成最终音频——由其他服务处理。每个服务只做一件事。

---

## 它在哪里发挥作用

```
qwen-asr-server  (本项目)          qwen-tts-server
  :8001                                :8000
  音频 → 文本 + 说话人分离              文本 → 用克隆声音生成音频
       │                                     │
       └──────────────┬──────────────────────┘
                      │
              ttsQwen (前端)
              编排整个流水线：
              录音 → 转录 → 翻译 → 声音克隆 → 生成音频
```

每个服务器只做一件事，并把它做好。前端通过 REST 调用它们。如果我明天换掉 ASR 引擎，前端甚至都不会察觉。

---

## 目前的功能

- **音频转文本转录**，使用 [`Qwen3-ASR`](https://huggingface.co/collections/Qwen/qwen3-asr) 模型（0.6B 和 1.7B）
- **说话人分离**，通过 [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) 实现——谁在什么时候说话
- **兼容 OpenAI 的 API**（`/v1/audio/transcriptions`，`/v1/models`）
- **长音频自动分段**（超过 30 分钟）
- **优雅降级**：如果 pyannote 失败，依然能转录并告知你
- **针对 CPU 优化**：Apple Silicon M4 上使用 `bfloat16`，0.6B 模型达到约 4.5 倍实时速度

---

## 快速开始

### 环境要求

- Python 3.12+
- ffmpeg（macOS 上 `brew install ffmpeg`）
- 约 4 GB 空闲内存（0.6B 模型）

### 安装

```bash
git clone https://github.com/feojeda/qwen-asr-server
cd qwen-asr-server
python3 -m venv .venv && source .venv/bin/activate

# Torch（CPU，Apple Silicon）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 运行

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

首次请求：约 60 秒（下载模型）。后续请求：即时响应。

### 测试

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

将音频转录为文本。可选说话人分离。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `file` | file | *必填* | 音频文件（mp3、wav、webm、ogg、m4a、flac） |
| `language` | string | `auto` | 语言代码：`en`、`es`、`zh`、`fr`、`de`、`pt` 等 |
| `prompt` | string | `""` | 用于引导模型的上下文 |
| `diarize` | bool | `false` | 启用说话人分离 |
| `num_speakers` | int | — | 说话人的确切数量 |
| `min_speakers` | int | — | 说话人的最少数量 |
| `max_speakers` | int | — | 说话人的最多数量 |

**简单响应：**

```json
{
  "text": "Good morning everyone.",
  "language": "English",
  "duration": 5.3,
  "processing_time": 2.1
}
```

**带说话人分离的响应：**

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

如果 pyannote 失败，响应会包含 `"diarization_failed": true`，而不是返回 500 错误。

### `POST /v1/audio/diarization`

仅进行说话人分离，不转录：

```bash
curl -X POST "http://localhost:8001/v1/audio/diarization" \
  -F "file=@meeting.mp3" \
  -F "min_speakers=2"
```

### `GET /health` · `GET /v1/models`

标准的健康检查和模型列表端点（兼容 OpenAI）。

---

## 配置

环境变量（`.env` 或行内设置）：

| 变量 | 默认值 | 说明 |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-0.6B` | ASR 模型（0.6B 或 1.7B） |
| `DEVICE` | `cpu` | 设备（`cpu`、`cuda`） |
| `PORT` | `8001` | 服务器端口 |
| `MAX_AUDIO_DURATION` | `3600` | 最大音频时长（秒） |
| `CHUNK_DURATION` | `30` | 每个分段的秒数 |
| `CHUNK_THRESHOLD_MINUTES` | `30` | 超过 N 分钟的音频启用分段 |
| `HF_TOKEN` | — | HuggingFace 令牌，用于 pyannote（说话人分离） |

### 说话人分离（可选）

需要接受 pyannote 的使用条款并获取令牌：

1. 在 [huggingface.co](https://huggingface.co/join) 创建账户
2. 接受条款：[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. 在 [设置 → 令牌](https://huggingface.co/settings/tokens) 生成令牌
4. `export HF_TOKEN=hf_xxx` 或将其添加到 `.env`

---

## 性能

在 **Mac Mini M4（10 核，16 GB 内存）** 上使用 `torch.bfloat16` 的实测数据：

| 模型 | 内存 | 初始加载 | 4 秒音频 | 实时倍率 |
|-------|-----|--------------|----------|------------------|
| **0.6B** | ~4 GB | ~67s | ~18s | **~4.5x** |
| 1.7B | ~8 GB | ~147s | >10 min | >150x |

**CPU 环境下请使用 0.6B。** 1.7B 模型适用于 GPU。

---

## 待完成事项（路线图）

- **音频分段 + 文本提取**：新增端点，返回指定说话人的裁剪音频及其转录文本。需要此功能来为每个真实说话人自动化声音克隆。
- **GPU 支持（NVIDIA）**：目前针对 Apple Silicon M4 优化。需适配 CUDA，同时不牺牲当前部署的简洁性。
- **自动说话人标注**：使用声音嵌入或元数据将 `SPEAKER_00` 映射为真实姓名。
- **集成翻译**：可选端点，在原文旁边返回翻译后的文本。

---

## 测试

```bash
# 单元测试（快速，无需 GPU，适用于 CI）
python -m pytest test_server.py test_schemas.py -v

# 集成测试（需要真实模型，约 40 秒，仅限本地运行）
python -m pytest test_integration.py -v --run-integration
```

---

## 技术栈

- **FastAPI** + **uvicorn** — REST API
- **qwen-asr** — 官方 Qwen3-ASR 封装
- **pyannote.audio** — 说话人分离
- **PyTorch** — 推理后端
- **ffmpeg** — 音频转换与分割

---

## 许可证

代码：MIT。Qwen3-ASR 模型：Apache 2.0。pyannote：CC-BY-4.0。
