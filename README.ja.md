# Qwen3-ASR Server

**音声 → テキスト（話者分離付き）。本物の人間のように聞こえる音声で終わるチェーンの最初のリンク。**

[English](README.md) | [Español](README.es.md) | [中文](README.zh.md) | **日本語**

---

## なぜこれを作ったのか

YouTubeの自動吹き替えには本当にイライラさせられます。日本語を話す女性が、スペイン語の吹き替えでは無機質な男性の声になる。さらに悪いのは、ロボットのような声で10秒で動画を閉じたくなるケースです。

クローン音声を使ってテキストを音声に変換するサーバー（[qwen-tts-server](https://github.com/feojeda/qwen-tts-server)）は既に持っていました。論理的な次のステップは、ループを閉じることでした：

```
Original audio → Text (ASR) → Translation → Cloned voice (TTS)
```

このプロジェクトは最初のリンクです：**音声をテキストに変換し、誰がいつ何を言ったかを把握する**。残りのチェーン（翻訳、音声クローン、最終音声の生成）は他のサービスが担当します。それぞれ単一の責務を持ちます。

---

## 全体の中での位置付け

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

各サーバーは一つのことをしっかりこなします。フロントエンドはREST経由でこれらを利用します。明日ASRエンジンを交換しても、フロントエンドは気づくことすらありません。

---

## 現在の機能

- [`Qwen3-ASR`](https://huggingface.co/collections/Qwen/qwen3-asr)モデル（0.6Bおよび1.7B）を使用した**音声テキスト変換**
- [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)による**話者分離** — 誰がいつ話したか
- **OpenAI互換API**（`/v1/audio/transcriptions`、`/v1/models`）
- 長時間の音声（30分以上）の**自動チャンク分割**
- **段階的機能低下**：pyannoteが失敗しても、文字起こしは続行し通知します
- **CPU最適化**：Apple Silicon M4で`bfloat16`、0.6Bモデルで約4.5倍のリアルタイム性能

---

## クイックスタート

### 必要条件

- Python 3.12+
- ffmpeg（macOSでは `brew install ffmpeg`）
- 約4GBの空きRAM（0.6Bモデル）

### インストール

```bash
git clone https://github.com/feojeda/qwen-asr-server
cd qwen-asr-server
python3 -m venv .venv && source .venv/bin/activate

# Torch (CPU, Apple Silicon)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 実行

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

初回リクエスト：約60秒（モデルをダウンロード）。以降のリクエスト：即時。

### テスト

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

音声をテキストに変換します。話者分離はオプションです。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `file` | file | *必須* | 音声ファイル（mp3, wav, webm, ogg, m4a, flac） |
| `language` | string | `auto` | 言語コード：`en`、`es`、`zh`、`fr`、`de`、`pt` など |
| `prompt` | string | `""` | モデルを導くためのコンテキスト |
| `diarize` | bool | `false` | 話者分離を有効にする |
| `num_speakers` | int | — | 正確な話者数 |
| `min_speakers` | int | — | 最小話者数 |
| `max_speakers` | int | — | 最大話者数 |

**シンプルな応答：**

```json
{
  "text": "Good morning everyone.",
  "language": "English",
  "duration": 5.3,
  "processing_time": 2.1
}
```

**話者分離付きの応答：**

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

pyannoteが失敗した場合、500エラーの代わりに `"diarization_failed": true` が応答に含まれます。

### `POST /v1/audio/diarization`

文字起こしなしの話者分離のみ：

```bash
curl -X POST "http://localhost:8001/v1/audio/diarization" \
  -F "file=@meeting.mp3" \
  -F "min_speakers=2"
```

### `GET /health` · `GET /v1/models`

標準的なヘルスチェックとモデル一覧のエンドポイント（OpenAI互換）。

---

## 設定

環境変数（`.env` またはインライン）：

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-0.6B` | ASRモデル（0.6Bまたは1.7B） |
| `DEVICE` | `cpu` | デバイス（`cpu`、`cuda`） |
| `PORT` | `8001` | サーバーポート |
| `MAX_AUDIO_DURATION` | `3600` | 最大音声長（秒） |
| `CHUNK_DURATION` | `30` | チャンクあたりの秒数 |
| `CHUNK_THRESHOLD_MINUTES` | `30` | N分を超える音声でチャンク分割を有効にする |
| `HF_TOKEN` | — | pyannote（話者分離）用のHuggingFaceトークン |

### 話者分離（オプション）

pyannoteの利用規約への同意とトークンが必要です：

1. [huggingface.co](https://huggingface.co/join)でアカウントを作成
2. 利用規約に同意：[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. [Settings → Tokens](https://huggingface.co/settings/tokens)でトークンを生成
4. `export HF_TOKEN=hf_xxx` または `.env` に追加

---

## パフォーマンス

**Mac Mini M4（10コア、16GB RAM）** での `torch.bfloat16` 使用時の測定値：

| モデル | RAM | 初回ロード | 4秒の音声 | リアルタイム倍率 |
|-------|-----|--------------|----------|------------------|
| **0.6B** | ~4 GB | ~67s | ~18s | **~4.5x** |
| 1.7B | ~8 GB | ~147s | >10 min | >150x |

**CPUの場合は0.6Bを使用してください。** 1.7BモデルはGPU向けです。

---

## 今後の予定（ロードマップ）

- **音声セグメント + テキスト抽出**：特定の話者のトリミング音声とその文字起こしを返す新しいエンドポイント。各実話者の音声クローンを自動化するために必要。
- **GPU対応（NVIDIA）**：現在はApple Silicon M4向けに最適化。現在のデプロイのシンプルさを損なわずにCUDAに対応させる。
- **自動話者ラベリング**：音声埋め込みやメタデータを使って `SPEAKER_00` → 実名にマッピング。
- **統合翻訳**：元のテキストとともに翻訳テキストを返すオプションのエンドポイント。

---

## テスト

```bash
# ユニットテスト（高速、GPU不要、CI対応）
python -m pytest test_server.py test_schemas.py -v

# 統合テスト（実モデル、約40秒、ローカルのみ）
python -m pytest test_integration.py -v --run-integration
```

---

## 技術スタック

- **FastAPI** + **uvicorn** — REST API
- **qwen-asr** — Qwen3-ASR公式ラッパー
- **pyannote.audio** — 話者分離
- **PyTorch** — 推論バックエンド
- **ffmpeg** — 音声変換とセグメンテーション

---

## ライセンス

コード：MIT。Qwen3-ASRモデル：Apache 2.0。pyannote：CC-BY-4.0。
