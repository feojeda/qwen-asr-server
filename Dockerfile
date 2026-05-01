# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Evitar prompts de apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar ffmpeg y dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox-dev \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Instalar torch CPU primero (reduce tamaño de imagen)
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY *.py .

# Variables de entorno por defecto
ENV MODEL_NAME=Qwen/Qwen3-ASR-0.6B
ENV DEVICE=cpu
ENV PORT=8001
ENV HOST=0.0.0.0
ENV MAX_AUDIO_DURATION=3600
ENV CHUNK_DURATION=30
ENV CHUNK_THRESHOLD_MINUTES=30
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Opcional: token de HuggingFace para descargar modelos de pyannote
# ENV HF_TOKEN=your_hf_token_here

# Exponer puerto
EXPOSE 8001

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
