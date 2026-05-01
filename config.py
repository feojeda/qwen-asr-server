import os
from pathlib import Path

from dotenv import load_dotenv

# Cargar variables desde .env si existe (no falla si no existe)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # Intenta cargar .env del directorio de trabajo


class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-ASR-0.6B")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    PORT: int = int(os.getenv("PORT", "8001"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    MAX_AUDIO_DURATION: int = int(os.getenv("MAX_AUDIO_DURATION", "3600"))
    CHUNK_DURATION: int = int(os.getenv("CHUNK_DURATION", "30"))
    CHUNK_THRESHOLD_MINUTES: int = int(os.getenv("CHUNK_THRESHOLD_MINUTES", "30"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
    ).split(",")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN", None)


settings = Settings()
