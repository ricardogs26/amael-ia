# piper-service — imagen CPU-only (~400MB total)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Instalar piper-tts ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    piper-tts==1.2.0 \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.32.1" \
    pydantic==2.10.4

# ── 2. Descargar modelos de voz en tiempo de build ───────────────────────────
#    Los modelos van baked en la imagen — sin PVC, sin job de descarga
RUN mkdir -p /models

# Español México — voz claude alta calidad (~63 MB)
RUN curl -sL -o /models/es_MX-claude-high.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_MX/claude/high/es_MX-claude-high.onnx" && \
    curl -sL -o /models/es_MX-claude-high.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_MX/claude/high/es_MX-claude-high.onnx.json"

# Inglés US — voz lessac alta calidad (~63 MB)
RUN curl -sL -o /models/en_US-lessac-high.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx" && \
    curl -sL -o /models/en_US-lessac-high.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json"

# Español Argentina — voz daniela alta calidad (~108 MB)
# Acento latinoamericano neutro — usado como default para notas de voz
COPY es_AR-daniela-high.onnx /models/
COPY es_AR-daniela-high.onnx.json /models/

# ── 3. Código del servicio ────────────────────────────────────────────────────
COPY main.py .

ENV PIPER_MODELS_DIR=/models
ENV PIPER_DEFAULT_VOICE=es_AR-daniela-high
ENV TTS_MAX_CHARS=1000

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:8010/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
