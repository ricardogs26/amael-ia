FROM python:3.10-slim

WORKDIR /app

# ── 1. Dependencias del sistema ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsox-dev \
    sox \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Pip base — instalamos setuptools de forma que pip lo inyecte
#      en sus build envs aislados (PEP 517). La clave es --no-build-isolation
#      para paquetes legacy que usan setup.py con pkg_resources ───────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools>=68" wheel

# ── 3. PyTorch CPU — índice SOLO CPU ──────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 4. openai-whisper con --no-build-isolation para que use el setuptools
#      del sistema en lugar del entorno aislado donde pkg_resources no existe ──
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# ── 5. Clonar CosyVoice ───────────────────────────────────────────────────────
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice

# ── 6. Filtrar dependencias incompatibles con CPU ─────────────────────────────
#    - onnxruntime-gpu  → CPU no tiene CUDA
#    - deepspeed        → requiere compilar con CUDA toolkit completo
#    - openai-whisper   → ya instalado arriba
RUN grep -vE "onnxruntime-gpu|deepspeed|openai-whisper" \
    /opt/CosyVoice/requirements.txt > /opt/CosyVoice/requirements_cpu.txt

# ── 7. onnxruntime CPU ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir onnxruntime==1.18.0

# ── 8. Resto de dependencias de CosyVoice ─────────────────────────────────────
RUN pip install --no-cache-dir --no-build-isolation \
    -r /opt/CosyVoice/requirements_cpu.txt

# ── 9. Servidor web ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# ── 10. PYTHONPATH correcto ───────────────────────────────────────────────────
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS"

# ── 11. Descargar modelo (capa separada = cacheable) ──────────────────────────
RUN python -c "\
    from modelscope import snapshot_download; \
    snapshot_download('iic/CosyVoice-300M', local_dir='/app/pretrained_models/CosyVoice-300M')"

# ── 12. Copiar API ────────────────────────────────────────────────────────────
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]