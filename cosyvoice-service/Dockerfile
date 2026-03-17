# cosyvoice-service — imagen CPU-only
FROM python:3.11-slim

# Deps del sistema para soundfile + git para clonar CosyVoice
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        ffmpeg \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 0. setuptools primero (evita pkg_resources missing en paquetes legacy) ───
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 1. PyTorch CPU (evita ~3GB extra de CUDA) ────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. Clonar CosyVoice CON submódulos ───────────────────────────────────────
# third_party/Matcha-TTS es la versión parcheada sin la dependencia de lightning
RUN git clone --depth=1 --recurse-submodules \
    https://github.com/FunAudioLLM/CosyVoice.git /cosyvoice

# Fix huggingface_hub compat: cached_download fue removido en >= 0.17
# transformers >= 4.x requiere huggingface_hub >= 0.24, por lo que no podemos
# bajar la versión — en su lugar parcheamos el source de CosyVoice directamente.
RUN find /cosyvoice -name "*.py" | \
    xargs grep -l "cached_download" 2>/dev/null | \
    xargs -I{} sed -i \
      's/from huggingface_hub import cached_download/from huggingface_hub import hf_hub_download as cached_download/g' \
      {} || true

# ── 3. Deps de CosyVoice para inferencia CPU ─────────────────────────────────
#    Excluidos: deepspeed, onnxruntime-gpu, tensorrt-cu12* (todos GPU-only)
#    onnxruntime = versión CPU en lugar de onnxruntime-gpu
RUN pip install --no-cache-dir \
    conformer==0.3.2 \
    "diffusers>=0.31.0" \
    gdown==5.1.0 \
    grpcio==1.57.0 \
    grpcio-tools==1.57.0 \
    hydra-core==1.3.2 \
    HyperPyYAML==1.2.3 \
    inflect==7.3.1 \
    librosa==0.10.2 \
    lightning==2.2.4 \
    matplotlib==3.7.5 \
    modelscope==1.20.0 \
    networkx==3.1 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    onnx==1.16.0 \
    onnxruntime==1.18.0 \
    protobuf==4.25 \
    pyarrow==18.1.0 \
    pyworld==0.3.4 \
    rich==13.7.1 \
    tensorboard==2.14.0 \
    transformers==4.51.3 \
    wetext==0.0.4 \
    wget==3.2 \
    x-transformers==2.11.24

# openai-whisper: instalar desde git evita el problema de pkg_resources
# en el build aislado de la versión PyPI 20231117
RUN pip install --no-cache-dir "openai-whisper @ git+https://github.com/openai/whisper.git"

# Instalar Matcha-TTS desde el submódulo de CosyVoice (versión parcheada sin lightning)
RUN pip install --no-cache-dir -e /cosyvoice/third_party/Matcha-TTS

# Fix diffusers compat: diffusers 0.25.x importa cached_download de huggingface_hub
# que fue removido en huggingface_hub>=0.17. Parchar el archivo directamente.
RUN sed -i \
    's/from huggingface_hub import cached_download, hf_hub_download, model_info/from huggingface_hub import hf_hub_download, model_info/g' \
    /usr/local/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py && \
    sed -i \
    's/resolved_module_file = cached_download(/resolved_module_file = hf_hub_download(/g' \
    /usr/local/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py && \
    echo "Patched diffusers dynamic_modules_utils.py OK"

# Agregar CosyVoice al path de Python
ENV PYTHONPATH=/cosyvoice

# Re-instalar setuptools<70 al final: setuptools>=70 no expone pkg_resources
# que CosyVoice necesita en runtime. setuptools==68.x mantiene el API legacy.
RUN pip install --no-cache-dir 'setuptools<70'

# ── 4. Dependencias propias del servicio FastAPI ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 5. Código del servicio ────────────────────────────────────────────────────
COPY main.py .

# El modelo se monta como volumen en /models/CosyVoice2-0.5B
ENV COSYVOICE_MODEL_DIR=/models/CosyVoice2-0.5B
ENV TTS_MAX_CHARS=500

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8010/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
