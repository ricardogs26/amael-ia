"""
cosyvoice-service — FastAPI TTS wrapper para CosyVoice2-0.5B.

Endpoints:
  POST /tts                 — Síntesis estándar (voz por defecto)
  POST /tts/clone           — Síntesis con clonación de voz (zero-shot)
  GET  /health              — Estado del servicio

Corre en CPU (sin GPU). El modelo se carga una vez en el lifespan.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── torchaudio monkeypatch ────────────────────────────────────────────────────
# torchaudio 2.5 usa torchcodec por defecto (no disponible en CPU-only).
# Reemplazamos torchaudio.load con soundfile para evitar ImportError.
import torchaudio as _torchaudio

def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, backend=None):
    """Drop-in replacement for torchaudio.load usando soundfile."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    if frame_offset:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    tensor = torch.from_numpy(data.T)  # (channels, frames)
    if not channels_first:
        tensor = tensor.T
    return tensor, sr

_torchaudio.load = _sf_load
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cosyvoice-service")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR        = os.environ.get("COSYVOICE_MODEL_DIR", "/models/CosyVoice2-0.5B")
SAMPLE_RATE      = 22050   # placeholder; real rate from model.sample_rate
MAX_CHARS        = int(os.environ.get("TTS_MAX_CHARS", "500"))
# Voz por defecto: audio de referencia bundled en el repo de CosyVoice
# Voz de referencia: mujer mexicana (generado con Piper es_MX-claude-high)
# Fallback: audio bundled en el repo de CosyVoice (voz china)
DEFAULT_REF_WAV  = "/models/reference/es_MX_female.wav"
DEFAULT_REF_TEXT = "Hola, me da mucho gusto saludarte hoy. Espero que tengas un excelente día lleno de buenas noticias."
_FALLBACK_REF_WAV  = "/cosyvoice/asset/zero_shot_prompt.wav"
_FALLBACK_REF_TEXT = "希望你以后能做的比我还好呦。"

# ── Modelo global ─────────────────────────────────────────────────────────────
_model      = None
_ref_audio  = None   # numpy array a 16kHz para la voz por defecto


def _load_model():
    global _model, _ref_audio
    logger.info(f"[cosyvoice] Cargando modelo desde {MODEL_DIR} ...")
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import CosyVoice2
    _model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False)
    elapsed = time.monotonic() - t0
    logger.info(f"[cosyvoice] Modelo listo en {elapsed:.1f}s (CPU)")

    # Seleccionar audio de referencia (mexicana > fallback china)
    ref_path = DEFAULT_REF_WAV if os.path.exists(DEFAULT_REF_WAV) else _FALLBACK_REF_WAV
    ref_text = DEFAULT_REF_TEXT if os.path.exists(DEFAULT_REF_WAV) else _FALLBACK_REF_TEXT
    if os.path.exists(ref_path):
        audio_np, sr = sf.read(ref_path, dtype="float32")
        duration = len(audio_np) / sr
        _ref_audio = (ref_path, ref_text)
        logger.info(f"[cosyvoice] Voz de referencia: {ref_path} ({duration:.1f}s @ {sr}Hz)")
    else:
        logger.warning("[cosyvoice] No se encontró ningún audio de referencia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    logger.info("[cosyvoice] Servicio detenido.")


app = FastAPI(title="cosyvoice-service", version="1.0.0", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text:     str              = Field(..., max_length=MAX_CHARS)
    language: str              = Field(default="es", description="'es' o 'en'")
    speed:    float            = Field(default=1.0, ge=0.5, le=2.0)

class TTSCloneRequest(BaseModel):
    text:                   str   = Field(..., max_length=MAX_CHARS)
    reference_audio_base64: str   = Field(..., description="WAV base64 del hablante de referencia")
    prompt_text:            str   = Field(..., description="Transcripción del audio de referencia")
    language:               str   = Field(default="es")

class TTSResponse(BaseModel):
    audio_base64:     str
    format:           str   = "wav"
    sample_rate:      int   = SAMPLE_RATE
    duration_seconds: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _audio_to_base64(audio_array: np.ndarray, sample_rate: int) -> tuple[str, float]:
    """Convierte numpy array de audio a base64 WAV."""
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    duration = len(audio_array) / sample_rate
    return b64, duration


def _collect_chunks(generator) -> np.ndarray:
    """Junta todos los chunks del generador de CosyVoice en un solo array."""
    chunks = []
    for chunk in generator:
        pcm = chunk["tts_speech"]
        if hasattr(pcm, "numpy"):
            pcm = pcm.numpy()
        if pcm.ndim > 1:
            pcm = pcm.squeeze()
        chunks.append(pcm)
    if not chunks:
        raise RuntimeError("El modelo no generó audio.")
    return np.concatenate(chunks)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "loading",
        "model":  "CosyVoice2-0.5B",
        "device": "cpu",
    }


@app.post("/tts", response_model=TTSResponse)
def synthesize(req: TTSRequest):
    """Síntesis de voz usando zero-shot con la voz de referencia bundled."""
    if _model is None:
        raise HTTPException(503, "Modelo aún cargando")
    if _ref_audio is None:
        raise HTTPException(503, "Audio de referencia no disponible")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "El texto no puede estar vacío")

    lang_tag = "<|es|>" if req.language.startswith("es") else "<|en|>"
    ref_path, ref_text = _ref_audio

    try:
        t0 = time.monotonic()
        gen = _model.inference_zero_shot(
            f"{lang_tag}{text}",
            ref_text,
            ref_path,
            stream=True,
        )
        audio = _collect_chunks(gen)
        b64, duration = _audio_to_base64(audio, _model.sample_rate)
        elapsed = time.monotonic() - t0
        logger.info(
            f"[cosyvoice] TTS ok | {len(text)} chars | {duration:.1f}s audio | {elapsed:.1f}s latencia"
        )
        return TTSResponse(
            audio_base64=b64,
            sample_rate=_model.sample_rate,
            duration_seconds=round(duration, 2),
        )
    except Exception as exc:
        logger.error(f"[cosyvoice] Error TTS: {exc}")
        raise HTTPException(500, str(exc))


@app.post("/tts/clone", response_model=TTSResponse)
def synthesize_clone(req: TTSCloneRequest):
    """Síntesis con clonación zero-shot usando audio de referencia."""
    if _model is None:
        raise HTTPException(503, "Modelo aún cargando")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "El texto no puede estar vacío")

    try:
        # Decodificar audio de referencia a archivo temporal
        ref_bytes = base64.b64decode(req.reference_audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            tmp_path = tmp.name

        lang_tag = "<|es|>" if req.language.startswith("es") else "<|en|>"
        t0 = time.monotonic()
        gen = _model.inference_zero_shot(
            f"{lang_tag}{text}",
            req.prompt_text,
            tmp_path,
            stream=True,
        )
        audio = _collect_chunks(gen)
        b64, duration = _audio_to_base64(audio, _model.sample_rate)
        elapsed = time.monotonic() - t0
        logger.info(
            f"[cosyvoice] TTS-clone ok | {len(text)} chars | {duration:.1f}s audio | {elapsed:.1f}s latencia"
        )
        os.unlink(tmp_path)
        return TTSResponse(
            audio_base64=b64,
            sample_rate=_model.sample_rate,
            duration_seconds=round(duration, 2),
        )
    except Exception as exc:
        logger.error(f"[cosyvoice] Error TTS-clone: {exc}")
        raise HTTPException(500, str(exc))
