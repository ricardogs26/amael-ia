import os
import io
import torch
import torchaudio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# CosyVoice se importa desde /opt/CosyVoice (ver PYTHONPATH en Dockerfile)
from cosyvoice.cli.cosyvoice import CosyVoice

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_DIR = "/app/pretrained_models/CosyVoice-300M"
SAMPLE_RATE = 22050  # CosyVoice-300M usa 22050 Hz

cosyvoice: CosyVoice | None = None


# ── Carga del modelo ───────────────────────────────────────────────────────────
def load_model():
    """
    Carga CosyVoice en CPU.

    Parámetros correctos para la versión actual del repo:
      - load_jit=False  → sin TorchScript (necesita compilar, falla en CPU cold-start)
      - load_trt=False  → sin TensorRT   (solo GPU con TRT instalado)
      - fp16=False      → fp16 no está soportado en CPU, causaría RuntimeError
    """
    global cosyvoice
    if cosyvoice is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CosyVoice] Iniciando en dispositivo: {device}")

    try:
        cosyvoice = CosyVoice(
            MODEL_DIR,
            load_jit=False,   # ← Sin JIT (más lento pero estable en CPU)
            load_trt=False,   # ← Sin TensorRT (fue load_onnx en versiones viejas)
            fp16=False,       # ← OBLIGATORIO False en CPU
        )
        print("[CosyVoice] Modelo cargado exitosamente.")
    except Exception as e:
        print(f"[CosyVoice] Error crítico cargando modelo: {e}")
        raise


# ── Lifespan (reemplaza el deprecado @app.on_event) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Cleanup si fuera necesario


app = FastAPI(title="Amael TTS Service - CosyVoice", lifespan=lifespan)


# ── Schemas ────────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    speaker: str = "中文女"  # Speaker SFT por defecto


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Liveness/readiness probe para Kubernetes."""
    return {
        "status": "ok",
        "model_loaded": cosyvoice is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/synthesize")
def synthesize_speech(request: TTSRequest):
    """
    Convierte texto a audio WAV usando CosyVoice SFT.

    inference_sft() devuelve un generador de dicts con la forma:
        {"tts_speech": Tensor(1, N_samples), "sample_rate": int}
    """
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    try:
        # Obtener el primer chunk de audio generado
        output = next(cosyvoice.inference_sft(request.text, request.speaker))

        # El tensor viene dentro del dict: output["tts_speech"]
        # shape: (1, N_samples) — ya en CPU
        audio_tensor = output["tts_speech"].cpu()

        # Serializar a WAV en memoria
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, SAMPLE_RATE, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")

    except StopIteration:
        raise HTTPException(status_code=500, detail="El modelo no generó audio")
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Formato de salida inesperado del modelo: {e}",
        )
    except Exception as e:
        print(f"[CosyVoice] Error generando audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)