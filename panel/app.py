import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import psutil
from pydantic import BaseModel

PROCESS = psutil.Process()

# Ensure panel module can import kws_engine
CURRENT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CURRENT_DIR))
from kws_engine import KWSEngine

import asyncio
import logging
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Detect port from sys.argv or default to 8000
    port = 8000
    for idx, arg in enumerate(sys.argv):
        if arg == "--port" and idx + 1 < len(sys.argv):
            try:
                port = int(sys.argv[idx + 1])
            except ValueError:
                pass
        elif arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    logger = logging.getLogger("uvicorn.error")
    logger.info(f"Open Developer Panel in Browser: http://localhost:{port}/ (or http://127.0.0.1:{port}/)")

    async def _print_banner():
        await asyncio.sleep(0.08)
        print("\n" + "=" * 60, flush=True)
        print("  PROXIMA KWS DEVELOPER PANEL IS READY!", flush=True)
        print(f"  --> Local Access:   http://localhost:{port}/", flush=True)
        print(f"  --> Loopback IP:    http://127.0.0.1:{port}/", flush=True)
        print("=" * 60 + "\n", flush=True)

    asyncio.create_task(_print_banner())
    yield

app = FastAPI(title="Proxima KWS Developer Panel", version="1.0.0", lifespan=lifespan)

# Initialize KWS engine with models folder
MODELS_DIR = CURRENT_DIR.parent / "models"
engine = KWSEngine(models_dir=str(MODELS_DIR), default_model="proxima_v3_float32.tflite")

class ModelSelectRequest(BaseModel):
    model_name: str

class StreamPredictionRequest(BaseModel):
    samples: List[float]
    sample_rate: Optional[int] = 16000
    enable_energy_gate: Optional[bool] = True
    energy_threshold_db: Optional[float] = -42.0
    model_name: Optional[str] = None

@app.get("/api/models")
async def get_models():
    """List all available models in models/."""
    models = engine.list_available_models()
    return {
        "active_model": engine.active_model_name,
        "models": models
    }

@app.post("/api/models/select")
async def select_model(req: ModelSelectRequest):
    """Switch active KWS model at runtime."""
    try:
        success = engine.load_model(req.model_name)
        return {
            "status": "success",
            "active_model": engine.active_model_name,
            "model_info": engine.model_info,
            "architecture": engine.architecture_graph
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/model/info")
async def get_model_info():
    """Retrieve metadata and hardware constraints of current model."""
    return {
        "active_model": engine.active_model_name,
        "info": engine.model_info,
        "architecture": engine.architecture_graph
    }

@app.post("/api/predict/stream")
async def predict_stream(req: StreamPredictionRequest):
    """Real-time live listening endpoint taking raw Float32 audio samples."""
    if not req.samples:
        raise HTTPException(status_code=400, detail="Empty audio samples")
        
    if req.model_name and req.model_name != engine.active_model_name:
        engine.load_model(req.model_name)

    audio_arr = np.array(req.samples, dtype=np.float32)
    result = engine.process_audio_buffer(
        audio_arr,
        sample_rate=req.sample_rate or 16000,
        enable_energy_gate=req.enable_energy_gate if req.enable_energy_gate is not None else True,
        energy_threshold_db=req.energy_threshold_db if req.energy_threshold_db is not None else -42.0
    )
    result["host_cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
    return result

@app.post("/api/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None)
):
    """Predict on uploaded or recorded WAV/audio file (WAV, MP3, OGG, FLAC, M4A)."""
    try:
        if model_name and model_name != engine.active_model_name:
            engine.load_model(model_name)
        content = await file.read()
        result = engine.process_wav_bytes(content, filename=file.filename or "audio")
        result["filename"] = file.filename
        result["host_cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding error: {str(e)}")

@app.get("/api/test_samples")
async def get_test_samples():
    """Provide verified test recordings from PROXIMA_DATA_1_SPLIT/test for instant testing."""
    test_dir = CURRENT_DIR.parent / "PROXIMA_DATA_1_SPLIT" / "test"
    samples = []
    
    if test_dir.exists():
        proxima_files = sorted(list(test_dir.rglob("proxima/*.wav")))[:6]
        unknown_files = sorted(list(test_dir.rglob("unknown/*.wav")))[:6]
        
        for p in proxima_files:
            rel_to_test = p.relative_to(test_dir).as_posix()
            spk_name = p.parent.parent.name
            samples.append({
                "id": rel_to_test,
                "name": f"[{spk_name}] {p.name}",
                "label": "PROXIMA (Keyword Sample)",
                "expected": "PROXIMA",
                "rel_path": str(p.relative_to(CURRENT_DIR.parent)).replace("\\", "/")
            })
        for u in unknown_files:
            rel_to_test = u.relative_to(test_dir).as_posix()
            spk_name = u.parent.parent.name
            samples.append({
                "id": rel_to_test,
                "name": f"[{spk_name}] {u.name}",
                "label": "UNKNOWN (Negative Sample)",
                "expected": "UNKNOWN",
                "rel_path": str(u.relative_to(CURRENT_DIR.parent)).replace("\\", "/")
            })
            
    return {"samples": samples}

@app.post("/api/predict/sample")
async def predict_sample(
    sample_id: str = Form(...),
    model_name: Optional[str] = Form(None)
):
    """Run inference on one of the reference test samples."""
    if model_name and model_name != engine.active_model_name:
        engine.load_model(model_name)
    file_path = CURRENT_DIR.parent / "PROXIMA_DATA_1_SPLIT" / "test" / sample_id
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Test sample file not found: {file_path}")
        
    with open(file_path, "rb") as f:
        content = f.read()
    result = engine.process_wav_bytes(content)
    result["sample_id"] = sample_id
    result["filename"] = file_path.name
    return result

# Mount static web UI assets
static_dir = CURRENT_DIR / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
