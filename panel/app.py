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
from pydantic import BaseModel

# Ensure panel module can import kws_engine
CURRENT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CURRENT_DIR))
from kws_engine import KWSEngine

app = FastAPI(title="Proxima KWS Developer Panel", version="1.0.0")

# Initialize KWS engine with models folder
MODELS_DIR = CURRENT_DIR.parent / "models"
engine = KWSEngine(models_dir=str(MODELS_DIR), default_model="proxima_v3_int8.tflite")

class ModelSelectRequest(BaseModel):
    model_name: str

class StreamPredictionRequest(BaseModel):
    samples: List[float]
    sample_rate: Optional[int] = 16000

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
        
    audio_arr = np.array(req.samples, dtype=np.float32)
    result = engine.process_audio_buffer(audio_arr, sample_rate=req.sample_rate)
    return result

@app.post("/api/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """Predict on uploaded or recorded WAV/audio file (WAV, MP3, OGG, FLAC, M4A)."""
    try:
        content = await file.read()
        result = engine.process_wav_bytes(content, filename=file.filename or "audio")
        result["filename"] = file.filename
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
async def predict_sample(sample_id: str = Form(...)):
    """Run inference on one of the reference test samples."""
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
