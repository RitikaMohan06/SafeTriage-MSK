"""FastAPI backend. Run: uvicorn app.server:app --host 0.0.0.0 --port 8000"""
import io, json, os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from app.inference import Engine

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CALIB = os.environ.get("CALIB", str(OUTPUTS_DIR / "calibration.json"))
RESULTS = os.environ.get("RESULTS", str(OUTPUTS_DIR / "results.json"))
FRONTEND_DIR = BASE_DIR / "frontend"

PART_ALIASES = {
    "ELBOW": "ELBOW",
    "FINGER": "FINGER",
    "FOREARM": "FOREARM",
    "HAND": "HAND",
    "HUMERUS": "HUMERUS",
    "HIP": "HIP",
    "LEG": "LEG",
    "LOWER_LIMB": "LEG",
    "SHOULDER": "SHOULDER",
    "WRIST": "WRIST",
    "UPPER_ARM": "HUMERUS",
    "ARM": "HUMERUS",
}


def _normalise_body_part(value):
    if value is None:
        return None
    cleaned = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    cleaned = cleaned.replace("_", "")
    alias = PART_ALIASES.get(cleaned)
    if alias:
        return alias
    for key, canonical in PART_ALIASES.items():
        if cleaned in key:
            return canonical
    return cleaned


def _parts():
    preferred = [
        "HAND", "HIP", "LEG", "SHOULDER", "ELBOW", "FINGER", "FOREARM", "HUMERUS", "WRIST"
    ]
    try:
        path = Path(CALIB).resolve()
        if path.exists():
            body_parts = json.load(open(path, encoding="utf-8"))['bodypart_tau_lo']
            if isinstance(body_parts, dict):
                items = [_normalise_body_part(k) for k in body_parts.keys() if k]
                valid = sorted({p for p in items if p})
                if valid:
                    return valid
    except Exception:
        pass
    return preferred


app = FastAPI(title="SafeTriage-MSK")
cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
engine = Engine()


@app.get("/api/health")
def health():
    return dict(model_ready=engine.ready, detail=engine.error, results=os.path.exists(RESULTS), parts=_parts())


@app.get("/api/parts")
def parts():
    return _parts()


@app.get("/api/results")
def results():
    if not os.path.exists(RESULTS):
        raise HTTPException(404, "results.json missing: run main.py or tools/make_demo_results.py")
    return JSONResponse(json.load(open(RESULTS)))


@app.post("/api/predict")
async def predict(body_part: str = Form(...), files: list[UploadFile] = File(...)):
    valid_parts = _parts()
    if not engine.ready:
        raise HTTPException(503, f"Model not loaded ({engine.error}). Train first with main.py.")
    body_part = _normalise_body_part(body_part)
    if body_part not in valid_parts:
        raise HTTPException(400, f"body_part must be one of {valid_parts}")
    if not 1 <= len(files) <= 3:
        raise HTTPException(400, "Upload 1-3 views of the same study.")
    try:
        imgs = [Image.open(io.BytesIO(await f.read())).convert("RGB") for f in files]
    except Exception:
        raise HTTPException(400, "Could not read one of the images.")
    return engine.predict(imgs, body_part)


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api-config.js")
def api_config():
    return FileResponse(
        str(FRONTEND_DIR / "api-config.js"),
        media_type="application/javascript",
    )


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
