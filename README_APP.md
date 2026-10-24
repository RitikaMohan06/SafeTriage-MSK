# SafeTriage-MSK — deployable app

Pipeline (`main.py`) → `outputs/{best_model.pt,calibration.json,results.json}` → FastAPI (`app/server.py`) → web UI (Triage + Dashboard).

## 1. Train on FracAtlas (GPU recommended)
    pip install -r requirements.txt -r requirements-app.txt
    python main.py --dataset fracatlas --data_dir /path/to/FracAtlas --epochs 15 --workers 0
The folder must contain `dataset.csv` and the `images/Fractured` and `images/Non_fractured` folders. See `data/DOWNLOAD_INSTRUCTIONS.md`.

## 2. Run the app
    uvicorn app.server:app --port 8000      # open http://localhost:8000
No weights yet? `python tools/make_demo_results.py` fills the dashboard with SYNTHETIC placeholder data (clearly bannered). `/api/predict` returns 503 until `best_model.pt` exists.

## 3. Deploy
    docker build -t safetriage . && docker run -p 8000:8000 safetriage
(Run training first so `outputs/` holds the real files; the image copies it.) Works on Render, Railway, Fly.io, or any VM; CPU is fine for inference.

## API
`GET /api/health` · `GET /api/results` · `POST /api/predict` (form: `body_part`, `files`×1–3)

## What changed vs. the original zip
- Temperature T was learned but never applied → now applied to all probabilities.
- Added upper cutoff (`abnormal_threshold`) so the system can say "confident abnormal", not only "normal or defer".
- Per-body-part cutoffs are now actually evaluated (global vs per-part miss/defer rates).
- Single decision rule shared by evaluation and serving (`src/serving.py`).

## Honest caveats for the paper
- The FNR bound is a finite-sample/marginal guarantee per calibration group; small groups fall back to the global value when there are too few samples.
- FracAtlas has no patient IDs. Its random image-level split may leak correlated images across train and test. Research prototype, not a medical device.

## Dataset and body parts
The app's dropdown follows `calibration.json`; FracAtlas parts are HAND/LEG/HIP/SHOULDER. Humerus is an upper-arm bone and is not one of FracAtlas's labeled regions. Multi-region images are dropped, and small per-part calibration sets fall back to the global cutoff.

## Accuracy check
The bundled FracAtlas dataset does not meet a 90% accuracy target in this repo. The current project metrics in `outputs/results.json` report 82.35% accuracy and 0.908 AUC on the held-out test split. That is below 90% and should be treated as a research prototype, not a production medical classifier. To reach >90% you need a larger, cleaner dataset and a stronger training/evaluation setup (preferably a real MURA split with patient-level holdout and more epochs / better augmentation).
