# SafeTriage-MSK

Body-part-aware fracture triage for musculoskeletal X-rays, with a FastAPI backend and browser UI.

The bundled dataset is **FracAtlas**, not MURA. FracAtlas labels the regions HAND, LEG, HIP, and SHOULDER. The dataset is not committed because its archive is large and has separate distribution terms; see [data/DOWNLOAD_INSTRUCTIONS.md](data/DOWNLOAD_INSTRUCTIONS.md).

## Run the app

Install dependencies and start the app:

```powershell
pip install -r requirements.txt -r requirements-app.txt
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The repository includes trained weights and calibration/results under `outputs/`.
## Screenshots
<img width="1200" height="927" alt="Screenshot 2026-10-24 143022" src="https://github.com/user-attachments/assets/e22944d4-988d-4bfe-a758-c4b1002a99ee" />

## Train on FracAtlas

After obtaining and extracting FracAtlas:

```powershell
python main.py --dataset fracatlas --data_dir "path\to\FracAtlas" --epochs 15 --workers 0
```

`--workers 0` avoids Windows multiprocessing issues. The pipeline writes model weights, calibration thresholds, and test metrics to `outputs/`.

## Current evaluation and limitations

The checked-in FracAtlas test results report **82.35% accuracy** and **0.908 AUC**. This does not meet a 90% accuracy target. FracAtlas has no patient identifiers, so the project uses a random image-level split and test metrics may be optimistic due to possible leakage. The app may defer uncertain predictions by design; this behavior does not establish diagnostic accuracy.

This is a research/education prototype, not a medical device, and must not be used to make clinical decisions. See [README_APP.md](README_APP.md) for API, deployment, and additional limitations.
