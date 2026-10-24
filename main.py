"""
Full SafeTriage-MSK pipeline: train -> temperature scaling -> conformal
calibration (lower + upper cutoffs, global + per body part) -> test evaluation.
Writes outputs/best_model.pt, calibration.json, results.json (used by the web app).

    python main.py --data_dir /path/to/MURA-v1.1 --epochs 10
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import parse_mura_csv, patient_level_split, MuraStudyDataset
from src.model import StudyLevelDenseNet
from src.train import train_model
from src.calibration import TemperatureScaler
from src.conformal import risk_controlled_threshold, per_bodypart_thresholds, abnormal_threshold
from src.report import build_results
from src.fracatlas import parse_fracatlas, random_test_split


def run_model(model, dataset, device, workers):
    """Returns RAW study logits, labels, body parts."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=workers)
    logits, labels, groups = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, y, bp, _ in loader:
            out, _ = model(imgs.to(device))
            logits.extend(out.cpu().tolist()); labels.extend(y.tolist()); groups.extend(bp)
    return np.array(logits), np.array(labels), np.array(groups)


def main(a):
    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.dataset == "fracatlas":
        full, test_df = random_test_split(parse_fracatlas(a.data_dir))
    else:
        full = parse_mura_csv(os.path.join(a.data_dir, "train_image_paths.csv"), a.data_dir)
        test_df = parse_mura_csv(os.path.join(a.data_dir, "valid_image_paths.csv"), a.data_dir)  # locked test set
    train_df, tune_df, calib_df = patient_level_split(full)
    print(f"train={len(train_df)} tune={len(tune_df)} calib={len(calib_df)} test={len(test_df)} studies")

    model = train_model(StudyLevelDenseNet(pretrained=True), train_df, tune_df, MuraStudyDataset,
                        epochs=a.epochs, device=device, out_path="outputs/best_model.pt")

    # Temperature scaling on the tune split
    tl, ty, _ = run_model(model, MuraStudyDataset(tune_df, train=False), device, a.workers)
    T = TemperatureScaler().fit(torch.tensor(tl), torch.tensor(ty))
    print(f"Temperature: {T:.3f}")
    sig = lambda z: 1 / (1 + np.exp(-z / T))

    # Conformal calibration (calibration split only)
    cl, cy, cg = run_model(model, MuraStudyDataset(calib_df, train=False), device, a.workers)
    cp = sig(cl)
    lo = float(risk_controlled_threshold(cp, cy, a.target_fnr))
    hi = float(abnormal_threshold(cp, cy, a.target_fpr))
    bp = {k: (None if v is None else float(v)) for k, v in per_bodypart_thresholds(cp, cy, cg, a.target_fnr).items()}
    cal = dict(temperature=float(T), tau_lo=lo, tau_hi=hi, bodypart_tau_lo=bp,
               target_fnr=a.target_fnr, target_fpr=a.target_fpr)
    json.dump(cal, open("outputs/calibration.json", "w"), indent=1)
    print("tau_lo", lo, "tau_hi", hi, "per-part", bp)

    # Locked test evaluation
    tl, ty, tg = run_model(model, MuraStudyDataset(test_df, train=False), device, a.workers)
    res = build_results(sig(tl), ty, tg, cal, a.target_fnr, synthetic=False)
    print(json.dumps(res["overall"], indent=1))
    print("Done. Start the app: uvicorn app.server:app --port 8000")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset", choices=["mura", "fracatlas"], default="mura")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--target_fnr", type=float, default=0.05)
    p.add_argument("--target_fpr", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=2)
    main(p.parse_args())
