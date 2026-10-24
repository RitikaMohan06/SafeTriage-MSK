"""
demo_synthetic_results.py
==========================
Generates the SAME plots main.py will produce, but with realistic FAKE
data (no MURA download needed) so you can see exactly what your report
figures will look like before you run real training. Do not use these
numbers in your report — they are placeholders.
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.conformal import build_prediction_set, risk_controlled_threshold
from src.evaluate import classification_metrics, expected_calibration_error, coverage_by_group
from src.deferral import sweep_cost_ratio
from src.visualize import (
    plot_roc, plot_reliability_diagram, plot_bodypart_coverage, plot_cost_vs_k,
    plot_risk_coverage_curve,
)

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(7)

BODY_PARTS = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
N = 1200

# --- Simulate a plausible test set: harder body parts (shoulder, humerus)
# get noisier probabilities than easier ones (wrist, finger), matching what
# the MURA paper reported qualitatively. ---
labels = rng.binomial(1, 0.42, size=N)
groups = rng.choice(BODY_PARTS, size=N, p=[0.14, 0.13, 0.10, 0.17, 0.08, 0.20, 0.18])

difficulty = {
    "WRIST": 0.06, "FINGER": 0.08, "HAND": 0.10, "FOREARM": 0.09,
    "ELBOW": 0.12, "SHOULDER": 0.18, "HUMERUS": 0.20,
}
probs = np.zeros(N)
for i in range(N):
    noise = difficulty[groups[i]]
    base = 0.82 if labels[i] == 1 else 0.18
    probs[i] = np.clip(base + rng.normal(0, noise), 0.01, 0.99)

# --- Standard metrics ---
metrics = classification_metrics(probs, labels)
ece = expected_calibration_error(probs, labels)
print("Example metrics (SYNTHETIC, not real results):")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print(f"  ECE: {ece:.4f}")

# --- Conformal risk control at target FNR = 5% ---
# split off a fake "calibration" subset to fit tau, evaluate on the rest
calib_idx = rng.choice(N, size=400, replace=False)
test_idx = np.setdiff1d(np.arange(N), calib_idx)
tau = risk_controlled_threshold(probs[calib_idx], labels[calib_idx], target_fnr=0.05)
pred_sets = [build_prediction_set(p, tau, mode="risk_control") for p in probs[test_idx]]

coverage_df = coverage_by_group(pred_sets, labels[test_idx], groups[test_idx])
print("\nExample per-body-part coverage (SYNTHETIC):")
print(coverage_df)

sweep = sweep_cost_ratio(pred_sets, labels[test_idx])

# alphas sweep for risk-coverage curve
alphas = [0.20, 0.10, 0.05, 0.02, 0.01]
pred_sets_by_alpha = {}
for a in alphas:
    t = risk_controlled_threshold(probs[calib_idx], labels[calib_idx], target_fnr=a)
    pred_sets_by_alpha[a] = [build_prediction_set(p, t, mode="risk_control") for p in probs[test_idx]]

# --- Save all plots, prefixed demo_ so they're clearly not real results ---
plot_roc(probs[test_idx], labels[test_idx], out_path="outputs/demo_roc_curve.png")
plot_reliability_diagram(probs[test_idx], labels[test_idx], out_path="outputs/demo_reliability_diagram.png")
plot_bodypart_coverage(coverage_df, out_path="outputs/demo_bodypart_coverage.png")
plot_cost_vs_k(sweep, out_path="outputs/demo_cost_vs_k.png")
plot_risk_coverage_curve(pred_sets_by_alpha, labels[test_idx], alphas, out_path="outputs/demo_risk_coverage_curve.png")

coverage_df.to_csv("outputs/demo_bodypart_coverage_table.csv", index=False)
print("\nDemo plots written to outputs/demo_*.png")
