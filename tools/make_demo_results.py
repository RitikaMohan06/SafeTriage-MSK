"""Writes SYNTHETIC outputs/results.json + calibration.json so the dashboard
runs before real training. Flagged synthetic=true; never cite these numbers."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.conformal import risk_controlled_threshold, abnormal_threshold, per_bodypart_thresholds
from src.report import build_results

rng = np.random.default_rng(7)
BP = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
diff = dict(WRIST=.06, FINGER=.08, HAND=.10, FOREARM=.09, ELBOW=.12, SHOULDER=.18, HUMERUS=.20)
N = 3000
y = rng.binomial(1, .42, N); g = rng.choice(BP, N, p=[.14, .13, .10, .17, .08, .20, .18])
p = np.clip(np.where(y == 1, .82, .18) + rng.normal(0, 1, N) * np.array([diff[x] for x in g]), .01, .99)
c = rng.random(N) < .5
lo = risk_controlled_threshold(p[c], y[c], .05); hi = abnormal_threshold(p[c], y[c], .10)
bp = {k: v for k, v in per_bodypart_thresholds(p[c], y[c], g[c], .05).items()}
bp = {k: (None if v is None else float(v)) for k, v in bp.items()}
cal = dict(temperature=1.0, tau_lo=float(lo), tau_hi=float(hi), bodypart_tau_lo=bp, target_fnr=.05)
os.makedirs("outputs", exist_ok=True)
json.dump(cal, open("outputs/calibration.json", "w"), indent=1)
r = build_results(p[~c], y[~c], g[~c], cal, .05, synthetic=True)
print("demo written; overall:", {k: round(v, 3) for k, v in r["overall"].items() if isinstance(v, float)})
