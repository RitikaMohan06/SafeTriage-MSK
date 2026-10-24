"""Shared decision rule used by training/evaluation AND the web backend."""
import json
import numpy as np

NORMAL, ABNORMAL, DEFER = "normal", "abnormal", "defer"


def decide(p, lo, hi):
    """p<lo -> confident normal; p>=hi -> confident abnormal; else defer."""
    if p < lo:
        return NORMAL
    if p >= hi:
        return ABNORMAL
    return DEFER


def to_pred_set(decision):
    return {"normal"} if decision == NORMAL else {"abnormal"} if decision == ABNORMAL else {"normal", "abnormal"}


class Calibration:
    """Loads outputs/calibration.json (written by main.py)."""

    def __init__(self, path):
        with open(path) as f:
            c = json.load(f)
        self.T = c["temperature"]
        self.lo = c["tau_lo"]
        self.hi = c["tau_hi"]
        self.bp_lo = {k: v for k, v in c["bodypart_tau_lo"].items() if v is not None}
        self.meta = c

    def thresholds(self, body_part):
        return self.bp_lo.get(body_part, self.lo), self.hi

    def probability(self, logit):
        return float(1 / (1 + np.exp(-logit / self.T)))
