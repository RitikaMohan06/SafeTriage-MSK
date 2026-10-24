"""
evaluate.py
===========
All the standard metrics the course's report structure asks for, plus the
calibration- and coverage-specific ones this project's novelty needs.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    confusion_matrix,
)


def classification_metrics(probs, labels, threshold=0.5):
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return dict(
        accuracy=(tp + tn) / len(labels),
        sensitivity=tp / max(tp + fn, 1),   # recall on abnormal class
        specificity=tn / max(tn + fp, 1),
        precision=precision_score(labels, preds, zero_division=0),
        f1=f1_score(labels, preds, zero_division=0),
        auc=roc_auc_score(labels, probs),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    )


def expected_calibration_error(probs, labels, n_bins=10):
    """
    ECE: bins predictions by confidence, and measures the gap between
    average confidence and actual accuracy in each bin. Lower = better
    calibrated. This is the number that justifies (or not) the temperature
    scaling step.
    """
    probs, labels = np.array(probs), np.array(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean() if labels[mask].mean() >= 0 else 0
        # here "accuracy" of the abnormal-probability = fraction that were
        # actually abnormal in that confidence bin
        ece += (mask.sum() / len(probs)) * abs(bin_conf - bin_acc)
    return ece


def coverage_by_group(pred_sets, true_labels, groups):
    """
    For each group (e.g. body part), compute:
      - empirical coverage: fraction of studies whose TRUE label is inside
        the predicted set (should be >= 1-alpha if calibration worked)
      - average set size (efficiency: smaller sets = more useful)
      - deferral rate within that group
    This is the table/plot that shows whether a global threshold
    under-covers specific body parts.
    """
    import pandas as pd
    df = pd.DataFrame({"group": groups, "label": true_labels})
    df["label_str"] = df["label"].map({0: "normal", 1: "abnormal"})
    df["pred_set"] = pred_sets
    df["covered"] = df.apply(lambda r: r["label_str"] in r["pred_set"], axis=1)
    df["set_size"] = df["pred_set"].apply(len)
    df["deferred"] = df["set_size"] == 2

    summary = df.groupby("group").agg(
        n=("label", "count"),
        coverage=("covered", "mean"),
        avg_set_size=("set_size", "mean"),
        deferral_rate=("deferred", "mean"),
    )
    return summary.reset_index()
