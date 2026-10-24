"""
conformal.py
============
The core novelty of the project. Three increasingly targeted versions:

1. Split conformal (baseline conformal): one global threshold, guarantees
   coverage ON AVERAGE across all studies.
2. Class-conditional conformal: separate thresholds for the normal and
   abnormal classes, so BOTH classes individually get the coverage
   guarantee (not just on average).
3. Conformal risk control: directly finds the threshold that bounds a
   chosen RISK function on the calibration set. We use it to bound the
   FALSE NEGATIVE RATE specifically, since missing an abnormality is the
   clinically expensive error.

All three are computed PER BODY PART as well as globally, so we can show
whether/how coverage varies by body part (the project's other novelty).
"""
import numpy as np


def nonconformity_score(prob_abnormal, true_label):
    """
    A simple, standard nonconformity score: how far the model's predicted
    probability of the TRUE class is from 1 (i.e., "how wrong was the
    model, from the point of view of the correct answer").
    Lower score = more conforming / more confident-and-correct.
    """
    p_true_class = prob_abnormal if true_label == 1 else (1 - prob_abnormal)
    return 1 - p_true_class


def split_conformal_threshold(cal_probs, cal_labels, alpha=0.10):
    """
    Standard split conformal: threshold = the (1-alpha) quantile of
    nonconformity scores on the calibration set (with the standard
    finite-sample correction). Guarantees marginal coverage of 1-alpha.
    """
    scores = np.array(
        [nonconformity_score(p, y) for p, y in zip(cal_probs, cal_labels)]
    )
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level, method="higher")


def class_conditional_thresholds(cal_probs, cal_labels, alpha=0.10):
    """
    Separate thresholds for label=0 and label=1 calibration subsets.
    Returns dict {0: threshold, 1: threshold}.
    This matters because if abnormal cases are rarer, a single global
    threshold is dominated by the majority (normal) class and under-covers
    the abnormal class — exactly the error we most want to avoid.
    """
    thresholds = {}
    cal_probs, cal_labels = np.array(cal_probs), np.array(cal_labels)
    for c in [0, 1]:
        mask = cal_labels == c
        thresholds[c] = split_conformal_threshold(
            cal_probs[mask], cal_labels[mask], alpha=alpha
        )
    return thresholds


def risk_controlled_threshold(cal_probs, cal_labels, target_fnr=0.05, tau_grid=None):
    """
    Conformal risk control for the FALSE NEGATIVE RATE specifically.

    Idea: sweep a decision threshold tau on predicted P(abnormal). Below
    tau -> predict "normal only" (no deferral). As tau increases, MORE
    studies get called "normal only" automatically, which is more
    convenient (less deferral) but riskier (more chance of missing a true
    abnormality). Risk is monotonically non-decreasing in tau, so we want
    the LARGEST tau (fewest deferred cases, most automation) such that the
    false negative rate on the calibration set still stays <= target_fnr,
    using the standard conformal risk control finite-sample correction so
    the guarantee holds on new data too, not just the calibration set.
    """
    cal_probs, cal_labels = np.array(cal_probs), np.array(cal_labels)
    if tau_grid is None:
        tau_grid = np.linspace(0.01, 0.99, 99)  # ascending

    best_tau = 0.0  # safest possible default: defer everything
    for tau in tau_grid:
        predicted_normal_only = cal_probs < tau
        false_negatives = np.sum(predicted_normal_only & (cal_labels == 1))
        n_abnormal = np.sum(cal_labels == 1)
        # finite-sample corrected risk estimate (upper confidence style bound)
        risk_hat = (false_negatives + 1) / (n_abnormal + 1) if n_abnormal > 0 else 0
        if risk_hat <= target_fnr:
            best_tau = tau  # keep going — looking for the LARGEST valid tau
        else:
            break  # risk only grows from here (tau_grid is ascending), stop
    return best_tau


def build_prediction_set(prob_abnormal, thresholds, mode="risk_control"):
    """
    Turns a single probability into a prediction set given calibrated
    thresholds. Returns one of: {"normal"}, {"abnormal"}, {"normal","abnormal"}
    (the last one = DEFER).
    """
    if mode == "risk_control":
        tau = thresholds  # single scalar tau from risk_controlled_threshold
        if prob_abnormal < tau:
            return {"normal"}
        else:
            return {"normal", "abnormal"}  # anything above tau gets deferred
    elif mode == "class_conditional":
        t0, t1 = thresholds[0], thresholds[1]
        in_normal_set = nonconformity_score(prob_abnormal, 0) <= t0
        in_abnormal_set = nonconformity_score(prob_abnormal, 1) <= t1
        pred_set = set()
        if in_normal_set:
            pred_set.add("normal")
        if in_abnormal_set:
            pred_set.add("abnormal")
        if len(pred_set) == 0:
            # Empty set: rare edge case, default to deferring (safest choice)
            pred_set = {"normal", "abnormal"}
        return pred_set
    else:
        raise ValueError(f"Unknown mode {mode}")


def per_bodypart_thresholds(cal_probs, cal_labels, cal_bodyparts, target_fnr=0.05):
    """
    Computes a separate risk-controlled threshold PER BODY PART. This is
    the project's headline analysis: does a single global threshold
    under-cover harder body parts (e.g. shoulder, humerus)? We compare
    global vs. per-body-part thresholds in evaluate.py / visualize.py.
    """
    cal_probs, cal_labels = np.array(cal_probs), np.array(cal_labels)
    cal_bodyparts = np.array(cal_bodyparts)
    thresholds = {}
    for bp in np.unique(cal_bodyparts):
        mask = cal_bodyparts == bp
        # Guard against tiny subgroups (e.g. humerus) with too few samples
        # for a stable quantile estimate — fall back to the global threshold.
        if mask.sum() < 30 or (cal_labels[mask] == 1).sum() < 20:  # need enough abnormal cases for an FNR bound
            thresholds[bp] = None  # caller should fall back to global tau
        else:
            thresholds[bp] = risk_controlled_threshold(
                cal_probs[mask], cal_labels[mask], target_fnr=target_fnr
            )
    return thresholds


def abnormal_threshold(cal_probs, cal_labels, target_fpr=0.10):
    """Upper cutoff: smallest tau_hi such that auto-calling 'abnormal' for
    p >= tau_hi keeps the false-positive rate on NORMAL studies <= target_fpr
    (same finite-sample correction as risk_controlled_threshold)."""
    cal_probs, cal_labels = np.array(cal_probs), np.array(cal_labels)
    neg = cal_probs[cal_labels == 0]
    if len(neg) == 0:
        return 1.01
    for tau in np.linspace(0.01, 0.99, 99):
        if ((neg >= tau).sum() + 1) / (len(neg) + 1) <= target_fpr:
            return float(tau)
    return 1.01  # never auto-call abnormal
