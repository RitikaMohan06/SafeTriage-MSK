"""
deferral.py
===========
Turns prediction sets into a deferral decision and computes the
cost/workload trade-off.

A study is DEFERRED (sent to a radiologist) if its prediction set has
2 elements ({normal, abnormal}) i.e. the model is not confident enough
under the chosen risk bound. Otherwise the model's single-element
prediction is used automatically.

Cost model: cost(false negative) = k * cost(false positive), plus a
fixed cost per deferral (radiologist time). We sweep k because the
"right" k depends on clinical context, and a Q1 paper should show the
whole trade-off curve, not pick one k and hide the rest.
"""
import numpy as np


def evaluate_deferral(pred_sets, true_labels, k=10, defer_cost=1.0):
    """
    pred_sets: list of sets, e.g. [{"normal"}, {"normal","abnormal"}, ...]
    true_labels: list of 0/1
    k: how many times worse a false negative is than a false positive
    defer_cost: cost of sending a case to the radiologist (in the same
                units as a false positive, e.g. "1 unit of review effort")

    Returns a dict of: deferral_rate, sensitivity, specificity,
    expected_cost, n_auto, n_deferred
    """
    n = len(true_labels)
    n_deferred = 0
    tp = fp = tn = fn = 0
    total_cost = 0.0

    for pred_set, y in zip(pred_sets, true_labels):
        if len(pred_set) == 2:
            n_deferred += 1
            total_cost += defer_cost
            continue
        pred = 1 if "abnormal" in pred_set else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
            total_cost += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 0 and y == 1:
            fn += 1
            total_cost += k  # missed abnormality is k times as costly

    deferral_rate = n_deferred / n
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return dict(
        deferral_rate=deferral_rate,
        sensitivity=sensitivity,
        specificity=specificity,
        expected_cost=total_cost / n,
        n_auto=n - n_deferred,
        n_deferred=n_deferred,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def sweep_cost_ratio(pred_sets, true_labels, k_values=(1, 2, 5, 10, 20, 50)):
    """Runs evaluate_deferral across several k values — the data behind
    the 'accuracy vs. deferral rate' and 'expected cost vs. k' plots."""
    results = []
    for k in k_values:
        r = evaluate_deferral(pred_sets, true_labels, k=k)
        r["k"] = k
        results.append(r)
    return results
