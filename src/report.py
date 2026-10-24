"""Builds outputs/results.json (numpy only) consumed by the dashboard."""
import json
import numpy as np
from src.serving import decide, to_pred_set
from src.deferral import sweep_cost_ratio, evaluate_deferral


def auc_roc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_points(p, y, n=40):
    pts = []
    for t in np.linspace(0, 1, n):
        pts.append([float(((p >= t) & (y == 0)).sum() / max((y == 0).sum(), 1)),
                    float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))])
    return pts


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); out = []; ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) | (i == bins - 1))
        if m.sum():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            out.append([conf, acc, int(m.sum())]); ece += m.mean() * abs(conf - acc)
    return out, float(ece)


def _stats(decs, y):
    decs = np.array(decs); n1 = max((y == 1).sum(), 1); n0 = max((y == 0).sum(), 1)
    auto = decs != "defer"
    return dict(
        n=int(len(y)),
        deferral_rate=float((~auto).mean()),
        fnr=float(((decs == "normal") & (y == 1)).sum() / n1),   # missed abnormal, auto-called normal
        fpr=float(((decs == "abnormal") & (y == 0)).sum() / n0),
        auto_accuracy=float(((decs == "abnormal") == (y == 1))[auto].mean()) if auto.any() else None,
    )


def build_results(probs, labels, groups, cal, target_fnr, synthetic=False, out="outputs/results.json"):
    """cal: dict(temperature, tau_lo, tau_hi, bodypart_tau_lo)."""
    probs, labels, groups = map(np.asarray, (probs, labels, groups))
    lo, hi, bp_lo = cal["tau_lo"], cal["tau_hi"], cal["bodypart_tau_lo"]
    d_global = [decide(p, lo, hi) for p in probs]
    d_part = [decide(p, bp_lo.get(g) if bp_lo.get(g) is not None else lo, hi) for p, g in zip(probs, groups)]
    rel, ece = reliability(probs, labels)
    parts = []
    for g in sorted(set(groups)):
        m = groups == g
        parts.append(dict(body_part=g,
                          global_tau=_stats(np.array(d_global)[m], labels[m]),
                          per_part_tau=_stats(np.array(d_part)[m], labels[m]),
                          tau_lo=bp_lo.get(g) if bp_lo.get(g) is not None else lo))
    pred = (probs >= 0.5).astype(int)
    res = dict(
        synthetic=synthetic, target_fnr=target_fnr, calibration=cal,
        overall=dict(accuracy=float((pred == labels).mean()), auc=auc_roc(probs, labels), ece=ece,
                     sensitivity=float(((pred == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1)),
                     specificity=float(((pred == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
                     global_tau=_stats(d_global, labels), per_part_tau=_stats(d_part, labels)),
        roc=roc_points(probs, labels), reliability=rel, body_parts=parts,
        cost_sweep=sweep_cost_ratio([to_pred_set(d) for d in d_global], labels.tolist()),
    )
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res
