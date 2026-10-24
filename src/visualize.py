"""
visualize.py
============
Every plot the report/presentation needs. Each function saves a PNG to
`outputs/` and also returns the matplotlib figure in case you want to
tweak it interactively in a notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set_theme(style="whitegrid")


def plot_roc(probs, labels, out_path="outputs/roc_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"DenseNet121 (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — Study-Level Abnormality Detection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_reliability_diagram(probs, labels, n_bins=10, out_path="outputs/reliability_diagram.png"):
    probs, labels = np.array(probs), np.array(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_acc, bin_conf = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_acc.append(labels[mask].mean())
        bin_conf.append(probs[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_centers, bin_acc, width=0.08, alpha=0.7, label="Observed frequency", edgecolor="black")
    ax.plot([0, 1], [0, 1], "--", color="red", label="Perfect calibration")
    ax.set_xlabel("Predicted P(abnormal)")
    ax.set_ylabel("Observed fraction abnormal")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_risk_coverage_curve(pred_sets_by_alpha, true_labels, alphas, out_path="outputs/risk_coverage_curve.png"):
    """
    pred_sets_by_alpha: dict {alpha_value: list_of_pred_sets} — i.e. you've
    already run conformal at several alpha/target_fnr levels.
    Plots: as we allow more deferral (looser alpha -> stricter tau ->
    more deferred), how does the false negative rate among AUTO-DECIDED
    cases drop?
    """
    deferral_rates, fnrs = [], []
    for a in alphas:
        pred_sets = pred_sets_by_alpha[a]
        n_auto = 0
        fn = 0
        n_abnormal_auto = 0
        for pset, y in zip(pred_sets, true_labels):
            if len(pset) == 2:
                continue
            n_auto += 1
            pred = 1 if "abnormal" in pset else 0
            if y == 1:
                n_abnormal_auto += 1
                if pred == 0:
                    fn += 1
        deferral_rates.append(1 - n_auto / len(true_labels))
        fnrs.append(fn / max(n_abnormal_auto, 1))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(deferral_rates, fnrs, marker="o")
    ax.set_xlabel("Deferral rate (fraction sent to radiologist)")
    ax.set_ylabel("False negative rate among auto-decided cases")
    ax.set_title("Risk–Coverage Trade-off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_bodypart_coverage(coverage_df, out_path="outputs/bodypart_coverage.png"):
    """coverage_df: output of evaluate.coverage_by_group()"""
    fig, ax = plt.subplots(figsize=(7, 5))
    order = coverage_df.sort_values("coverage")["group"]
    sns.barplot(data=coverage_df, x="group", y="coverage", order=order, ax=ax, color="#4C72B0")
    ax.axhline(0.90, ls="--", color="red", label="Target coverage (1-alpha)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_xlabel("Body part")
    ax.set_title("Coverage by Body Part (global threshold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_cost_vs_k(sweep_results, out_path="outputs/cost_vs_k.png"):
    """sweep_results: output of deferral.sweep_cost_ratio()"""
    ks = [r["k"] for r in sweep_results]
    costs = [r["expected_cost"] for r in sweep_results]
    defer_rates = [r["deferral_rate"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(ks, costs, marker="o", color="#C44E52", label="Expected cost")
    ax1.set_xlabel("Cost ratio k (false negative vs. false positive)")
    ax1.set_ylabel("Expected cost per study", color="#C44E52")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(ks, defer_rates, marker="s", color="#4C72B0", label="Deferral rate")
    ax2.set_ylabel("Deferral rate", color="#4C72B0")

    fig.suptitle("Cost-Sensitive Deferral: Expected Cost & Deferral Rate vs. k")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_gradcam_examples(images, cams, titles, out_path="outputs/gradcam_examples.png"):
    """images: list of HxWx3 numpy arrays (0-1 range); cams: matching heatmaps."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(titles[i], fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(images[i])
        axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
    fig.suptitle("Confident vs. Deferred Cases — Where the Model Looks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig
