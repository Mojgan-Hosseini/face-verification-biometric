"""
biometric.py — Proper biometric verification metrics.

All metrics operate on raw similarity scores (higher = more similar),
not on binary predictions. This is critical: accuracy and F1 are
meaningless in biometrics because they depend on an arbitrary threshold.

The evaluation protocol:
  - Genuine pairs  (label=1): same identity, should have HIGH score
  - Impostor pairs (label=0): different identity, should have LOW score

From these score distributions we derive:
  - FAR(t): fraction of impostor pairs with score >= t  (false accepts)
  - FRR(t): fraction of genuine pairs with score <  t  (false rejects)
  - EER:    threshold t* where FAR(t*) ≈ FRR(t*)
  - TAR@FAR=k: TAR = 1 - FRR at the threshold where FAR = k
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Dict
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers and scripts
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Core rate computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep thresholds over the score range and compute FAR and FRR at each.

    Args:
        scores:         1-D array of similarity scores
        labels:         1-D array of {0, 1} labels (1 = genuine)
        num_thresholds: number of threshold points to sample

    Returns:
        thresholds, far_array, frr_array  — all length num_thresholds
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    genuine_scores  = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # Extend slightly above max so FRR reaches 1.0 at the last threshold
    thresholds = np.linspace(scores.min(), scores.max() + 1e-9, num_thresholds)

    # FAR: fraction of impostors accepted (score >= threshold)
    far = np.array(
        [np.mean(impostor_scores >= t) for t in thresholds], dtype=np.float64
    )
    # FRR: fraction of genuines rejected (score < threshold)
    frr = np.array(
        [np.mean(genuine_scores < t) for t in thresholds], dtype=np.float64
    )

    return thresholds, far, frr


# ─────────────────────────────────────────────────────────────────────────────
# EER
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 500,
) -> Tuple[float, float]:
    """
    Equal Error Rate: the operating point where FAR == FRR.

    We interpolate linearly between the two threshold samples that straddle
    the crossover, which is more stable than taking the nearest sample.

    Returns:
        (eer_value, eer_threshold)  — EER as a fraction in [0, 1]
    """
    thresholds, far, frr = compute_rates(scores, labels, num_thresholds)

    # diff changes sign at the crossover
    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # Edge case: curves don't cross (degenerate score distribution)
        idx = np.argmin(np.abs(diff))
        return float((far[idx] + frr[idx]) / 2), float(thresholds[idx])

    idx = sign_changes[0]

    # Linear interpolation between idx and idx+1
    t0, t1   = thresholds[idx], thresholds[idx + 1]
    d0, d1   = diff[idx],       diff[idx + 1]
    frac     = -d0 / (d1 - d0 + 1e-12)
    t_eer    = t0 + frac * (t1 - t0)
    far_eer  = far[idx]  + frac * (far[idx + 1]  - far[idx])
    frr_eer  = frr[idx]  + frac * (frr[idx + 1]  - frr[idx])
    eer      = (far_eer + frr_eer) / 2

    return float(np.clip(eer, 0.0, 1.0)), float(t_eer)


# ─────────────────────────────────────────────────────────────────────────────
# TAR @ fixed FAR
# ─────────────────────────────────────────────────────────────────────────────

def compute_tar_at_far(
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.001,
    num_thresholds: int = 500,
) -> float:
    """
    True Accept Rate at a fixed False Accept Rate.

    TAR@FAR=0.1% is the standard operational metric for high-security systems
    (e.g., border control) where you can only tolerate 1 impostor per 1000.

    Returns:
        TAR (= 1 - FRR) as a fraction in [0, 1]
    """
    thresholds, far, frr = compute_rates(scores, labels, num_thresholds)

    # Interpolate TAR as a function of FAR
    # FAR is monotonically decreasing with threshold, so we reverse for interp
    far_rev = far[::-1]
    tar_rev = (1 - frr)[::-1]

    if target_far <= far_rev[0]:
        return 0.0
    if target_far >= far_rev[-1]:
        return 1.0

    interp = interp1d(far_rev, tar_rev, kind="linear", fill_value="extrapolate")
    result = float(interp(target_far))
    if np.isnan(result):
        return 0.0
    return float(np.clip(result, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# ROC curve + AUC
# ─────────────────────────────────────────────────────────────────────────────

def compute_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    ROC curve (FAR on x-axis, TAR on y-axis) and AUC.

    Note: in biometrics the ROC x-axis is FAR, not FPR — same quantity,
    different terminology. TAR = TPR = 1 - FRR.

    Returns:
        (far_array, tar_array, auc)
    """
    thresholds, far, frr = compute_rates(scores, labels, num_thresholds)
    tar = 1 - frr

    # Sort by FAR ascending for AUC (trapezoidal integration)
    order = np.argsort(far)
    far_s  = far[order]
    tar_s  = tar[order]
    auc    = float(np.trapz(tar_s, far_s))

    return far_s, tar_s, auc


# ─────────────────────────────────────────────────────────────────────────────
# Full metric summary
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    far_thresholds: list[float] | None = None,
    num_thresholds: int = 500,
) -> Dict:
    """
    Compute the full biometric metric suite for a score-label array.

    Returns a dict with:
        eer, eer_threshold,
        tar_at_far (dict keyed by FAR value),
        auc,
        roc: {far, tar}
    """
    if far_thresholds is None:
        far_thresholds = [0.001, 0.0001, 0.00001]

    eer, eer_thresh  = compute_eer(scores, labels, num_thresholds)
    far_arr, tar_arr, auc = compute_roc(scores, labels, num_thresholds)

    tar_at_far = {
        t: compute_tar_at_far(scores, labels, target_far=t, num_thresholds=num_thresholds)
        for t in far_thresholds
    }

    return {
        "eer":            eer,
        "eer_threshold":  eer_thresh,
        "auc":            auc,
        "tar_at_far":     tar_at_far,
        "roc":            {"far": far_arr, "tar": tar_arr},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Dict],   # {"ModelName": evaluate() output, ...}
    title: str = "ROC Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay ROC curves for multiple models on a single axes.
    X-axis is FAR (log scale), Y-axis is TAR — standard biometric ROC.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=len(results))

    for (name, res), color in zip(results.items(), palette):
        far = res["roc"]["far"]
        tar = res["roc"]["tar"]
        auc = res["auc"]
        eer = res["eer"]
        ax.plot(far, tar, label=f"{name}  (AUC={auc:.4f}, EER={eer*100:.2f}%)",
                color=color, linewidth=2)
        ax.axvline(eer, color=color, linestyle="--", alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlim([1e-4, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Accept Rate (FAR)", fontsize=13)
    ax.set_ylabel("True Accept Rate (TAR)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_robustness_curves(
    data: Dict,   # {model: {deg_type: {level: eer}}}
    save_dir: str = "plots",
) -> None:
    """
    One figure per degradation type.  Each line = one model.
    X-axis = degradation intensity level, Y-axis = EER (%).
    Higher on y-axis = worse performance.
    """
    import os, pathlib
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Collect all degradation types from first model
    first_model = next(iter(data.values()))
    deg_types   = list(first_model.keys())
    palette     = sns.color_palette("tab10", n_colors=len(data))

    for deg in deg_types:
        fig, ax = plt.subplots(figsize=(8, 5))

        for (model_name, deg_data), color in zip(data.items(), palette):
            levels = sorted(deg_data[deg].keys())
            eers   = [deg_data[deg][lvl] * 100 for lvl in levels]
            ax.plot(levels, eers, marker="o", label=model_name,
                    color=color, linewidth=2, markersize=6)

        ax.set_xlabel("Degradation Intensity", fontsize=13)
        ax.set_ylabel("EER (%)", fontsize=13)
        ax.set_title(f"Robustness to {deg.replace('_', ' ').title()}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = os.path.join(save_dir, f"robustness_{deg}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_score_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Score Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay genuine vs. impostor score histograms.
    The overlap area is proportional to EER — useful for sanity-checking
    that the model has actually learned a meaningful embedding.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    genuine_scores  = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    ax.hist(impostor_scores, bins=80, alpha=0.6, color="tomato",   label="Impostor", density=True)
    ax.hist(genuine_scores,  bins=80, alpha=0.6, color="steelblue", label="Genuine",  density=True)

    ax.set_xlabel("Similarity Score", fontsize=13)
    ax.set_ylabel("Density",          fontsize=13)
    ax.set_title(title,               fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
