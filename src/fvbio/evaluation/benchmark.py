"""
benchmark.py — Dataset-agnostic biometric verification evaluator.

This module runs any model × any dataset combination and produces a
standardised results dict. The design is intentionally flat:
  - No subclassing required
  - Models only need .score_pairs(pairs) → (scores, labels)
  - Datasets only need to return a list of pairs with .load() and .label

The output dict schema is:
{
  "model":    str,
  "dataset":  str,
  "eer":      float,            # Equal Error Rate
  "auc":      float,            # Area Under ROC
  "tar_at_far": {               # TAR at fixed FAR values
      0.001:  float,            # TAR@FAR=0.1%
      0.0001: float,            # TAR@FAR=0.01%
  },
  "roc": {
      "far": np.ndarray,
      "tar": np.ndarray,
  },
  "scores": np.ndarray,
  "labels": np.ndarray,
}
"""

from __future__ import annotations

import os
import json
import pathlib
import numpy as np
from typing import Any, Dict, List

from fvbio.metrics.biometric import evaluate, plot_roc_curves, plot_score_distributions


# ─────────────────────────────────────────────────────────────────────────────
# Single model × dataset run
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model: Any,
    pairs: List[Any],
    model_name: str,
    dataset_name: str,
    far_thresholds: List[float] | None = None,
    num_thresholds: int = 500,
) -> Dict:
    """
    Score all pairs with the model, then compute the full metric suite.

    Args:
        model:          verifier object with .score_pairs()
        pairs:          list of pair objects with .load() and .label
        model_name:     display name (for logging and plot labels)
        dataset_name:   display name
        far_thresholds: list of FAR values for TAR@FAR computation
        num_thresholds: number of threshold points for ROC/EER curves

    Returns:
        results dict (see module docstring for schema)
    """
    if far_thresholds is None:
        far_thresholds = [0.001, 0.0001, 0.00001]

    print(f"\n[eval] {model_name} on {dataset_name} ({len(pairs)} pairs)")

    scores, labels = model.score_pairs(pairs)

    metrics = evaluate(
        scores,
        labels,
        far_thresholds=far_thresholds,
        num_thresholds=num_thresholds,
    )

    result = {
        "model":      model_name,
        "dataset":    dataset_name,
        "eer":        metrics["eer"],
        "auc":        metrics["auc"],
        "tar_at_far": metrics["tar_at_far"],
        "roc":        metrics["roc"],
        "scores":     scores,
        "labels":     labels,
    }

    _print_summary(result)
    return result


def _print_summary(result: Dict) -> None:
    print(f"  EER:  {result['eer'] * 100:.2f}%")
    print(f"  AUC:  {result['auc']:.4f}")
    for far, tar in result["tar_at_far"].items():
        print(f"  TAR@FAR={far:.4%}: {tar * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark: N models × M datasets
# ─────────────────────────────────────────────────────────────────────────────

def run_full_benchmark(
    models_dict: Dict[str, Any],    # {"lbp_svm": LBPVerifier(), ...}
    datasets_dict: Dict[str, List], # {"lfw": [...pairs...], ...}
    far_thresholds: List[float] | None = None,
    plots_dir: str = "plots",
    results_dir: str = "results",
) -> Dict[str, Dict[str, Dict]]:
    """
    Run all model × dataset combinations, save plots and JSON results.

    Returns:
        Nested dict: results[dataset_name][model_name] = result_dict
    """
    pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Dict]] = {}

    for ds_name, pairs in datasets_dict.items():
        all_results[ds_name] = {}

        for model_name, model in models_dict.items():
            result = run_evaluation(
                model=model,
                pairs=pairs,
                model_name=model_name,
                dataset_name=ds_name,
                far_thresholds=far_thresholds,
            )
            all_results[ds_name][model_name] = result

        # ── Per-dataset ROC overlay plot ──────────────────────────────────
        roc_inputs = {
            m: all_results[ds_name][m]
            for m in models_dict
        }
        roc_save = os.path.join(plots_dir, f"roc_{ds_name}.png")
        plot_roc_curves(
            roc_inputs,
            title=f"ROC Curves — {ds_name.upper()}",
            save_path=roc_save,
        )
        print(f"  ROC plot saved: {roc_save}")

        # ── Per-model score distribution plots ────────────────────────────
        for model_name, result in all_results[ds_name].items():
            dist_save = os.path.join(plots_dir, f"dist_{ds_name}_{model_name}.png")
            plot_score_distributions(
                result["scores"],
                result["labels"],
                title=f"Score Distribution — {model_name} / {ds_name}",
                save_path=dist_save,
            )

    # ── Save serialisable JSON summary ────────────────────────────────────
    _save_results_json(all_results, results_dir)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation (strips numpy arrays from the top-level results)
# ─────────────────────────────────────────────────────────────────────────────

def _save_results_json(
    all_results: Dict[str, Dict[str, Dict]],
    results_dir: str,
) -> None:
    """
    Save a JSON-serialisable summary (no numpy arrays).
    The raw score arrays are saved separately as .npy files.
    """
    summary = {}

    for ds_name, ds_results in all_results.items():
        summary[ds_name] = {}

        for model_name, res in ds_results.items():
            summary[ds_name][model_name] = {
                "eer":            res["eer"],
                "auc":            res["auc"],
                "tar_at_far":     {str(k): v for k, v in res["tar_at_far"].items()},
            }
            # Save raw scores and labels as .npy for offline analysis
            np.save(
                os.path.join(results_dir, f"scores_{ds_name}_{model_name}.npy"),
                res["scores"],
            )
            np.save(
                os.path.join(results_dir, f"labels_{ds_name}_{model_name}.npy"),
                res["labels"],
            )

    out_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print benchmark table
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmark_table(all_results: Dict[str, Dict[str, Dict]]) -> None:
    """
    Print a LaTeX-ready Markdown-style table of EER and AUC values.
    """
    # Collect all model and dataset names
    datasets = list(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())

    header_parts = ["Model"]
    for ds in datasets:
        header_parts += [f"{ds} EER (%)", f"{ds} AUC"]
    header = " | ".join(header_parts)
    separator = " | ".join(["---"] * len(header_parts))

    print("\n" + header)
    print(separator)

    for model in models:
        row = [model]
        for ds in datasets:
            res = all_results[ds][model]
            row.append(f"{res['eer'] * 100:.2f}")
            row.append(f"{res['auc']:.4f}")
        print(" | ".join(row))
