"""
robustness.py — Cross-domain degradation robustness sweep.

For each (model, degradation_type, intensity_level) triple:
  1. Load all image pairs from the dataset
  2. Apply the degradation to BOTH images at the given intensity
  3. Score with the model
  4. Compute EER

This produces the data for the robustness curves:
  x-axis: degradation intensity
  y-axis: EER (%)
  one curve per model

The sweep is the main research contribution of the project:
it answers "how gracefully does classical vs. deep degrade under
blur / low-light / sketch-style conditions?"

Design notes
────────────
- We apply the degradation AFTER loading (not baked into the dataset),
  so we can reuse the same pair list across all sweep points.
- We cache embeddings at intensity=0 (baseline) to avoid re-computing
  for LBP/FaceNet which are deterministic.
- Results are returned as a nested dict for easy serialisation:
    {model_name: {deg_type: {intensity: eer_value}}}
"""

from __future__ import annotations

import os
import json
import pathlib
import numpy as np
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from fvbio.metrics.biometric import compute_eer
from fvbio.degradation.transforms import apply_degradation


# ─────────────────────────────────────────────────────────────────────────────
# Core sweep function
# ─────────────────────────────────────────────────────────────────────────────

def score_pairs_with_degradation(
    model: Any,
    pairs: List[Any],
    degradation_type: str,
    intensity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score all pairs after applying a degradation transform to both images.

    Args:
        model:            verifier with .similarity(img1, img2) method
        pairs:            list of pair objects (must have .load() and .label)
        degradation_type: one of "blur", "low_light", "sketch"
        intensity:        scalar intensity value

    Returns:
        scores: float64 array of cosine similarities
        labels: int32 array of 0/1 labels
    """
    scores = []
    labels = []

    for pair in pairs:
        img1_orig, img2_orig = pair.load()

        img1 = apply_degradation(img1_orig, degradation_type, intensity)
        img2 = apply_degradation(img2_orig, degradation_type, intensity)

        score = model.similarity(img1, img2)
        scores.append(score)
        labels.append(pair.label)

    return (
        np.array(scores, dtype=np.float64),
        np.array(labels,  dtype=np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single model × single degradation sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_degradation(
    model: Any,
    pairs: List[Any],
    degradation_type: str,
    intensity_levels: List[float],
    num_thresholds: int = 500,
    verbose: bool = True,
) -> Dict[float, float]:
    """
    Sweep a single degradation type across all intensity levels for one model.

    Returns:
        {intensity: eer_value} dict
    """
    eer_by_level: Dict[float, float] = {}

    for intensity in intensity_levels:
        if verbose:
            print(f"    intensity={intensity:.2f}", end=" ... ", flush=True)

        scores, labels = score_pairs_with_degradation(
            model, pairs, degradation_type, intensity
        )
        eer, _ = compute_eer(scores, labels, num_thresholds=num_thresholds)
        eer_by_level[intensity] = eer

        if verbose:
            print(f"EER={eer * 100:.2f}%")

    return eer_by_level


# ─────────────────────────────────────────────────────────────────────────────
# Full robustness sweep: all models × all degradations
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness_sweep(
    models_dict: Dict[str, Any],
    pairs: List[Any],
    degradation_cfg: List[Dict],
    dataset_name: str = "lfw",
    plots_dir: str = "plots",
    results_dir: str = "results",
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """
    Full sweep: every model × every degradation type × every intensity level.

    Args:
        models_dict:     {"model_name": verifier_object, ...}
        pairs:           list of pair objects from a dataset loader
        degradation_cfg: list of dicts from experiment.yaml:
                         [{"name": "blur", "levels": [0, 0.5, ...]}, ...]
        dataset_name:    used for output file naming
        plots_dir:       directory for robustness curve plots
        results_dir:     directory for JSON output

    Returns:
        {model_name: {deg_type: {intensity: eer}}}
    """
    pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Dict[float, float]]] = {}

    for model_name, model in models_dict.items():
        print(f"\n[robustness] Model: {model_name}")
        results[model_name] = {}

        for deg_cfg in degradation_cfg:
            deg_name   = deg_cfg["name"]
            levels     = deg_cfg["levels"]
            print(f"  Degradation: {deg_name}")

            eer_by_level = sweep_degradation(
                model=model,
                pairs=pairs,
                degradation_type=deg_name,
                intensity_levels=levels,
            )
            results[model_name][deg_name] = eer_by_level

    # Save plots
    from fvbio.metrics.biometric import plot_robustness_curves
    plot_robustness_curves(results, save_dir=plots_dir)

    # Save JSON
    _save_robustness_json(results, dataset_name, results_dir)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation (float keys → string)
# ─────────────────────────────────────────────────────────────────────────────

def _save_robustness_json(
    results: Dict[str, Dict[str, Dict[float, float]]],
    dataset_name: str,
    results_dir: str,
) -> None:
    """
    JSON doesn't support float keys — convert to strings.
    """
    serialisable: Dict = {}

    for model_name, deg_data in results.items():
        serialisable[model_name] = {}
        for deg_name, level_data in deg_data.items():
            serialisable[model_name][deg_name] = {
                str(k): v for k, v in level_data.items()
            }

    out_path = os.path.join(results_dir, f"robustness_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\nRobustness results saved: {out_path}")
