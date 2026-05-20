"""
run_robustness.py — Cross-domain degradation robustness sweep entry point.

Usage:
    python experiments/run_robustness.py
    python experiments/run_robustness.py --dataset cfp_fp
    python experiments/run_robustness.py --models facenet arcface
    python experiments/run_robustness.py --degradations blur low_light

What this script does:
  1. Load config (same as benchmark)
  2. Build specified models
  3. Load ONE dataset (default: LFW — smaller, faster to sweep)
  4. For each (model, degradation, intensity) triple:
       - Apply transform to both images in every pair
       - Score and compute EER
  5. Save robustness curves (one PNG per degradation type)
  6. Save JSON with all EER values

Total number of evaluations:
  3 models × 3 degradations × 6 levels × 6000 pairs = 324,000 pair scorings
  (LBP runs fast; FaceNet/ArcFace will benefit from GPU)
"""

import argparse
import os
import sys
import random

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fvbio.datasets.lfw           import load_lfw_pairs
from fvbio.datasets.cfp           import load_cfp_pairs
from fvbio.models.lbp_svm         import LBPVerifier
from fvbio.models.deep_embeddings import FaceNetVerifier, ArcFaceVerifier
from fvbio.evaluation.robustness  import run_robustness_sweep


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run robustness degradation sweep")
    p.add_argument("--config",       default="configs/experiment.yaml")
    p.add_argument("--dataset",      default="lfw",
                   choices=["lfw", "cfp_fp"])
    p.add_argument("--models",       nargs="+",
                   choices=["lbp_svm", "facenet", "arcface"],
                   default=["lbp_svm", "facenet", "arcface"])
    p.add_argument("--degradations", nargs="+",
                   choices=["blur", "low_light", "sketch"],
                   default=None,   # None = all from config
                   help="Subset of degradation types to sweep")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap the number of pairs evaluated (random sample, reproducible). "
                        "Useful for deep models on CPU: 500 pairs gives accurate trends.")
    p.add_argument("--no-verify", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    plots_dir   = cfg["plots_dir"]
    results_dir = cfg["results_dir"]
    verify      = not args.no_verify

    # ── Filter degradation config ─────────────────────────────────────────
    deg_cfg = cfg["degradation"]["types"]
    if args.degradations is not None:
        deg_cfg = [d for d in deg_cfg if d["name"] in args.degradations]

    # ── Build models ──────────────────────────────────────────────────────
    models_dict = {}
    model_cfgs  = {m["name"]: m for m in cfg["models"]}

    for name in args.models:
        print(f"Loading model: {name}")
        mcfg = model_cfgs.get(name, {})

        if name == "lbp_svm":
            lbp_cfg = cfg.get("lbp", {})
            models_dict[name] = LBPVerifier(
                image_size=tuple(lbp_cfg.get("image_size", [128, 128])),
                n_points=lbp_cfg.get("n_points", 24),
                radius=lbp_cfg.get("radius", 3),
                n_bins=lbp_cfg.get("n_bins", 256),
            )

        elif name == "facenet":
            models_dict[name] = FaceNetVerifier(
                weights=mcfg.get("weights", "vggface2"),
            )

        elif name == "arcface":
            models_dict[name] = ArcFaceVerifier(
                model_pack=mcfg.get("model_pack", "buffalo_l"),
            )

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    try:
        if args.dataset == "lfw":
            pairs = load_lfw_pairs(
                lfw_root=cfg["data"]["lfw_root"],
                pairs_file=cfg["data"]["lfw_pairs"],
                verify_files=verify,
            )
        else:
            pairs = load_cfp_pairs(
                cfp_root=cfg["data"]["cfp_root"],
                pairs_file=cfg["data"].get("cfp_pairs"),
                protocol="FP",
                verify_files=verify,
            )
        print(f"  {len(pairs)} pairs loaded")
        if args.max_pairs and args.max_pairs < len(pairs):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pairs), size=args.max_pairs, replace=False)
            pairs = [pairs[i] for i in sorted(idx)]
            print(f"  Subsampled to {len(pairs)} pairs (--max-pairs {args.max_pairs})")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Run the download script in data/")
        sys.exit(1)

    # ── Run sweep ─────────────────────────────────────────────────────────
    print(f"\nSweeping {len(args.models)} models × "
          f"{len(deg_cfg)} degradations × "
          f"{max(len(d['levels']) for d in deg_cfg)} levels")

    results = run_robustness_sweep(
        models_dict=models_dict,
        pairs=pairs,
        degradation_cfg=deg_cfg,
        dataset_name=args.dataset,
        plots_dir=plots_dir,
        results_dir=results_dir,
    )

    # ── Print summary table ───────────────────────────────────────────────
    print("\n── Robustness Summary (EER % at max degradation) ──")
    for model_name, deg_data in results.items():
        print(f"\n  {model_name}:")
        for deg_name, level_data in deg_data.items():
            max_level = max(level_data.keys())
            max_eer   = level_data[max_level] * 100
            base_eer  = list(level_data.values())[0] * 100
            print(f"    {deg_name}: {base_eer:.2f}% → {max_eer:.2f}% EER")

    print(f"\nRobustness plots: {plots_dir}/")
    print(f"JSON results:     {results_dir}/")


if __name__ == "__main__":
    main()
