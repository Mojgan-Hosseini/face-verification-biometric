"""
run_benchmark.py — Main entry point for benchmark evaluation.

Usage:
    # From repo root (after pip install -e .):
    python experiments/run_benchmark.py

    # With a custom config:
    python experiments/run_benchmark.py --config configs/experiment.yaml

    # Run only specific models or datasets:
    python experiments/run_benchmark.py --models lbp_svm facenet
    python experiments/run_benchmark.py --datasets lfw

What this script does:
  1. Load experiment config
  2. Build all models (LBP, FaceNet, ArcFace)
  3. Load LFW and CFP-FP pair lists
  4. Run evaluation: model.score_pairs() → metrics
  5. Save ROC plots, score distribution plots, and JSON summary
  6. Print the benchmark table (ready to paste into README)
"""

import argparse
import os
import sys
import random

import numpy as np
import yaml

# Make src/ importable without install (fallback for dev without pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fvbio.datasets.lfw      import load_lfw_pairs
from fvbio.datasets.cfp      import load_cfp_pairs
from fvbio.models.lbp_svm    import LBPVerifier
from fvbio.models.deep_embeddings import FaceNetVerifier, ArcFaceVerifier
from fvbio.evaluation.benchmark   import run_full_benchmark, print_benchmark_table


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run biometric benchmark")
    p.add_argument("--config",   default="configs/experiment.yaml")
    p.add_argument("--models",   nargs="+",
                   choices=["lbp_svm", "facenet", "arcface"],
                   default=["lbp_svm", "facenet", "arcface"])
    p.add_argument("--datasets", nargs="+",
                   choices=["lfw", "cfp_fp"],
                   default=["lfw", "cfp_fp"])
    p.add_argument("--no-verify", action="store_true",
                   help="Skip file existence checks (faster if already verified)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    plots_dir   = cfg["plots_dir"]
    results_dir = cfg["results_dir"]
    verify      = not args.no_verify

    # ── Build models ─────────────────────────────────────────────────────────
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

    # ── Load datasets ────────────────────────────────────────────────────────
    datasets_dict = {}

    if "lfw" in args.datasets:
        print("Loading LFW pairs...")
        try:
            pairs = load_lfw_pairs(
                lfw_root=cfg["data"]["lfw_root"],
                pairs_file=cfg["data"]["lfw_pairs"],
                verify_files=verify,
            )
            datasets_dict["lfw"] = pairs
            print(f"  LFW: {len(pairs)} pairs loaded")
        except FileNotFoundError as e:
            print(f"  [SKIP] LFW not found: {e}")
            print("  Run: bash data/download_lfw.sh")

    if "cfp_fp" in args.datasets:
        print("Loading CFP-FP pairs...")
        try:
            pairs = load_cfp_pairs(
                cfp_root=cfg["data"]["cfp_root"],
                pairs_file=cfg["data"].get("cfp_pairs"),
                protocol="FP",
                verify_files=verify,
            )
            datasets_dict["cfp_fp"] = pairs
            print(f"  CFP-FP: {len(pairs)} pairs loaded")
        except FileNotFoundError as e:
            print(f"  [SKIP] CFP-FP not found: {e}")
            print("  Run: bash data/download_cfp.sh")

    if not datasets_dict:
        print("\nNo datasets available. Run the download scripts first.")
        sys.exit(1)

    if not models_dict:
        print("\nNo models specified.")
        sys.exit(1)

    # ── Run benchmark ────────────────────────────────────────────────────────
    far_thresholds = cfg["metrics"]["far_thresholds"]

    all_results = run_full_benchmark(
        models_dict=models_dict,
        datasets_dict=datasets_dict,
        far_thresholds=far_thresholds,
        plots_dir=plots_dir,
        results_dir=results_dir,
    )

    # ── Print table ──────────────────────────────────────────────────────────
    print_benchmark_table(all_results)
    print(f"\nPlots saved to:   {plots_dir}/")
    print(f"Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
