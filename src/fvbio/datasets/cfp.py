"""
cfp.py — CFP-FP (Celebrities in Frontal-Profile) dataset loader.

CFP-FP evaluation protocol
────────────────────────────
CFP contains 500 subjects × 10 frontal + 4 profile images = 7,000 images.
The Frontal-Profile (FP) protocol pairs each subject's frontal images against
their own and other subjects' profile images — making it much harder than LFW
because pose variation is extreme (≈90°).

Directory layout after extraction:
  cfp-dataset/
    Data/
      Images/
        <subject_id>/
          frontal/   ← 10 images
          profile/   ← 4 images
    Protocol/
      Pair_list_F.txt   ← frontal-frontal pairs (FF protocol, not used here)
      Pair_list_FP.txt  ← frontal-profile pairs (FP protocol — THIS FILE)
      Split/
        FP/
          pair_list_<split>.mat

Pair_list_FP.txt format:
  Each line: "subject_id/frontal/img.jpg,subject_id/profile/img.jpg,label"
  OR the mat-file-based split format. We support both.

Note: Some CFP releases store pair lists as .mat files (MATLAB format).
We provide a fallback that reads them with scipy.io.loadmat.

Download: http://www.cfpw.io/
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Pair record
# ─────────────────────────────────────────────────────────────────────────────

class CFPPair:
    """
    A single CFP-FP evaluation pair.

    Attributes:
        path1:  path to frontal image
        path2:  path to profile image
        label:  1 = same identity, 0 = different identity
    """
    __slots__ = ("path1", "path2", "label")

    def __init__(self, path1: str, path2: str, label: int):
        self.path1 = path1
        self.path2 = path2
        self.label = label

    def load(self) -> Tuple[Image.Image, Image.Image]:
        img1 = Image.open(self.path1).convert("RGB")
        img2 = Image.open(self.path2).convert("RGB")
        return img1, img2

    def __repr__(self) -> str:
        tag = "genuine" if self.label == 1 else "impostor"
        return f"CFPPair({Path(self.path1).name} | {Path(self.path2).name} | {tag})"


# ─────────────────────────────────────────────────────────────────────────────
# Text-format loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_txt(
    cfp_root: str,
    pairs_file: str,
    verify_files: bool = True,
) -> List[CFPPair]:
    """
    Load pairs from a CSV/TSV-style text file.

    Expected format per line (comma or tab separated):
        path1, path2, label
    where paths are relative to cfp_root/Data/Images/
    """
    pairs: List[CFPPair] = []
    images_root = os.path.join(cfp_root, "Data", "Images")
    missing: List[str] = []

    with open(pairs_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                # Try tab-split
                row = row[0].split("\t") if row else []
            if len(row) < 3:
                continue

            rel1, rel2, label_str = row[0].strip(), row[1].strip(), row[2].strip()
            p1 = os.path.join(images_root, rel1)
            p2 = os.path.join(images_root, rel2)
            label = int(label_str)

            if verify_files:
                for p in (p1, p2):
                    if not os.path.exists(p):
                        missing.append(p)

            pairs.append(CFPPair(p1, p2, label))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CFP image(s) not found. First 5: {missing[:5]}"
        )

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# MAT-file loader (official CFP protocol splits)
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_mat(
    cfp_root: str,
    split_dir: str | None = None,
    protocol: str = "FP",
    verify_files: bool = True,
) -> List[CFPPair]:
    """
    Load pairs from the official CFP .mat split files.

    The official CFP release stores pairs in MATLAB .mat files under:
        cfp_root/Protocol/Split/FP/pair_list_<split>.mat

    Each .mat file contains:
        pairs:  (N, 2) cell array of relative image paths
        labels: (N,) array of 0/1 labels

    We aggregate all 10 splits.
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy is required to load CFP .mat files: pip install scipy")

    if split_dir is None:
        split_dir = os.path.join(cfp_root, "Protocol", "Split", protocol)

    images_root = os.path.join(cfp_root, "Data", "Images")
    pairs: List[CFPPair] = []
    missing: List[str]   = []

    mat_files = sorted(Path(split_dir).glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {split_dir}")

    for mat_path in mat_files:
        mat = scipy.io.loadmat(str(mat_path))

        # Key names vary across CFP releases
        pair_key  = next((k for k in mat if "pair" in k.lower() and not k.startswith("_")), None)
        label_key = next((k for k in mat if "label" in k.lower() and not k.startswith("_")), None)

        if pair_key is None or label_key is None:
            continue

        pair_data  = mat[pair_key]   # shape (N, 2) cell array
        label_data = mat[label_key].flatten().astype(int)

        for i in range(len(label_data)):
            # Each cell may be a numpy array of strings
            rel1 = str(pair_data[i, 0]).strip()
            rel2 = str(pair_data[i, 1]).strip()
            p1   = os.path.join(images_root, rel1)
            p2   = os.path.join(images_root, rel2)

            if verify_files:
                for p in (p1, p2):
                    if not os.path.exists(p):
                        missing.append(p)

            pairs.append(CFPPair(p1, p2, int(label_data[i])))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CFP image(s) not found. First 5: {missing[:5]}"
        )

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_cfp_pairs(
    cfp_root: str,
    pairs_file: str | None = None,
    protocol: str = "FP",
    verify_files: bool = True,
) -> List[CFPPair]:
    """
    Load CFP-FP pairs, auto-detecting whether to use text or .mat format.

    Args:
        cfp_root:     root of the extracted CFP dataset
        pairs_file:   explicit path to a text-format pairs file.
                      If None, falls back to the .mat split directory.
        protocol:     "FP" (frontal-profile) or "FF" (frontal-frontal)
        verify_files: raise on missing images

    Returns:
        List of CFPPair objects
    """
    if pairs_file is not None and os.path.isfile(pairs_file):
        pairs = _load_from_txt(cfp_root, pairs_file, verify_files)
    else:
        # Try .mat split directory
        split_dir = os.path.join(cfp_root, "Protocol", "Split", protocol)
        if os.path.isdir(split_dir):
            pairs = _load_from_mat(cfp_root, split_dir, protocol, verify_files)
        else:
            raise FileNotFoundError(
                f"Could not find CFP pairs. "
                f"Expected text file at '{pairs_file}' "
                f"or .mat directory at '{split_dir}'.\n"
                f"Run data/download_cfp.sh to fetch the dataset."
            )

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary(pairs: List[CFPPair]) -> dict:
    n_genuine  = sum(p.label == 1 for p in pairs)
    n_impostor = sum(p.label == 0 for p in pairs)
    summary    = {"total_pairs": len(pairs), "genuine": n_genuine, "impostor": n_impostor}
    print(f"CFP-FP pairs loaded: {len(pairs)} total "
          f"({n_genuine} genuine, {n_impostor} impostor)")
    return summary
