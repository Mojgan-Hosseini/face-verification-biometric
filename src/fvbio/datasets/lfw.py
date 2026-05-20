"""
lfw.py — LFW (Labeled Faces in the Wild) dataset loader.

LFW standard evaluation protocol
──────────────────────────────────
The official "pairs.txt" file defines 6,000 pairs across 10 folds:
  - 3,000 genuine pairs  (same identity, label=1)
  - 3,000 impostor pairs (different identity, label=0)

pairs.txt format:
  Line 1:          "10\t300"   (num_folds, pairs_per_fold)
  Genuine pairs:   "name\tidx1\tidx2"   → name_idx1.jpg vs name_idx2.jpg
  Impostor pairs:  "name1\tidx1\tname2\tidx2"

Image filename format:   <name>/<name>_<idx:04d>.jpg
  e.g.  Aaron_Eckhart/Aaron_Eckhart_0001.jpg

Download:
  http://vis-www.cs.umass.edu/lfw/lfw.tgz           (all images)
  http://vis-www.cs.umass.edu/lfw/pairs.txt          (standard split)

The script data/download_lfw.sh automates this.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Pair record
# ─────────────────────────────────────────────────────────────────────────────

class LFWPair:
    """
    A single evaluation pair.

    Attributes:
        path1:  absolute path to the first image
        path2:  absolute path to the second image
        label:  1 = same identity (genuine), 0 = different (impostor)
    """
    __slots__ = ("path1", "path2", "label")

    def __init__(self, path1: str, path2: str, label: int):
        self.path1 = path1
        self.path2 = path2
        self.label = label

    def load(self) -> Tuple[Image.Image, Image.Image]:
        """Load both images as RGB PIL Images."""
        img1 = Image.open(self.path1).convert("RGB")
        img2 = Image.open(self.path2).convert("RGB")
        return img1, img2

    def __repr__(self) -> str:
        tag = "genuine" if self.label == 1 else "impostor"
        return f"LFWPair({Path(self.path1).name} | {Path(self.path2).name} | {tag})"


# ─────────────────────────────────────────────────────────────────────────────
# Path helper
# ─────────────────────────────────────────────────────────────────────────────

def _image_path(lfw_root: str, name: str, idx: int) -> str:
    """
    Construct the full path to an LFW image.

    LFW stores images as:  <lfw_root>/<name>/<name>_<idx:04d>.jpg
    """
    filename = f"{name}_{idx:04d}.jpg"
    return os.path.join(lfw_root, name, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_lfw_pairs(
    lfw_root: str,
    pairs_file: str,
    verify_files: bool = True,
) -> List[LFWPair]:
    """
    Parse pairs.txt and build a list of LFWPair objects.

    Args:
        lfw_root:     root directory of extracted LFW images
        pairs_file:   path to pairs.txt
        verify_files: if True, raise FileNotFoundError for missing images
                      (useful during setup; set False for speed once confirmed)

    Returns:
        List of LFWPair objects (length = num_folds * pairs_per_fold * 2)
    """
    pairs: List[LFWPair] = []
    missing: List[str]   = []

    with open(pairs_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # First line: "num_folds\tpairs_per_fold"
    header     = lines[0].split()
    num_folds  = int(header[0])
    pairs_per_fold = int(header[1])
    data_lines = lines[1:]  # everything after the header

    # Each fold = pairs_per_fold genuine lines + pairs_per_fold impostor lines
    # Total data lines = num_folds * pairs_per_fold * 2
    for line in data_lines:
        parts = line.split("\t")

        if len(parts) == 3:
            # Genuine pair: name, idx1, idx2
            name, idx1, idx2 = parts[0], int(parts[1]), int(parts[2])
            p1 = _image_path(lfw_root, name, idx1)
            p2 = _image_path(lfw_root, name, idx2)
            label = 1

        elif len(parts) == 4:
            # Impostor pair: name1, idx1, name2, idx2
            name1, idx1, name2, idx2 = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1 = _image_path(lfw_root, name1, idx1)
            p2 = _image_path(lfw_root, name2, idx2)
            label = 0

        else:
            # Skip malformed lines (e.g. blank lines inside folds)
            continue

        if verify_files:
            for p in (p1, p2):
                if not os.path.exists(p):
                    missing.append(p)

        pairs.append(LFWPair(p1, p2, label))

    if missing:
        msg = f"{len(missing)} image(s) not found. First 5: {missing[:5]}"
        raise FileNotFoundError(msg)

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary(pairs: List[LFWPair]) -> dict:
    """Print and return basic dataset statistics."""
    n_genuine  = sum(p.label == 1 for p in pairs)
    n_impostor = sum(p.label == 0 for p in pairs)
    summary = {
        "total_pairs": len(pairs),
        "genuine":     n_genuine,
        "impostor":    n_impostor,
    }
    print(f"LFW pairs loaded: {len(pairs)} total "
          f"({n_genuine} genuine, {n_impostor} impostor)")
    return summary
