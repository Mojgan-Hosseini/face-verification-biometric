"""
lbp_svm.py — Classical face verification pipeline: LBP features + cosine similarity.

Algorithm overview
──────────────────
1. Face alignment: resize to fixed size, optional histogram equalization
2. LBP feature extraction:
   - Divide the face into a 4×4 grid of non-overlapping cells
   - In each cell, compute the LBP histogram (n_points=24, radius=3)
   - Concatenate all cell histograms → 1D feature vector (256 bins × 16 cells = 4096 dims)
   - L2-normalize
3. Verification: cosine similarity between two feature vectors
   - Score = cos(f1, f2) ∈ [−1, 1]; higher = more similar

Why LBP?
  LBP is illumination-invariant by design (it encodes relative intensity
  orderings, not absolute values) and is the canonical baseline in face
  recognition papers from Ahonen et al. (2006) onward.

Why cosine, not SVM decision boundary?
  In the verification setting we need a SCORE (not a binary decision) to
  compute ROC/EER curves. Cosine similarity is the natural score for
  normalized histograms. The SVM in the class name refers to the historical
  pipeline — many papers train an SVM on LBP features for closed-set ID, but
  for open-set verification cosine is more appropriate.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Tuple

from skimage.feature import local_binary_pattern


class LBPVerifier:
    """
    LBP-based face verification.

    Usage:
        verifier = LBPVerifier()
        score = verifier.similarity(img1, img2)   # float in [-1, 1]

    The verifier is stateless (no training required) — LBP is an unsupervised
    feature extractor. All configuration is passed at construction time.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        n_points: int = 24,
        radius: int = 3,
        n_bins: int = 256,
        grid_size: Tuple[int, int] = (4, 4),
        equalize_hist: bool = True,
    ):
        """
        Args:
            image_size:    target (width, height) before feature extraction
            n_points:      number of circular LBP sample points
            radius:        radius of the circular LBP neighborhood (pixels)
            n_bins:        number of histogram bins per cell
            grid_size:     (rows, cols) grid to divide the face image into
            equalize_hist: apply CLAHE before LBP for illumination robustness
        """
        self.image_size    = image_size
        self.n_points      = n_points
        self.radius        = radius
        self.n_bins        = n_bins
        self.grid_size     = grid_size
        self.equalize_hist = equalize_hist

        # LBP method: "uniform" patterns cover ~90% of all patterns and are
        # more discriminative than the full non-uniform set
        self.lbp_method = "uniform"
        # scikit-image uniform LBP returns values in [0, P+1]; P+2 as exclusive upper bound
        self._n_uniform = n_points + 2

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Resize, convert to grayscale, optionally equalize histogram.
        Returns:  H×W uint8 grayscale array
        """
        img = image.convert("L").resize(self.image_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)

        if self.equalize_hist:
            # Contrast Limited Adaptive Histogram Equalization
            from skimage.exposure import equalize_adapthist
            arr = (equalize_adapthist(arr) * 255).astype(np.uint8)

        return arr

    # ── Feature extraction ───────────────────────────────────────────────────

    def _cell_histogram(self, cell: np.ndarray) -> np.ndarray:
        """
        Compute the LBP histogram for a single image cell.
        Returns: L2-normalized histogram of length n_bins
        """
        lbp = local_binary_pattern(
            cell,
            P=self.n_points,
            R=self.radius,
            method=self.lbp_method,
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.n_bins,
            range=(0, self._n_uniform),
            density=False,
        )
        hist = hist.astype(np.float64)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract the grid-LBP feature vector from a face image.

        Returns: 1-D float64 array of length (grid_rows * grid_cols * n_bins)
        """
        arr           = self._preprocess(image)
        h, w          = arr.shape
        rows, cols    = self.grid_size
        cell_h        = h // rows
        cell_w        = w // cols
        histograms    = []

        for r in range(rows):
            for c in range(cols):
                cell = arr[r * cell_h:(r + 1) * cell_h,
                           c * cell_w:(c + 1) * cell_w]
                histograms.append(self._cell_histogram(cell))

        feature = np.concatenate(histograms)
        # Final L2-norm over the full concatenated vector
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        return feature

    # ── Similarity ───────────────────────────────────────────────────────────

    def similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Cosine similarity between two face images.
        Returns:  float in [-1, 1]; 1 = identical, -1 = maximally dissimilar
        """
        f1 = self.extract(img1)
        f2 = self.extract(img2)
        return float(np.dot(f1, f2))   # both already L2-normalized

    def similarity_from_features(
        self, f1: np.ndarray, f2: np.ndarray
    ) -> float:
        """Cosine similarity from pre-extracted feature vectors."""
        n1 = np.linalg.norm(f1)
        n2 = np.linalg.norm(f2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(f1, f2) / (n1 * n2))

    # ── Batch evaluation ─────────────────────────────────────────────────────

    def score_pairs(self, pairs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a list of dataset pairs.

        Args:
            pairs: list of LFWPair or CFPPair objects (must have .load() and .label)

        Returns:
            scores: float array of cosine similarities
            labels: int array of 0/1 labels
        """
        from tqdm import tqdm

        scores = []
        labels = []

        for pair in tqdm(pairs, desc="LBP scoring", unit="pair"):
            img1, img2 = pair.load()
            scores.append(self.similarity(img1, img2))
            labels.append(pair.label)

        return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32)
