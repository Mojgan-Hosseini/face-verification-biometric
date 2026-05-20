"""
test_metrics.py — Unit tests for src/metrics/biometric.py

Testing strategy
────────────────
All tests use synthetic score distributions with known analytical answers.
This is the right approach for metric code: we don't need real face images
to verify that EER is computed correctly — we need controlled inputs where
the expected output can be derived by hand or from first principles.

Fixtures
--------
- perfect_scores:    genuines all score 1.0, impostors all score 0.0
                     → EER = 0%, AUC = 1.0, TAR@any_FAR = 1.0

- random_scores:     genuines and impostors from the same distribution
                     → EER ≈ 50%, AUC ≈ 0.5 (random classifier)

- gaussian_scores:   genuines ~ N(0.7, 0.1), impostors ~ N(0.3, 0.1)
                     → EER ≈ 2–6% (well-separated), AUC close to 1.0

- inverted_scores:   genuines all score 0.0, impostors all score 1.0
                     → EER = 100%, AUC ≈ 0.0 (worst possible)
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fvbio.metrics.biometric import (
    compute_rates,
    compute_eer,
    compute_tar_at_far,
    compute_roc,
    evaluate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_scores():
    """Perfect classifier: genuines score 1.0, impostors score 0.0."""
    n = 500
    scores = np.concatenate([np.ones(n), np.zeros(n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


@pytest.fixture
def random_scores():
    """Random classifier: both classes from U(0, 1)."""
    rng = np.random.default_rng(0)
    n = 1000
    scores = rng.uniform(0, 1, 2 * n)
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


@pytest.fixture
def gaussian_scores():
    """Well-separated Gaussians: genuines ~ N(0.7, 0.1), impostors ~ N(0.3, 0.1)."""
    rng = np.random.default_rng(42)
    n = 3000
    genuine  = rng.normal(0.7, 0.10, n)
    impostor = rng.normal(0.3, 0.10, n)
    scores   = np.concatenate([genuine, impostor])
    labels   = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


@pytest.fixture
def inverted_scores():
    """Worst classifier: genuines score 0.0, impostors score 1.0."""
    n = 500
    scores = np.concatenate([np.zeros(n), np.ones(n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


# ─────────────────────────────────────────────────────────────────────────────
# compute_rates
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRates:

    def test_returns_three_arrays(self, gaussian_scores):
        scores, labels = gaussian_scores
        thresholds, far, frr = compute_rates(scores, labels)
        assert thresholds.shape == far.shape == frr.shape

    def test_far_decreasing_with_threshold(self, gaussian_scores):
        """Higher threshold → fewer impostors accepted → FAR decreases."""
        scores, labels = gaussian_scores
        thresholds, far, frr = compute_rates(scores, labels, num_thresholds=200)
        # FAR should be non-increasing as threshold increases
        assert np.all(np.diff(far) <= 1e-9), "FAR should be non-increasing"

    def test_frr_increasing_with_threshold(self, gaussian_scores):
        """Higher threshold → more genuines rejected → FRR increases."""
        scores, labels = gaussian_scores
        thresholds, far, frr = compute_rates(scores, labels, num_thresholds=200)
        assert np.all(np.diff(frr) >= -1e-9), "FRR should be non-decreasing"

    def test_far_range(self, gaussian_scores):
        scores, labels = gaussian_scores
        _, far, _ = compute_rates(scores, labels)
        assert np.all(far >= 0.0), "FAR must be >= 0"
        assert np.all(far <= 1.0), "FAR must be <= 1"

    def test_frr_range(self, gaussian_scores):
        scores, labels = gaussian_scores
        _, _, frr = compute_rates(scores, labels)
        assert np.all(frr >= 0.0), "FRR must be >= 0"
        assert np.all(frr <= 1.0), "FRR must be <= 1"

    def test_perfect_classifier_extremes(self, perfect_scores):
        """At very low threshold: FAR=1, FRR=0. At very high: FAR=0, FRR=1."""
        scores, labels = perfect_scores
        thresholds, far, frr = compute_rates(scores, labels)
        # At minimum threshold (accepts everything)
        assert far[0]  == pytest.approx(1.0, abs=0.05)
        assert frr[0]  == pytest.approx(0.0, abs=0.05)
        # At maximum threshold (rejects everything)
        assert far[-1] == pytest.approx(0.0, abs=0.05)
        assert frr[-1] == pytest.approx(1.0, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# compute_eer
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEER:

    def test_perfect_eer_is_zero(self, perfect_scores):
        """Perfect classifier should have EER close to 0."""
        scores, labels = perfect_scores
        eer, _ = compute_eer(scores, labels)
        assert eer == pytest.approx(0.0, abs=0.02), f"Expected EER≈0, got {eer:.4f}"

    def test_random_eer_is_half(self, random_scores):
        """Random classifier EER should be close to 0.5."""
        scores, labels = random_scores
        eer, _ = compute_eer(scores, labels)
        assert eer == pytest.approx(0.5, abs=0.05), f"Expected EER≈0.5, got {eer:.4f}"

    def test_gaussian_eer_reasonable(self, gaussian_scores):
        """Well-separated Gaussians: EER should be well below 0.1."""
        scores, labels = gaussian_scores
        eer, _ = compute_eer(scores, labels)
        assert eer < 0.10, f"Expected EER<10%, got {eer*100:.2f}%"
        assert eer > 0.0,  "EER should be > 0 for noisy distributions"

    def test_eer_returns_float(self, gaussian_scores):
        scores, labels = gaussian_scores
        eer, thresh = compute_eer(scores, labels)
        assert isinstance(eer,    float)
        assert isinstance(thresh, float)

    def test_eer_in_range(self, gaussian_scores):
        scores, labels = gaussian_scores
        eer, thresh = compute_eer(scores, labels)
        assert 0.0 <= eer <= 1.0

    def test_eer_threshold_in_score_range(self, gaussian_scores):
        scores, labels = gaussian_scores
        _, thresh = compute_eer(scores, labels)
        assert scores.min() <= thresh <= scores.max()

    def test_inverted_eer_near_one(self, inverted_scores):
        """Worst classifier: EER should be close to 1.0."""
        scores, labels = inverted_scores
        eer, _ = compute_eer(scores, labels)
        assert eer == pytest.approx(1.0, abs=0.05), f"Expected EER≈1, got {eer:.4f}"

    def test_num_thresholds_stability(self, gaussian_scores):
        """EER should be stable regardless of threshold resolution."""
        scores, labels = gaussian_scores
        eer_200,  _ = compute_eer(scores, labels, num_thresholds=200)
        eer_1000, _ = compute_eer(scores, labels, num_thresholds=1000)
        assert abs(eer_200 - eer_1000) < 0.005, "EER should be stable across resolutions"

    def test_single_pair_each_class(self):
        """Edge case: one genuine pair, one impostor pair."""
        scores = np.array([0.9, 0.1])
        labels = np.array([1,   0])
        eer, _ = compute_eer(scores, labels)
        assert 0.0 <= eer <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_tar_at_far
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTARatFAR:

    def test_perfect_tar_at_any_far(self, perfect_scores):
        """Perfect classifier: TAR=1.0 at any FAR threshold."""
        scores, labels = perfect_scores
        for far_target in [0.1, 0.01, 0.001, 0.0001]:
            tar = compute_tar_at_far(scores, labels, target_far=far_target)
            assert tar == pytest.approx(1.0, abs=0.05), \
                f"Perfect classifier TAR@FAR={far_target} should be ≈1.0, got {tar:.4f}"

    def test_random_tar_equals_far(self, random_scores):
        """Random classifier: TAR ≈ FAR (diagonal ROC)."""
        scores, labels = random_scores
        for far_target in [0.1, 0.2, 0.3]:
            tar = compute_tar_at_far(scores, labels, target_far=far_target)
            assert tar == pytest.approx(far_target, abs=0.08), \
                f"Random classifier TAR@FAR={far_target} ≈ FAR, got {tar:.4f}"

    def test_tar_in_range(self, gaussian_scores):
        scores, labels = gaussian_scores
        tar = compute_tar_at_far(scores, labels, target_far=0.01)
        assert 0.0 <= tar <= 1.0

    def test_tar_monotone_in_far(self, gaussian_scores):
        """Higher FAR threshold should yield at least as high TAR."""
        scores, labels = gaussian_scores
        tar_strict = compute_tar_at_far(scores, labels, target_far=0.001)
        tar_loose  = compute_tar_at_far(scores, labels, target_far=0.01)
        assert tar_loose >= tar_strict - 1e-6, \
            "TAR should be non-decreasing as FAR threshold increases"

    def test_tar_below_detection_floor(self, gaussian_scores):
        """At FAR=0 (unachievable), TAR should be 0.0."""
        scores, labels = gaussian_scores
        tar = compute_tar_at_far(scores, labels, target_far=0.0)
        assert tar == pytest.approx(0.0, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# compute_roc
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeROC:

    def test_returns_three_values(self, gaussian_scores):
        scores, labels = gaussian_scores
        result = compute_roc(scores, labels)
        assert len(result) == 3
        far, tar, auc = result
        assert isinstance(auc, float)

    def test_perfect_auc_is_one(self, perfect_scores):
        scores, labels = perfect_scores
        _, _, auc = compute_roc(scores, labels)
        assert auc == pytest.approx(1.0, abs=0.02)

    def test_random_auc_near_half(self, random_scores):
        scores, labels = random_scores
        _, _, auc = compute_roc(scores, labels)
        assert auc == pytest.approx(0.5, abs=0.06)

    def test_gaussian_auc_high(self, gaussian_scores):
        scores, labels = gaussian_scores
        _, _, auc = compute_roc(scores, labels)
        assert auc > 0.95, f"Well-separated distributions should have AUC>0.95, got {auc:.4f}"

    def test_inverted_auc_near_zero(self, inverted_scores):
        scores, labels = inverted_scores
        _, _, auc = compute_roc(scores, labels)
        assert auc < 0.1, f"Inverted classifier AUC should be <0.1, got {auc:.4f}"

    def test_auc_in_range(self, gaussian_scores):
        scores, labels = gaussian_scores
        _, _, auc = compute_roc(scores, labels)
        assert 0.0 <= auc <= 1.0

    def test_far_sorted(self, gaussian_scores):
        """FAR array must be sorted ascending (required for AUC trapz)."""
        scores, labels = gaussian_scores
        far, _, _ = compute_roc(scores, labels)
        assert np.all(np.diff(far) >= 0), "FAR must be sorted ascending in ROC output"

    def test_far_tar_same_length(self, gaussian_scores):
        scores, labels = gaussian_scores
        far, tar, _ = compute_roc(scores, labels)
        assert len(far) == len(tar)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate (full suite)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_output_keys(self, gaussian_scores):
        scores, labels = gaussian_scores
        result = evaluate(scores, labels)
        for key in ('eer', 'eer_threshold', 'auc', 'tar_at_far', 'roc'):
            assert key in result, f"Missing key: {key}"

    def test_tar_at_far_keys(self, gaussian_scores):
        scores, labels = gaussian_scores
        far_thresholds = [0.01, 0.001]
        result = evaluate(scores, labels, far_thresholds=far_thresholds)
        for t in far_thresholds:
            assert t in result['tar_at_far'], f"Missing TAR@FAR={t}"

    def test_roc_has_far_tar(self, gaussian_scores):
        scores, labels = gaussian_scores
        result = evaluate(scores, labels)
        assert 'far' in result['roc']
        assert 'tar' in result['roc']

    def test_consistency_between_helpers(self, gaussian_scores):
        """EER from evaluate() must match compute_eer() called directly."""
        scores, labels = gaussian_scores
        result   = evaluate(scores, labels)
        eer_direct, _ = compute_eer(scores, labels)
        assert result['eer'] == pytest.approx(eer_direct, abs=1e-9)

    def test_perfect_classifier_full(self, perfect_scores):
        scores, labels = perfect_scores
        result = evaluate(scores, labels, far_thresholds=[0.001])
        assert result['eer'] == pytest.approx(0.0, abs=0.02)
        assert result['auc'] == pytest.approx(1.0, abs=0.02)
        assert result['tar_at_far'][0.001] == pytest.approx(1.0, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Label-invariance: shuffling should not change any metric
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelInvariance:

    def test_eer_invariant_to_permutation(self, gaussian_scores):
        scores, labels = gaussian_scores
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(scores))
        eer1, _ = compute_eer(scores,       labels)
        eer2, _ = compute_eer(scores[idx],  labels[idx])
        assert eer1 == pytest.approx(eer2, abs=1e-9)

    def test_auc_invariant_to_permutation(self, gaussian_scores):
        scores, labels = gaussian_scores
        rng = np.random.default_rng(8)
        idx = rng.permutation(len(scores))
        _, _, auc1 = compute_roc(scores,      labels)
        _, _, auc2 = compute_roc(scores[idx], labels[idx])
        assert auc1 == pytest.approx(auc2, abs=1e-9)
