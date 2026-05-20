"""
deep_embeddings.py — FaceNet and ArcFace verification wrappers.

Both models follow the same interface:
    model.similarity(img1, img2) -> float   # cosine similarity in [-1, 1]
    model.embed(image) -> np.ndarray        # 512-d L2-normalised embedding
    model.score_pairs(pairs) -> (scores, labels)

FaceNet (Schroff et al., 2015)
──────────────────────────────
  Architecture: InceptionResNetV1
  Training:     VGGFace2 (3.3M images, 9K identities) with triplet loss
  Package:      facenet-pytorch (https://github.com/timesler/facenet-pytorch)
  Input size:   160×160 aligned face
  Embedding:    512-d, L2-normalised

ArcFace (Deng et al., 2019)
────────────────────────────
  Architecture: ResNet50 + ArcFace loss (additive angular margin)
  Training:     MS1Mv2 (5.8M images, 85K identities)
  Package:      insightface (https://github.com/deepinsight/insightface)
  Model pack:   buffalo_l (ships ONNX weights, no PyTorch needed for inference)
  Input size:   112×112 aligned face
  Embedding:    512-d, L2-normalised

Why cosine similarity?
  Both models are trained to cluster identities on a hypersphere; cosine
  similarity (= dot product of L2-normalised vectors) is the geometrically
  correct distance metric. Euclidean distance would also work but cosine is
  invariant to vector magnitude.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from typing import Tuple, List
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ─────────────────────────────────────────────────────────────────────────────
# FaceNet verifier
# ─────────────────────────────────────────────────────────────────────────────

class FaceNetVerifier:
    """
    Wrapper around facenet-pytorch for face verification.

    Face detection + alignment uses MTCNN (Multi-task CNN), which outputs
    a 160×160 aligned crop ready for InceptionResNetV1.

    If MTCNN fails to detect a face (e.g. extreme degradation), we fall back
    to a centre crop + resize so the pipeline doesn't crash.
    """

    def __init__(
        self,
        weights: str = "vggface2",
        image_size: int = 160,
        margin: int = 20,
        min_face_size: int = 40,
        device: str = "auto",
    ):
        from facenet_pytorch import InceptionResnetV1, MTCNN

        self.device     = _resolve_device(device)
        self.image_size = image_size

        # MTCNN: detects + aligns faces
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            device=self.device,
            keep_all=False,        # return only the largest detected face
            post_process=True,     # standardise pixel values to [-1, 1]
        )

        # InceptionResNetV1: produces 512-d embeddings
        self.model = InceptionResnetV1(pretrained=weights).eval().to(self.device)

    def _align(self, image: Image.Image) -> torch.Tensor:
        """
        Detect and align face using MTCNN.
        Returns a (1, 3, 160, 160) tensor on self.device.

        Falls back to centre-crop if no face detected.
        """
        face_tensor = self.mtcnn(image)

        if face_tensor is None:
            # Fallback: resize and normalise manually
            img = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # → [-1, 1]
            face_tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # C×H×W

        return face_tensor.unsqueeze(0).to(self.device)   # → 1×C×H×W

    def embed(self, image: Image.Image) -> np.ndarray:
        """
        Extract a 512-d L2-normalised embedding.
        Returns: float32 numpy array of shape (512,)
        """
        with torch.no_grad():
            tensor = self._align(image)
            emb    = self.model(tensor)          # (1, 512)
            emb    = emb / emb.norm(dim=1, keepdim=True)  # L2-normalise
        return emb.cpu().numpy().squeeze(0)

    def similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Cosine similarity between two face images. Range: [-1, 1]."""
        e1 = self.embed(img1)
        e2 = self.embed(img2)
        return float(np.dot(e1, e2))

    def score_pairs(self, pairs) -> Tuple[np.ndarray, np.ndarray]:
        """Score a list of dataset pairs. Returns (scores, labels)."""
        scores, labels = [], []
        for pair in tqdm(pairs, desc="FaceNet scoring", unit="pair"):
            img1, img2 = pair.load()
            scores.append(self.similarity(img1, img2))
            labels.append(pair.label)
        return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace verifier
# ─────────────────────────────────────────────────────────────────────────────

class ArcFaceVerifier:
    """
    Wrapper around insightface's ArcFace model for face verification.

    insightface ships ONNX-exported models — no PyTorch training loop needed.
    The buffalo_l pack includes:
      - det_10g.onnx  → RetinaFace detector
      - w600k_r50.onnx → ArcFace ResNet50 recogniser

    The FaceAnalysis object handles detection + alignment + embedding in one
    call. We extract only the embedding from the first detected face.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        device: str = "auto",
    ):
        import insightface
        from insightface.app import FaceAnalysis

        dev = _resolve_device(device)
        ctx_id = 0 if dev == "cuda" else -1   # insightface uses -1 for CPU

        self.app = FaceAnalysis(
            name=model_pack,
            providers=["CUDAExecutionProvider" if dev == "cuda"
                       else "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed(self, image: Image.Image) -> np.ndarray:
        """
        Extract a 512-d L2-normalised ArcFace embedding.

        insightface expects a BGR numpy array (OpenCV convention).
        Returns: float32 numpy array of shape (512,)
        """
        import cv2

        # PIL (RGB) → numpy BGR
        img_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        faces = self.app.get(img_bgr)

        if len(faces) == 0:
            # Fallback: run on resized image without detection
            img_112 = cv2.resize(img_bgr, (112, 112))
            faces   = self.app.get(img_112)

        if len(faces) == 0:
            # If still no face found, return a zero vector
            # (will score as ~0 similarity — correctly uncertain)
            return np.zeros(512, dtype=np.float32)

        emb  = faces[0].embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Cosine similarity. Range: [-1, 1]."""
        e1 = self.embed(img1)
        e2 = self.embed(img2)
        return float(np.dot(e1, e2))

    def score_pairs(self, pairs) -> Tuple[np.ndarray, np.ndarray]:
        """Score a list of dataset pairs. Returns (scores, labels)."""
        scores, labels = [], []
        for pair in tqdm(pairs, desc="ArcFace scoring", unit="pair"):
            img1, img2 = pair.load()
            scores.append(self.similarity(img1, img2))
            labels.append(pair.label)
        return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, cfg: dict):
    """
    Instantiate a model by its config name.

    Args:
        name: one of "lbp_svm", "facenet", "arcface"
        cfg:  the model's sub-dict from experiment.yaml

    Returns:
        A verifier object with .similarity() and .score_pairs() methods
    """
    if name == "lbp_svm":
        from fvbio.models.lbp_svm import LBPVerifier
        return LBPVerifier()

    elif name == "facenet":
        return FaceNetVerifier(
            weights=cfg.get("weights", "vggface2"),
        )

    elif name == "arcface":
        return ArcFaceVerifier(
            model_pack=cfg.get("model_pack", "buffalo_l"),
        )

    else:
        raise ValueError(f"Unknown model '{name}'. Expected: lbp_svm | facenet | arcface")
