"""
demo.py — Gradio interactive face verification demo.

Usage:
    python app/demo.py
    python app/demo.py --port 7860 --share   # public link

What the demo does:
  - User uploads two face images
  - Selects a model (LBP, FaceNet, ArcFace)
  - Selects a degradation type + intensity
  - Gets:
      • Degraded versions of both images (visual preview)
      • Cosine similarity score
      • MATCH / NO MATCH verdict (threshold = EER operating point)

Architecture notes:
  - This file is self-contained: it imports from src/ but adds no new logic
  - All model loading is deferred to first use (lazy init) to keep startup fast
  - Models are cached in a module-level dict so they aren't reloaded on each call
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr
from fvbio.degradation.transforms import apply_degradation, TRANSFORM_REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# EER-based thresholds (set after running experiments/run_benchmark.py)
# These are typical reported values; replace with your actual results.
# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity threshold above which we declare MATCH
EER_THRESHOLDS = {
    "lbp_svm":  0.72,   # typical LBP cosine threshold on LFW
    "facenet":  0.40,   # FaceNet: EER ~0.4 cosine (VGGFace2 weights)
    "arcface":  0.28,   # ArcFace: tighter margin → lower threshold
}

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model cache
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: dict = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        print(f"[demo] Loading model: {model_name}")

        if model_name == "lbp_svm":
            from fvbio.models.lbp_svm import LBPVerifier
            _model_cache[model_name] = LBPVerifier()

        elif model_name == "facenet":
            from fvbio.models.deep_embeddings import FaceNetVerifier
            _model_cache[model_name] = FaceNetVerifier(weights="vggface2")

        elif model_name == "arcface":
            from fvbio.models.deep_embeddings import ArcFaceVerifier
            _model_cache[model_name] = ArcFaceVerifier(model_pack="buffalo_l")

    return _model_cache[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function (called by Gradio on every submit)
# ─────────────────────────────────────────────────────────────────────────────

def verify_faces(
    image1: np.ndarray | None,
    image2: np.ndarray | None,
    model_name: str,
    degradation_type: str,
    intensity: float,
) -> tuple:
    """
    Main inference function wired to the Gradio interface.

    Args:
        image1, image2:   numpy arrays from gr.Image (RGB, uint8)
        model_name:       "lbp_svm" | "facenet" | "arcface"
        degradation_type: "none" | "blur" | "low_light" | "sketch"
        intensity:        float, type-specific scale

    Returns:
        (degraded_img1, degraded_img2, score_text, verdict_text)
    """
    if image1 is None or image2 is None:
        return None, None, "Please upload both images.", ""

    # Convert numpy → PIL
    pil1 = Image.fromarray(image1.astype(np.uint8)).convert("RGB")
    pil2 = Image.fromarray(image2.astype(np.uint8)).convert("RGB")

    # Apply degradation
    if degradation_type == "none":
        deg1, deg2 = pil1, pil2
    else:
        deg1 = apply_degradation(pil1, degradation_type, intensity)
        deg2 = apply_degradation(pil2, degradation_type, intensity)

    # Get model and compute similarity
    try:
        model     = _get_model(model_name)
        score     = model.similarity(deg1, deg2)
    except Exception as e:
        err = f"Model error: {e}"
        return np.array(deg1), np.array(deg2), err, ""

    # Verdict
    threshold = EER_THRESHOLDS.get(model_name, 0.5)
    match     = score >= threshold
    verdict   = "✅  MATCH" if match else "❌  NO MATCH"

    # Score text
    score_text = (
        f"Similarity score: **{score:.4f}**\n"
        f"Threshold ({model_name} EER): {threshold:.4f}\n"
        f"Degradation: {degradation_type}  |  Intensity: {intensity:.2f}"
    )

    return np.array(deg1), np.array(deg2), score_text, verdict


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ─────────────────────────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    deg_choices = ["none"] + list(TRANSFORM_REGISTRY.keys())

    with gr.Blocks(
        title="Face Verification Biometric Demo",
    ) as demo:

        gr.Markdown(
            """
            # Face Verification Biometric Research Demo
            **Models**: LBP+SVM (classical) · FaceNet · ArcFace
            **Benchmarks**: LFW · CFP-FP (frontal vs. profile)
            Upload two face images, choose a model and optional degradation, and get a similarity score.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                img1_input  = gr.Image(label="Face 1", type="numpy", height=300)
                img1_output = gr.Image(label="Face 1 (degraded)", height=300, interactive=False)

            with gr.Column(scale=1):
                img2_input  = gr.Image(label="Face 2", type="numpy", height=300)
                img2_output = gr.Image(label="Face 2 (degraded)", height=300, interactive=False)

        with gr.Row():
            model_select = gr.Dropdown(
                label="Verification Model",
                choices=["lbp_svm", "facenet", "arcface"],
                value="facenet",
            )
            deg_select = gr.Dropdown(
                label="Degradation Type",
                choices=deg_choices,
                value="none",
            )
            intensity_slider = gr.Slider(
                label="Degradation Intensity",
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                value=0.0,
                info="Blur: sigma | Low-light: gamma (1=no change) | Sketch: blend [0-1]",
            )

        run_btn = gr.Button("Verify", variant="primary", size="lg")

        with gr.Row():
            score_out   = gr.Markdown(label="Score")
            verdict_out = gr.Textbox(label="Verdict", interactive=False, lines=1)

        # Intensity slider range adapts to degradation type
        def update_slider(deg_type: str):
            if deg_type == "blur":
                return gr.Slider(minimum=0.0, maximum=5.0, step=0.5, value=0.0)
            elif deg_type == "low_light":
                return gr.Slider(minimum=1.0, maximum=5.0, step=0.5, value=1.0,
                                 info="1.0 = no change; higher = darker")
            elif deg_type == "sketch":
                return gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0)
            else:  # none
                return gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0,
                                 interactive=False)

        deg_select.change(update_slider, inputs=deg_select, outputs=intensity_slider)

        run_btn.click(
            fn=verify_faces,
            inputs=[img1_input, img2_input, model_select, deg_select, intensity_slider],
            outputs=[img1_output, img2_output, score_out, verdict_out],
        )

        gr.Markdown(
            """
            ---
            **Note on thresholds**: The MATCH/NO MATCH threshold is set at the EER operating point
            for each model on the LFW benchmark. For your own use case, calibrate on a held-out set.
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Launch Gradio face verification demo")
    p.add_argument("--port",  type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    p.add_argument("--host",  default="127.0.0.1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
