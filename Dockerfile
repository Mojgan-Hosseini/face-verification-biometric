# ─────────────────────────────────────────────────────────────────────────────
# Face Verification Biometric — Dockerfile
#
# Two build targets:
#   research   Full environment: benchmark experiments + Gradio demo
#   demo       Lightweight image: Gradio demo only (no heavy training deps)
#
# Usage:
#   # Full research environment
#   docker build --target research -t fvbio:research .
#   docker run --rm -v $(pwd)/data:/app/data \
#              -v $(pwd)/results:/app/results \
#              -v $(pwd)/plots:/app/plots \
#              fvbio:research python experiments/run_benchmark.py
#
#   # Gradio demo (port 7860)
#   docker build --target demo -t fvbio:demo .
#   docker run --rm -p 7860:7860 fvbio:demo
#
#   # GPU support (requires nvidia-docker)
#   docker run --gpus all --rm -p 7860:7860 fvbio:demo
# ─────────────────────────────────────────────────────────────────────────────

# ── Base: Python 3.11 slim (smaller than full, has build tools via apt) ──────
FROM python:3.11-slim AS base

# System dependencies required by OpenCV, insightface, and pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        # insightface ONNX runtime
        libstdc++6 \
        # curl for potential data downloads inside container
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first — layer is cached until they change
COPY requirements.txt setup.py ./
COPY src/ ./src/

# ── Research target: full install ────────────────────────────────────────────
FROM base AS research

# Install all dependencies including heavy torch/facenet/insightface stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy the rest of the project
COPY configs/  ./configs/
COPY experiments/ ./experiments/
COPY app/      ./app/
COPY notebooks/ ./notebooks/
COPY data/     ./data/

# Create output directories
RUN mkdir -p results plots

# Default: run the benchmark (override with docker run ... <command>)
CMD ["python", "experiments/run_benchmark.py", "--no-verify"]


# ── Demo target: lightweight Gradio app ──────────────────────────────────────
FROM base AS demo

# Install only what the demo needs
# We keep torch/facenet/insightface because the demo loads the models,
# but skip heavy dev/test extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch>=2.0 \
        torchvision>=0.15 \
        facenet-pytorch>=2.5 \
        insightface>=0.7 \
        onnxruntime>=1.16 \
        scikit-learn>=1.3 \
        scikit-image>=0.21 \
        opencv-python-headless>=4.8 \
        Pillow>=10.0 \
        numpy>=1.24 \
        scipy>=1.11 \
        gradio>=4.0 \
        PyYAML>=6.0 \
    && pip install --no-cache-dir -e .

COPY configs/ ./configs/
COPY app/     ./app/

# insightface downloads model weights to ~/.insightface on first run.
# Pre-download at build time so the container starts instantly.
# (Remove this RUN step if you prefer smaller images + internet at runtime.)
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1, det_size=(640,640)); \
print('ArcFace weights downloaded.')" || true

# Gradio binds to 0.0.0.0 inside the container; host maps 7860 → 7860
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

CMD ["python", "app/demo.py", "--host", "0.0.0.0", "--port", "7860"]
