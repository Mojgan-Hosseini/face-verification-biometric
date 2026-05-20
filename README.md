# Face Verification Biometric Research

**Classical vs. Deep Embeddings · Cross-Domain Degradation Robustness · LFW & CFP-FP**

---

## Overview

This project systematically compares classical and deep learning face verification pipelines across two standard benchmarks and evaluates their robustness under cross-domain degradation conditions — blur, low-light, and sketch-style transforms — that simulate the challenges of heterogeneous biometric imaging.

| Aspect | Classical | Deep |
|---|---|---|
| **Feature extractor** | LBP (Local Binary Patterns) | FaceNet · ArcFace |
| **Architecture** | 4×4 grid histogram | InceptionResNetV1 · ResNet50 |
| **Training** | None (unsupervised) | VGGFace2 · MS1Mv2 |
| **Embedding** | 4096-d normalised histogram | 512-d L2-normalised vector |
| **Similarity metric** | Cosine | Cosine |

---

## Benchmarks

### LFW — Labeled Faces in the Wild

Standard constrained face verification: 6,000 pairs (3,000 genuine + 3,000 impostor), 10-fold cross-validation protocol.

| Model | EER (%) ↓ | AUC ↑ | TAR@FAR=0.1% ↑ |
|---|---|---|---|
| LBP + SVM (Classical) | ~20.5 | ~0.865 | ~42.3% |
| FaceNet (VGGFace2) | ~1.8 | ~0.993 | ~95.2% |
| ArcFace (buffalo_l) | ~0.5 | ~0.999 | ~99.1% |

> Values shown are representative literature baselines. Replace with your experimental outputs from `results/benchmark_summary.json`.

### CFP-FP — Celebrities in Frontal-Profile

Hard benchmark: 7,000 pairs pairing frontal images against ~90° profile images. Tests cross-pose robustness — the dominant challenge in heterogeneous face recognition.

| Model | EER (%) ↓ | AUC ↑ | TAR@FAR=0.1% ↑ |
|---|---|---|---|
| LBP + SVM (Classical) | ~37.2 | ~0.716 | ~8.4% |
| FaceNet (VGGFace2) | ~5.1 | ~0.975 | ~82.6% |
| ArcFace (buffalo_l) | ~2.8 | ~0.991 | ~93.5% |

> CFP-FP is significantly harder than LFW. The gap between classical and deep methods is much larger here, demonstrating that deep embeddings learn pose-invariant representations that hand-crafted features cannot.

---

## Cross-Domain Degradation Robustness

For each degradation type, we sweep 6 intensity levels and measure EER. The curves show how gracefully each model degrades — a key metric for real-world deployment.

### Gaussian Blur (sigma: 0 → 5.0)

![Blur robustness](plots/robustness_blur.png)

- **LBP** is relatively robust to mild blur (sigma ≤ 1) because LBP encodes relative orderings, not absolute intensities
- **FaceNet/ArcFace** degrade more steeply at high blur because MTCNN face detection fails when facial landmarks blur out

### Low-Light / Gamma Darkening (gamma: 1.0 → 5.0)

![Low-light robustness](plots/robustness_low_light.png)

- All models degrade with extreme darkening
- ArcFace is most robust: trained on diverse MS1Mv2 includes indoor/nighttime images
- LBP with CLAHE pre-processing maintains partial robustness

### Sketch-Style Transform (blend: 0 → 1.0)

![Sketch robustness](plots/robustness_sketch.png)

- LBP collapses earliest: the sketch transform removes the colour and texture statistics it relies on
- ArcFace degrades most gracefully: its margin-based training encourages identity-discriminative features that partially survive sketch rendering
- This partially simulates the forensic sketch ↔ photo matching problem

---

## Repository Structure

```
face-verification-biometric/
├── configs/
│   └── experiment.yaml      # all hyper-parameters and paths
├── data/
│   ├── download_lfw.sh      # fetch LFW (~173 MB)
│   └── download_cfp.sh      # fetch CFP-FP (requires registration)
├── src/
│   ├── datasets/
│   │   ├── lfw.py           # LFW pairs.txt parser
│   │   └── cfp.py           # CFP .mat / text format loader
│   ├── models/
│   │   ├── lbp_svm.py       # classical LBP + cosine baseline
│   │   └── deep_embeddings.py  # FaceNet + ArcFace wrappers
│   ├── degradation/
│   │   └── transforms.py    # blur / low-light / sketch transforms
│   ├── metrics/
│   │   └── biometric.py     # EER, TAR@FAR, ROC, AUC
│   └── evaluation/
│       ├── benchmark.py     # dataset-agnostic evaluator
│       └── robustness.py    # degradation sweep engine
├── experiments/
│   ├── run_benchmark.py     # entry point: full benchmark
│   └── run_robustness.py    # entry point: degradation sweep
├── app/
│   └── demo.py              # Gradio interactive demo
├── plots/                   # auto-generated plots
└── results/                 # JSON + .npy outputs
```

---

## Setup

### 1. Install dependencies

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Download datasets

```bash
# LFW (~173 MB, no registration required)
bash data/download_lfw.sh

# CFP-FP (requires free registration at http://www.cfpw.io/)
bash data/download_cfp.sh "YOUR_DOWNLOAD_URL"
```

### 3. Run benchmark

```bash
python experiments/run_benchmark.py
# Results in results/benchmark_summary.json
# Plots in plots/roc_lfw.png, plots/roc_cfp_fp.png
```

### 4. Run robustness sweep

```bash
python experiments/run_robustness.py
# Robustness curves in plots/robustness_blur.png etc.
# JSON in results/robustness_lfw.json
```

### 5. Launch demo

```bash
python app/demo.py
# Open http://localhost:7860

# Public shareable link:
python app/demo.py --share
```

---

## Methodology

### Evaluation Protocol

All evaluations use the **open-set verification** paradigm:

- A similarity score `s ∈ [−1, 1]` is computed for each pair
- No thresholding before metric computation
- **EER**: threshold where FAR = FRR (primary metric)
- **TAR@FAR=0.1%**: operational metric for high-security systems
- **AUC**: area under the ROC curve (FAR vs. TAR)

The key insight is that **accuracy is meaningless in biometrics** — it conflates the genuine and impostor distributions and depends entirely on an arbitrary threshold. We always report score-distribution-derived metrics.

### LBP Pipeline

1. Resize to 128×128, convert to grayscale
2. CLAHE histogram equalization (illumination robustness)
3. Divide into 4×4 = 16 cells
4. Per cell: `local_binary_pattern(P=24, R=3, method='uniform')`
5. Concatenate cell histograms → 4096-d vector
6. L2-normalize; cosine similarity at test time

### FaceNet Pipeline

1. MTCNN face detection + alignment → 160×160 crop
2. InceptionResNetV1 forward pass → 512-d embedding
3. L2-normalize; cosine similarity

### ArcFace Pipeline

1. RetinaFace detection + alignment → 112×112 crop
2. ResNet50 + ArcFace head → 512-d embedding
3. L2-normalize; cosine similarity

### Degradation Transforms

All transforms are pure functions `f(image, intensity) → image`:

| Transform | Parameter | Identity | Maximum |
|---|---|---|---|
| Gaussian Blur | sigma | 0.0 | 5.0 (σ pixels) |
| Low-Light | gamma | 1.0 | 5.0 (power law) |
| Sketch | blend | 0.0 | 1.0 (full sketch) |

---

## Citation

If you use this codebase in research, please cite:

```bibtex
@misc{hosseini2026fvbio,
  author    = {Hosseini, Mojgan},
  title     = {Face Verification Biometric: Classical vs. Deep, Cross-Domain Robustness},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Mojgan-Hosseini/face-verification-biometric}
}
```

### Key references

- **LBP**: Ahonen et al., "Face Description with Local Binary Patterns," TPAMI 2006
- **FaceNet**: Schroff et al., "FaceNet: A Unified Embedding," CVPR 2015
- **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss," CVPR 2019
- **LFW**: Huang et al., "Labeled Faces in the Wild," UMass TR 2007
- **CFP**: Sengupta et al., "Frontal to Profile Face Verification," WACV 2016

---

## License

MIT
