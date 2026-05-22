# Face Verification Biometric

Comparing classical and deep learning approaches to face verification, and testing how well each holds up under degraded image conditions.

---

## What this project does

The main question is: given two face photos, are they the same person?

Three models are compared:

| | Classical | Deep |
|---|---|---|
| **Model** | LBP + cosine similarity | FaceNet · ArcFace |
| **How it works** | Texture histogram over a 4×4 grid | 512-d embeddings from a neural network |
| **Training data** | None | VGGFace2 · MS1Mv2 |
| **Similarity** | Cosine | Cosine |

---

## Results on LFW

LFW (Labeled Faces in the Wild) — 6,000 pairs, 3,000 same person and 3,000 different people.

| Model | EER (%) ↓ | AUC ↑ | TAR@FAR=0.1% ↑ |
|---|---|---|---|
| LBP + cosine | 45.6 | 0.561 | 0.9% |
| FaceNet (VGGFace2) | 4.6 | 0.979 | 93.3% |
| ArcFace (buffalo_l) | 2.8 | 0.987 | 96.9% |

EER is the main metric here — it measures the point where false accepts and false rejects are equal. Lower is better. Accuracy alone is not very useful for this kind of task since it changes depending on what threshold you pick.

---

## Robustness under image degradation

We also tested how each model performs when images are degraded. For each type, we ran 6 intensity levels and measured EER at each step.

### Blur (sigma 0 → 5)

![Blur robustness](plots/robustness_blur.png)

- LBP is not very affected by blur since it works on relative pixel comparisons anyway
- FaceNet and ArcFace drop more at high blur because face detection starts to fail

### Low light (gamma 1 → 5)

![Low-light robustness](plots/robustness_low_light.png)

- All three models get worse with heavy darkening
- ArcFace holds up better than FaceNet
- LBP uses CLAHE preprocessing which helps a bit with lighting

### Sketch-style transform (blend 0 → 1)

![Sketch robustness](plots/robustness_sketch.png)

- This simulates matching a photo against a forensic sketch
- LBP struggles most since the sketch removes the texture it relies on
- ArcFace is surprisingly robust — still performs well even at full sketch intensity

---

## Project structure

```
face-verification-biometric/
├── configs/
│   └── experiment.yaml          # paths and model settings
├── data/
│   └── lfw/                     # LFW dataset goes here
├── src/fvbio/
│   ├── datasets/                # data loaders
│   ├── models/                  # LBP, FaceNet, ArcFace
│   ├── degradation/             # blur, low-light, sketch transforms
│   ├── metrics/                 # EER, TAR@FAR, ROC, AUC
│   └── evaluation/              # benchmark and robustness sweep logic
├── experiments/
│   ├── run_benchmark.py         # run full benchmark
│   └── run_robustness.py        # run degradation sweep
├── app/
│   └── demo.py                  # Gradio demo
├── tests/
│   └── test_metrics.py          # unit tests for metrics
└── results/                     # saved outputs
```

---

## Setup

### 1. Install

```bash
pip install -e .
```

### 2. Get the dataset

Download the LFW deepfunneled dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) and place it under `data/lfw/`:

```
data/lfw/
  pairs.txt
  George_W_Bush/
    George_W_Bush_0001.jpg
    ...
```

### 3. Run benchmark

```bash
python experiments/run_benchmark.py
```

Results saved to `results/benchmark_summary.json`, plots to `plots/`.

### 4. Run robustness sweep

```bash
python experiments/run_robustness.py
```

For deep models on CPU, use `--max-pairs 500` to keep it manageable.

### 5. Launch demo

```bash
python app/demo.py
# opens at http://localhost:7860
```

---

## Metrics used

- **EER** — Equal Error Rate: the point where false accepts = false rejects
- **TAR@FAR=0.1%** — how many genuine pairs are correctly accepted when only 1 in 1000 impostors gets through
- **AUC** — area under the ROC curve

---

## References

- **LBP**: Ahonen et al., "Face Description with Local Binary Patterns," TPAMI 2006
- **FaceNet**: Schroff et al., "FaceNet: A Unified Embedding," CVPR 2015
- **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss," CVPR 2019
- **LFW**: Huang et al., "Labeled Faces in the Wild," UMass TR 2007
- **CFP**: Sengupta et al., "Frontal to Profile Face Verification," WACV 2016
