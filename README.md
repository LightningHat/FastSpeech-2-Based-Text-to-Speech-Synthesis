# FastSpeech 2: Text-to-Speech Synthesis for English and Telugu

This repository contains the complete implementation, training pipeline, acoustic feature extraction, and evaluation suite for **FastSpeech 2-Based Text-to-Speech Synthesis**, developed separately for **English** (using the *AI4Bharat Svarah* dataset) and **Telugu** (using the *SPRING Lab IndicTTS Telugu* dataset).

---

## 1. Project Directory Structure

The repository maintains the exact submission structure specified in the assignment:

```text
project/
├── English/
│   ├── train.py                 # Complete training loop with loss logging
│   ├── preprocess.py            # Svarah extraction & feature extraction
│   ├── inference.py             # Inference & variance adaptor plotting
│   ├── config.yaml              # Hyperparameters and audio settings
│   └── checkpoints/             # Saved model checkpoints
├── Telugu/
│   ├── train.py                 # Complete training loop with loss logging
│   ├── preprocess.py            # IndicTTS extraction & akshara decomposition
│   ├── inference.py             # Inference & variance adaptor plotting
│   ├── config.yaml              # Hyperparameters and audio settings
│   └── checkpoints/             # Saved model checkpoints
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/       # E1.wav, E2_a.wav, E2_b.wav, E3.wav, E4.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/                # Per-phoneme predicted duration bar plots
│   │       ├── pitch/                   # Predicted F0 pitch contours
│   │       └── energy/                  # Predicted frame energy contours
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/       # T1.wav, T2_a.wav, T2_b.wav, T3.wav, T4.wav
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf     # Publication-quality technical report
├── README.md
└── requirements.txt
```

---

## 2. Architecture & Key Features

* **Feed-Forward Transformer (FFT) Blocks:** Multi-Head Attention (2 heads, hidden dim 256) + 1D Convolutional Positionwise Feed-Forward Network (filter dim 1024, kernel sizes `[9, 1]`).
* **Variance Adaptor:**
  * **Duration Predictor & Length Regulator:** 2-layer Conv1D with LayerNorm/Dropout (0.5) predicting log-frame duration per phoneme.
  * **Pitch Predictor:** Frame-level $F_0$ pitch extraction via YIN/PyIN, quantized into 256 linear bins with learned embeddings.
  * **Energy Predictor:** Frame-level $L_2$-norm of STFT magnitudes, quantized into 256 bins with learned embeddings.
* **PostNet:** 5-layer 1D Conv residual network with Tanh activation and BatchNorm to refine high-frequency mel harmonics.
* **Telugu Phonetic Engine (G2P):** Custom akshara parser handling vowels (*achchulu*), consonants (*hallulu*), matras (*gunintalu*), virama (*pollu*), anusvara, and visarga with 100% vocabulary coverage.
* **Text Normalization:** Automated expansion for English and Telugu numerals, calendar dates, timestamps (`10:45 AM` -> *ten forty-five AM* / *పది నలభై ఐదు*), and split compound token correction.

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10 or 3.11
* CUDA-enabled GPU (tested on NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.1)

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Guide

### English Model Pipeline

```bash
# Preprocess Svarah dataset (~5h train, ~1h val split with zero overlap)
python English/preprocess.py --config English/config.yaml

# Train FastSpeech 2 English model
python English/train.py --config English/config.yaml --epochs 25 --batch_size 16

# Run inference on all 5 English test sentences (generates audio & variance plots)
python English/inference.py --config English/config.yaml
```

### Telugu Model Pipeline

```bash
# Preprocess IndicTTS Telugu dataset (~5h train, ~1h val split with zero overlap)
python Telugu/preprocess.py --config Telugu/config.yaml

# Train FastSpeech 2 Telugu model
python Telugu/train.py --config Telugu/config.yaml --epochs 25 --batch_size 16

# Run inference on all 5 Telugu test sentences (generates audio & variance plots)
python Telugu/inference.py --config Telugu/config.yaml
```

---

## 5. Test Sentences & Evaluation

Speech has been generated for all required test sentences with variance adaptor plots saved in `Results/`:

### English Test Sentences
* **E1:** *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.*
* **E2 (a):** *The orchestra's conduct during the ceremony was praised by the audience.*
* **E2 (b):** *The researchers will conduct an experiment to evaluate the proposed method.*
* **E3:** *The rhyth m of the orchestra changed unexpectedly.*
* **E4:** *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.*

### Telugu Test Sentences
* **T1:** *న్యూరోఫార్మకోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.*
* **T2 (a):** *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.*
* **T2 (b):** *ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.*
* **T3:** *పులి అడవిలో నిశ్శబ్దంగా నడిచింది.*
* **T4:** *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉండండి.*

---

## 6. Model Checkpoints & Download Links

Due to GitHub's 100 MB per-file upload limit, the full model checkpoints (`best_model.pth`, ~346 MB each) can either be tracked via **Git LFS** or downloaded from the external storage links below:

* **English Checkpoint (`best_model.pth`):** [Download Link / Hugging Face / Google Drive](https://drive.google.com/)
* **Telugu Checkpoint (`best_model.pth`):** [Download Link / Hugging Face / Google Drive](https://drive.google.com/)

*(To use Git LFS: `git lfs track "*.pth"` followed by `git add .gitattributes` and `git commit`)*.

---

## 7. Technical Report

The complete academic analysis report is available at:
👉 **[`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)**

It includes in-depth discussion on:
1. Architecture components & modifications from `ming024/FastSpeech2`.
2. Dataset selection, subsampling, and validation split design.
3. Loss curves analysis, stability, and convergence behavior.
4. Qualitative and quantitative speech observations on all 10 test cases.
5. In-depth analysis of variance adaptor plots (duration, pitch, energy).
6. Cross-linguistic comparative observations (stress-timed vs. syllable-timed dynamics, Dravidian phonology, limitations).
