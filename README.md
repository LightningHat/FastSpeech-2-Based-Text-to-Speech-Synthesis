# FastSpeech 2 for English and Telugu TTS

This repository contains an implementation of FastSpeech 2 for English and Telugu text-to-speech synthesis, built for the take-home research assignment.

Two separate models were trained:
1. **English**: Trained on the AI4Bharat Svarah dataset (~5 hours train, ~1 hour validation).
2. **Telugu**: Trained on the IndicTTS Telugu dataset (~5 hours train, ~1 hour validation).

---

## Model Checkpoints (Google Drive)

Because each checkpoint file is ~116 MB and exceeds GitHub's 100 MB file limit, the trained model weights are hosted on Google Drive:

**[Google Drive Checkpoints Folder](https://drive.google.com/drive/folders/17G4YMY4ce79uIVG7E7gOVIq9E0f4qVMf?usp=drive_link)**

The Drive folder contains:
- `English/best_model.pth` (116 MB)
- `Telugu/best_model.pth` (116 MB)

### Where to place the downloaded checkpoints
After downloading, place the `.pth` files into their respective checkpoint folders:
```
English/checkpoints/best_model.pth
Telugu/checkpoints/best_model.pth
```

---

## Directory Structure

```
.
├── English/
│   ├── config.yaml              # Audio, model, and training settings
│   ├── preprocess.py            # Feature extraction (mel, pitch, energy)
│   ├── train.py                 # Training script
│   ├── inference.py             # Synthesis and variance adaptor plotting
│   └── checkpoints/             # Directory for best_model.pth
├── Telugu/
│   ├── config.yaml              # Telugu configuration
│   ├── preprocess.py            # Akshara parsing & feature extraction
│   ├── train.py                 # Training script
│   ├── inference.py             # Synthesis and variance adaptor plotting
│   └── checkpoints/             # Directory for best_model.pth
├── Results/
│   ├── English/
│   │   ├── training_loss.png    # Loss curves
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/   # E1.wav to E4.wav
│   │   └── variance_adaptor_plots/  # Duration, pitch, and energy plots
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/   # T1.wav to T4.wav
│       └── variance_adaptor_plots/  # Duration, pitch, and energy plots
├── Report/
│   └── Technical_Report.pdf     # Technical report
├── core/
│   ├── model.py                 # FastSpeech 2 PyTorch model
│   ├── audio.py                 # Mel spectrogram, YIN pitch, energy
│   ├── hifigan.py               # HiFi-GAN neural vocoder
│   ├── dataset.py               # PyTorch Dataset and collate fn
│   ├── text_en.py               # English phoneme conversion (g2p_en)
│   └── text_te.py               # Telugu akshara tokenizer
├── requirements.txt
├── make_report.py
├── report_notes.json
└── README.md
```

---

## Setup

### Environment setup
Python 3.10 or 3.11 with a CUDA-enabled GPU is recommended.

```bash
git clone https://github.com/LightningHat/FastSpeech-2-Based-Text-to-Speech-Synthesis.git
cd FastSpeech-2-Based-Text-to-Speech-Synthesis

python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running Inference

To run inference on the test sentences and generate the duration, pitch, and energy plots:

### English
```bash
python English/inference.py --config English/config.yaml --checkpoint English/checkpoints/best_model.pth
```

### Telugu
```bash
python Telugu/inference.py --config Telugu/config.yaml --checkpoint Telugu/checkpoints/best_model.pth
```

Output audio files will be written to `Results/English/test_generated_audios/` and `Results/Telugu/test_generated_audios/`. The corresponding variance adaptor plots are saved under `variance_adaptor_plots/`.

To test custom input text:
```bash
python English/inference.py --text "Text to speech synthesis using FastSpeech 2."
```

---

## Training and Preprocessing

If you want to preprocess the datasets or retrain from scratch:

### English
```bash
# Preprocess ~5h train and ~1h validation data:
python English/preprocess.py --config English/config.yaml

# Train model for 35 epochs:
python English/train.py --config English/config.yaml --epochs 35
```

### Telugu
```bash
# Preprocess IndicTTS Telugu:
python Telugu/preprocess.py --config Telugu/config.yaml

# Train model for 35 epochs:
python Telugu/train.py --config Telugu/config.yaml --epochs 35
```

---

## Test Sentences

The assignment test sentences evaluated in this project:

### English
- **E1**: *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.*
- **E2_a**: *The orchestra's conduct during the ceremony was praised by the audience.*
- **E2_b**: *The researchers will conduct an experiment to evaluate the proposed method.*
- **E3**: *The rhyth m of the orchestra changed unexpectedly.*
- **E4**: *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.*

### Telugu
- **T1**: *న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.*
- **T2_a**: *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.*
- **T2_b**: *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.*
- **T3**: *పులి అడవిలో నిశ్శబ్దంగా నడిచింది.*
- **T4**: *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.*

---

## Report

The full technical report detailing the model architecture, training observations, loss curves, and acoustic analysis is available at:
- [Report/Technical_Report.pdf](Report/Technical_Report.pdf)

To rebuild the report:
```bash
python make_report.py
```

---

## References

1. Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, T.-Y. Liu, "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", 2020.
2. Ming024 FastSpeech 2 PyTorch reference implementation: https://github.com/ming024/FastSpeech2
3. AI4Bharat Svarah Speech Dataset.
4. IndicTTS Database (IIT Madras).
