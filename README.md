# FastSpeech 2: Non-Autoregressive Text-to-Speech for English and Telugu

A PyTorch implementation of FastSpeech 2 for non-autoregressive text-to-speech synthesis, developed and evaluated across two languages: **English** (using the AI4Bharat Svarah corpus) and **Telugu** (using the IndicTTS Telugu corpus).

The repository includes end-to-end pipelines for audio preprocessing, phonetic text processing (G2P), model training, variance adaptor visualization (duration, pitch, energy), and evaluation on targeted linguistic test cases.

---

## Repository Structure

```text
.
├── English/
│   ├── config.yaml              # Hyperparameters and audio settings
│   ├── preprocess.py            # Audio extraction, mel/F0/energy processing
│   ├── train.py                 # FastSpeech 2 training loop with loss logging
│   ├── inference.py             # Audio synthesis and variance contour plotting
│   └── checkpoints/             # Saved model weights (.pth)
├── Telugu/
│   ├── config.yaml              # Hyperparameters and audio settings
│   ├── preprocess.py            # Akshara parsing, mel/F0/energy processing
│   ├── train.py                 # FastSpeech 2 training loop with loss logging
│   ├── inference.py             # Audio synthesis and variance contour plotting
│   └── checkpoints/             # Saved model weights (.pth)
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/       # E1.wav to E4.wav
│   │   └── variance_adaptor_plots/      # Duration, pitch, and energy plots
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/       # T1.wav to T4.wav
│       └── variance_adaptor_plots/      # Duration, pitch, and energy plots
├── Report/
│   └── Technical_Report.pdf     # Detailed experimental analysis report
├── core/
│   ├── model.py                 # FastSpeech 2 architecture, FFT, Variance Adaptor
│   ├── audio.py                 # Mel-spectrogram, F0 (YIN/PyIN), Griffin-Lim
│   ├── dataset.py               # Dataset class and collate functions
│   ├── text_en.py               # English G2P and ARPAbet symbol dictionary
│   └── text_te.py               # Telugu Akshara decomposition and G2P
├── requirements.txt
└── README.md
```

---

## Model Architecture

The implementation follows the FastSpeech 2 architecture ([Ren et al., 2020](https://arxiv.org/abs/2006.04558)):

1. **Feed-Forward Transformer (FFT) Blocks:**
   - Multi-Head Attention (2 attention heads, hidden dimension 256).
   - 1D Convolutional Positionwise Feed-Forward Network (filter channel size 1024, kernel size 9 and 1).
   - Sinusoidal positional encodings applied at encoder and decoder inputs.

2. **Variance Adaptor:**
   - **Duration Predictor:** 2-layer 1D convolution with ReLU, LayerNorm, and Dropout (0.5), mapped through a Length Regulator to expand phoneme hidden representations to frame length.
   - **Pitch Predictor:** Frame-level fundamental frequency ($F_0$) extracted via YIN/PyIN, linearly quantized into 256 bins and embedded into the hidden representations.
   - **Energy Predictor:** Frame-level $L_2$-norm of STFT magnitudes, quantized into 256 bins and embedded into the representations.

3. **Mel Decoder & PostNet:**
   - 4-layer FFT decoder projecting hidden states to 80-channel mel-spectrograms.
   - 5-layer 1D convolutional PostNet with residual connections for high-frequency mel refinement.
   - Waveform inversion using Griffin-Lim phase reconstruction.

4. **Grapheme-to-Phoneme (G2P) Frontends:**
   - **English:** ARPAbet conversion via `g2p_en` with custom text normalization for numerals, dates, and split compound tokens.
   - **Telugu:** Custom akshara parser resolving base vowels (*achchulu*), consonants (*hallulu*), vowel signs (*gunintalu*), virama (*pollu*), anusvara, and visarga with Telugu number and timestamp expansion.

---

## Setup & Installation

### Requirements
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop GPU, CUDA 12.1)

```bash
# Clone the repository
git clone https://github.com/LightningHat/FastSpeech-2-Based-Text-to-Speech-Synthesis.git
cd FastSpeech-2-Based-Text-to-Speech-Synthesis

# Set up virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipelines

### 1. English Pipeline

```bash
# Preprocessing: Extract mel-spectrograms, F0 pitch, and energy from Svarah
python English/preprocess.py --config English/config.yaml

# Training: Train the FastSpeech 2 English model
python English/train.py --config English/config.yaml --epochs 25 --batch_size 16

# Inference: Synthesize all test sentences and export variance plots
python English/inference.py --config English/config.yaml
```

### 2. Telugu Pipeline

```bash
# Preprocessing: Akshara parsing, mel extraction, and pitch/energy extraction
python Telugu/preprocess.py --config Telugu/config.yaml

# Training: Train the FastSpeech 2 Telugu model
python Telugu/train.py --config Telugu/config.yaml --epochs 25 --batch_size 16

# Inference: Synthesize all test sentences and export variance plots
python Telugu/inference.py --config Telugu/config.yaml
```

---

## Test Sentences & Evaluation

Synthesized speech samples are saved under `Results/<Language>/test_generated_audios/`, and corresponding variance adaptor contour plots are saved under `Results/<Language>/variance_adaptor_plots/`.

### English Test Set
| ID | Sentence | Evaluation Focus |
|:---|:---|:---|
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | Polysyllabic stability and rhythm |
| **E2 (a)** | *The orchestra's conduct during the ceremony was praised by the audience.* | Heteronym noun (`/ˈkɒndʌkt/`) |
| **E2 (b)** | *The researchers will conduct an experiment to evaluate the proposed method.* | Heteronym verb (`/kənˈdʌkt/`) |
| **E3** | *The rhyth m of the orchestra changed unexpectedly.* | Split token recovery (`rhyth m` → `rhythm`) |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | Numerical, date/time normalization & code-mixed loanword |

### Telugu Test Set
| ID | Sentence | Evaluation Focus |
|:---|:---|:---|
| **T1** | *న్యూరోఫార్మకోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.* | Long Sanskrit/English loanwords, complex conjuncts |
| **T2 (a)** | *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* | Geminate stop context 1 (*grip*) |
| **T2 (b)** | *ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.* | Geminate stop context 2 (*silk*) |
| **T3** | *పులి అడవిలో నిశ్శబ్దంగా నడిచింది.* | Short narrative prosody with split token repair |
| **T4** | *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉండండి.* | Telugu numeral, date, and clock time expansion |

---

## Checkpoints

Model checkpoints (`best_model.pth` and periodic epoch checkpoints) are saved directly in `English/checkpoints/` and `Telugu/checkpoints/`.

> **Note:** Because PyTorch checkpoint state dictionaries exceed GitHub's 100 MB file limit, `*.pth` files are omitted from git tracking via `.gitignore`. The inference scripts automatically load from `checkpoints/best_model.pth` once training completes, or you can supply custom paths via `--checkpoint <path>`.

---

## Technical Report

For detailed analysis on architectural design choices, loss convergence curves, variance predictor contour dynamics, and cross-linguistic acoustic comparisons, refer to:

- [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)

---

## References

1. Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, T.-Y. Liu, "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", *arXiv:2006.04558*, 2020.
2. AI4Bharat Svarah Indian English Speech Corpus.
3. IndicTTS Database, Speech and Language Technologies Lab, IIT Madras.
