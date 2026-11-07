# FastSpeech 2: Non-Autoregressive Text-to-Speech for English and Telugu

A PyTorch implementation of FastSpeech 2 for non-autoregressive text-to-speech synthesis, developed and evaluated across two languages: **English** (using the AI4Bharat Svarah corpus) and **Telugu** (using the IndicTTS Telugu corpus).

The repository includes end-to-end pipelines for audio preprocessing, phonetic text processing (G2P), acoustic modeling, variance adaptor visualization (duration, pitch F0, energy), and evaluation on targeted linguistic benchmark sentences.

---

## Repository Structure

```text
.
├── English/
│   ├── config.yaml              # Hyperparameters and audio settings
│   ├── preprocess.py            # Audio extraction, mel/F0/energy processing
│   ├── train.py                 # FastSpeech 2 training loop with loss logging
│   ├── inference.py             # Audio synthesis and variance contour plotting
│   └── checkpoints/             # Directory for saved model weights (.pth)
├── Telugu/
│   ├── config.yaml              # Hyperparameters and audio settings
│   ├── preprocess.py            # Akshara parsing, mel/F0/energy processing
│   ├── train.py                 # FastSpeech 2 training loop with loss logging
│   ├── inference.py             # Audio synthesis and variance contour plotting
│   └── checkpoints/             # Directory for saved model weights (.pth)
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
│   └── Technical_Report.pdf     # Publication-grade technical report
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

## Model Architecture & Telugu Adaptations

The architecture follows FastSpeech 2 ([Ren et al., 2020](https://arxiv.org/abs/2006.04558)):

1. **Feed-Forward Transformer (FFT) Blocks:**
   - Multi-Head Attention (2 attention heads, hidden dimension 256, projection dimension 128).
   - 1D Convolutional Positionwise Feed-Forward Network (filter channel size 1024, kernel sizes 9 and 1, ReLU activation).
   - Attention masks use `torch.finfo(attn.dtype).min` for numerical stability under mixed-precision training.

2. **Variance Adaptor:**
   - **Duration Predictor:** 2-layer 1D convolution with LayerNorm, Dropout (0.5), and minimum duration clamp (`min=1`).
   - **Length Regulator:** Expands phoneme representations to mel frame length based on predicted or ground-truth durations.
   - **Pitch Predictor:** Frame-level fundamental frequency ($F_0$) extracted via YIN/PyIN, quantized into 256 linear bins with learned embeddings.
   - **Energy Predictor:** Frame-level $L_2$-norm of STFT magnitudes, quantized into 256 bins with learned embeddings.

3. **Mel Decoder & PostNet:**
   - 4-layer FFT decoder projecting hidden representations into 80-channel mel-spectrograms.
   - 5-layer 1D convolutional PostNet with residual connections for high-frequency spectral refinement.
   - Waveform synthesis via Griffin-Lim phase estimation (default).

4. **Telugu-Specific Adaptations:**
   - Unlike Latin orthographies, Telugu is an alpha-syllabic (abugida) Dravidian script where consonants carry an inherent vowel `/a/`.
   - **Akshara Parsing:** Custom phonetic engine (`core/text_te.py`) that separates base consonants, applies vowel diacritics (*gunintalu*), and suppresses inherent vowels upon encountering a virama (*pollu* ్).
   - **Phonemic Inventory:** 67 discrete phonetic symbols covering all 16 vowels (*achchulu*), 35 consonants (*hallulu*), anusvara, visarga, and punctuation.
   - **Text Normalization:** Dedicated expansion for Telugu numerals, calendar dates, clock timestamps, and typographic compound token repairs.

---

## Dataset Selection & Subsampling Strategy

Both models are trained on separate ~5-hour training and ~1-hour validation splits with **zero overlap**:

- **English (AI4Bharat Svarah):**
  - Indian English multi-speaker corpus.
  - Subsampled into ~5.0 hours train and ~1.0 hour validation with strictly disjoint utterance partitions.
  - Preprocessing prunes clips $<1.0\text{ s}$ or $>12.0\text{ s}$, applies peak normalization, and resamples audio to 22,050 Hz.
- **Telugu (SPRING Lab IndicTTS):**
  - Native Telugu high-fidelity speech corpus.
  - Partitioned into ~5.0 hours train and ~1.0 hour validation with disjoint utterances.
  - Text is pre-processed through the Unicode Akshara decomposition engine to generate discrete phoneme sequence targets.

---

## Setup & Installation

### Requirements
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support

```bash
# Clone repository
git clone https://github.com/LightningHat/FastSpeech-2-Based-Text-to-Speech-Synthesis.git
cd FastSpeech-2-Based-Text-to-Speech-Synthesis

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Pipeline Execution

### 1. English Pipeline

```bash
# Preprocessing (extracts 80-mel, F0, energy, durations):
python English/preprocess.py --config English/config.yaml

# Training (trains FastSpeech 2 English model):
python English/train.py --config English/config.yaml --epochs 35 --batch_size 16

# Inference (synthesizes all 5 benchmark sentences and variance plots):
python English/inference.py --config English/config.yaml --test
```

### 2. Telugu Pipeline

```bash
# Preprocessing (akshara parsing, feature extraction):
python Telugu/preprocess.py --config Telugu/config.yaml

# Training (trains FastSpeech 2 Telugu model):
python Telugu/train.py --config Telugu/config.yaml --epochs 35 --batch_size 16

# Inference (synthesizes all 5 benchmark sentences and variance plots):
python Telugu/inference.py --config Telugu/config.yaml --test
```

---

## Benchmark Test Sentences

Synthesized audio files are saved under `Results/<Language>/test_generated_audios/`, and variance adaptor contour plots are saved under `Results/<Language>/variance_adaptor_plots/`.

### English Test Set
| ID | Exact Test Sentence | Linguistic Focus |
|:---|:---|:---|
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | Polysyllabic rhythm & phoneme stability |
| **E2 (a)** | *The orchestra's conduct during the ceremony was praised by the audience.* | Heteronym noun (`/ˈkɒndʌkt/`) initial stress |
| **E2 (b)** | *The researchers will conduct an experiment to evaluate the proposed method.* | Heteronym verb (`/kənˈdʌkt/`) second-syllable stress |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | Internal whitespace typo (`rhyth  m`) recovery |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | Alphanumerics, clock time, & code-mixed loanword |

### Telugu Test Set
| ID | Exact Test Sentence | Linguistic Focus |
|:---|:---|:---|
| **T1** | *న్యూ రోఫార్మ కోలాజికల్  పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని  విశ్లేషించారు.* | Long Sanskrit/English loanword conjuncts |
| **T2 (a)** | *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* | Geminate retroflex stop /ʈʈ/ (*grip*) |
| **T2 (b)** | *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చ ింది .* | Compound noun modifier (*silk saree*) & spacing |
| **T3** | *పులి అడ  విలో నిశ్శ బ్దంగా నడిచింది.* | Split akshara tokens & narrative prosody |
| **T4** | *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.* | Telugu numerals, date/time, & clause cadence |

---

## Checkpoints & Pretrained Models

Because model checkpoint state dictionaries (~116 MB each) exceed GitHub's 100 MB upload limit, `.pth` files are omitted from git tracking via `.gitignore`. 

Checkpoints and the complete results bundle are saved to Google Drive:

- **Google Drive Storage Directory:** `MyDrive/FastSpeech2_Checkpoints/`
- **Stored Artifacts:**
  - `English/checkpoints/best_model.pth` (116 MB)
  - `Telugu/checkpoints/best_model.pth` (116 MB)
  - `Honest_Results_Bundle.zip` (8.3 MB bundle of all audio, plots, and manifests)

To run inference using external checkpoints:
```bash
python English/inference.py --checkpoint path/to/best_model.pth --test
python Telugu/inference.py --checkpoint path/to/best_model.pth --test
```

---

## Technical Report

For in-depth analysis covering architecture, training dynamics, loss convergence curves, variance predictor contour dynamics, speech pronunciation quality, and cross-linguistic acoustic comparisons, refer to:

- [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)

---

## References

1. Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, T.-Y. Liu, "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", *arXiv:2006.04558*, 2020.
2. AI4Bharat Svarah Indian English Speech Corpus.
3. IndicTTS Database, Speech and Language Technologies Lab, IIT Madras.
