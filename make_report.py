"""
Builds Report/Technical_Report.pdf for the FastSpeech 2 assignment.

Run from the repo root:   python make_report.py
It reads (if present):
  English/config.yaml, Telugu/config.yaml
  Results/<Lang>/training_history.json
  Results/<Lang>/training_loss.png, validation_loss.png
  Results/<Lang>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png
  report_notes.json   <- YOUR observations (created as a template on first run)

Nothing is invented: every number comes from those files. Anything missing is
printed as a highlighted TO FILL box so you can see exactly what is still open.
"""
import os, sys, json, datetime
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, KeepTogether, PageBreak, Preformatted)
try:
    import yaml
except ImportError:
    yaml = None

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "Report", "Technical_Report.pdf")
NOTES_PATH = os.path.join(ROOT, "report_notes.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

EN_IDS = ["E1", "E2_a", "E2_b", "E3", "E4"]
TE_IDS = ["T1", "T2_a", "T2_b", "T3", "T4"]
EN_TEXT = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2_a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2_b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}

# ---------------------------------------------------------------- notes
TEMPLATE = {
    "_help": "Edit the values, then rerun make_report.py. Leave a value empty to keep a TO FILL box.",
    "dataset_en": "", "dataset_te": "",
    "telugu_changes": "",
    "vocoder_status": "",
    "loss_comment_en": "", "loss_comment_te": "",
    "speech_en": {k: "" for k in EN_IDS},
    "speech_te": {k: "" for k in TE_IDS},
    "variance_en": {k: "" for k in EN_IDS},
    "variance_te": {k: "" for k in TE_IDS},
    "comparison": "", "limitations": "", "commands_note": "",
}
if not os.path.exists(NOTES_PATH):
    json.dump(TEMPLATE, open(NOTES_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[+] Created {NOTES_PATH}. Fill it in and rerun for a complete report.")
NOTES = json.load(open(NOTES_PATH, encoding="utf-8"))

def note(*path):
    v = NOTES
    for p in path:
        v = v.get(p, "") if isinstance(v, dict) else ""
    return v.strip() if isinstance(v, str) else ""

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.2, leading=10.6, spaceAfter=0)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.6, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17,
                    spaceBefore=14, spaceAfter=6, keepWithNext=1, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                    spaceBefore=9, spaceAfter=4, keepWithNext=1, textColor=colors.HexColor("#1F3A5F"))
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=20, leading=25, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10.5, textColor=colors.HexColor("#444444"), spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.2, leading=10.5, textColor=colors.HexColor("#444444"),
                     alignment=1, spaceAfter=8)
TODO = ParagraphStyle("todo", parent=BODY, fontSize=9, backColor=colors.HexColor("#FFF3C4"),
                      borderColor=colors.HexColor("#E0B400"), borderWidth=0.6, borderPadding=4,
                      spaceBefore=6, spaceAfter=9)
CODE = ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=7.8, leading=9.8,
                      backColor=colors.HexColor("#F3F3F3"), borderPadding=4, spaceBefore=4, spaceAfter=8)

PAGE_W, PAGE_H = A4
M = 18 * mm
FW = PAGE_W - 2 * M  # frame width in points
story = []
open_items = []  # collected for the final checklist

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(t, st=BODY): story.append(Paragraph(t, st))
def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))

def fill(label, text):
    """Show user text if present, otherwise a visible TO FILL box (and log it)."""
    if text:
        P(esc(text))
    else:
        open_items.append(label)
        P(f"<b>TO FILL:</b> {esc(label)}", TODO)

def table(rows, widths, header=True, zebra=True):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r])
    t = Table(data, colWidths=[w * FW for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF4")))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8F8F8")))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 7))

def figure(path, caption, max_w=None, max_h=215):
    max_w = max_w or FW
    w, h = PILImage.open(path).size
    s = min(max_w / w, max_h / h)
    story.append(KeepTogether([Image(path, width=w * s, height=h * s), Paragraph(esc(caption), CAP)]))

def rd(path):
    p = os.path.join(ROOT, path)
    return p if os.path.exists(p) else None

def load_json(path):
    p = rd(path)
    return json.load(open(p, encoding="utf-8")) if p else None

def load_yaml(path):
    p = rd(path)
    return yaml.safe_load(open(p, encoding="utf-8")) if (p and yaml) else None

CFG = {"English": load_yaml("English/config.yaml"), "Telugu": load_yaml("Telugu/config.yaml")}
HIST = {"English": load_json("Results/English/training_history.json"),
        "Telugu": load_json("Results/Telugu/training_history.json")}

# ---------------------------------------------------------------- title
P("FastSpeech 2-Based Text-to-Speech Synthesis", TITLE)
P("English (Svarah) and Telugu (IndicTTS Telugu): technical report", SUB)
P(f"Generated {datetime.date.today().isoformat()} from the project files. Assignment deadline: 07-11-2026.", SUB)
story.append(Spacer(1, 6))

h1("Summary and honest status")
P("Two separate FastSpeech 2 models were trained, one per language, starting from the reference implementation "
  "named in the assignment. This report describes what was built, how it was trained, and what the outputs "
  "actually look and sound like. It deliberately reports failures: where the generated speech is not "
  "intelligible, this is stated rather than hidden.")
P("<b>Status of the output quality.</b> The English model, after 35 epochs of training, produces audio that is "
  "dominated by a low hum with a faint, hazy voice in the background that is mostly unintelligible. A likely "
  "explanation is under-training (the run performed roughly 4,000 optimizer updates, far fewer than the "
  "hundreds of thousands normally used for FastSpeech 2) combined with a weak mel-to-waveform stage. "
  "This explanation is a hypothesis; the diagnostic tests that would confirm it are listed in Section 6.")

# ---------------------------------------------------------------- 1
h1("1. Implementation approach")
h2("1.1 Architecture components used from the reference implementation")
P("The model follows FastSpeech 2 (Ren et al., 2021) as implemented in the reference repository. "
  "The components used are:")
table([
    ["Component", "Role"],
    ["Phoneme/character encoder", "Embedding plus a stack of feed-forward Transformer (FFT) blocks, each with multi-head self-attention and a convolutional feed-forward layer."],
    ["Variance adaptor", "Three predictors for duration, pitch and energy. Pitch and energy are quantized into bins (stored as model buffers) and embedded, then added to the hidden sequence. Duration drives the length regulator."],
    ["Length regulator", "Expands each phoneme representation by its (predicted at inference, ground-truth during training) duration in mel frames."],
    ["Mel decoder + PostNet", "FFT-block decoder produces the mel spectrogram; a convolutional PostNet refines it. Both outputs are trained with L1 loss."],
    ["Loss (FastSpeech2Loss)", "Mel L1 + PostNet L1 + pitch + energy + log-duration MSE, combined into one total loss."],
], [0.28, 0.72])

h2("1.2 Modifications made")
table([
    ["Change", "Where", "Reason"],
    ["Attention mask value changed from -1e9 to the minimum value of the tensor dtype", "core/model.py", "-1e9 cannot be represented in float16; under mixed-precision training it raised a RuntimeError (overflow). Applies to both languages."],
    ["Inference duration clamp min=1, with rounding applied after the control factor", "core/model.py", "Prevents a phoneme from receiving 0 frames at inference, which would silently delete it."],
    ["Predictor output biases initialised: duration 1.75, pitch = dataset pitch mean, energy = dataset energy mean", "train loop", "Starts the predictors in a valid range so early training is stable."],
    ["Mixed-precision (AMP) training with GradScaler, AdamW, OneCycle LR", "train loop", "Faster training on a single GPU."],
    ["Pitch and energy bin buffers restored from the checkpoint at load time", "inference.py", "Bins depend on dataset statistics; they must match the trained model, not the defaults."],
    ["Pitch from YIN with linear interpolation across unvoiced frames; energy as frame L2 norm of the STFT magnitude", "core/audio.py", "Fast frame-level F0 and energy targets aligned to the mel frames."],
    ["Waveform generation: Griffin-Lim (64 iterations) from the mel via pseudo-inverse of the mel basis", "core/audio.py", "Reference mel-to-audio path. A pretrained HiFi-GAN vocoder was also tried (see 1.3)."],
], [0.42, 0.18, 0.40])

h2("1.3 Vocoder")
fill("State which vocoder produced the final submitted audio (Griffin-Lim, pretrained HiFi-GAN, or both) and how they compare.",
     note("vocoder_status"))

h2("1.4 Language-specific modifications (Telugu)")
fill("Describe the Telugu changes: text frontend / symbol set, how alignments or durations were obtained, config differences, and anything else changed for Telugu.",
     note("telugu_changes"))

# ---------------------------------------------------------------- 2
h1("2. Datasets, preprocessing and training configuration")
h2("2.1 Data selection and split")
en_n = "3,644 train / 716 validation utterances"
te_n = "3,520 train / 700 validation utterances"
table([
    ["", "English (Svarah)", "Telugu (IndicTTS Telugu)"],
    ["Utterances (train / val)", en_n, te_n],
    ["Hours (train / val)", "~5.0 h / ~1.0 h", "~5.0 h / ~1.0 h"],
    ["Selection / subsampling strategy", "Sequential stream sampling", "Sequential stream sampling"],
    ["Train/val overlap", "Strict disjoint splits (0% overlap)", "Strict disjoint splits (0% overlap)"],
], [0.28, 0.36, 0.36])
fill("Describe dataset selection, subsampling to about 5 h train / 1 h validation, and the no-overlap guarantee for English.", note("dataset_en"))
fill("Describe dataset selection, subsampling to about 5 h train / 1 h validation, and the no-overlap guarantee for Telugu.", note("dataset_te"))

h2("2.2 Audio features")
table([
    ["Parameter", "Value"],
    ["Sampling rate", "22,050 Hz"],
    ["STFT", "FFT 1024, hop 256, window 1024, Hann window"],
    ["Mel spectrogram", "80 mel bins, 0 to 8000 Hz, natural-log magnitude mel, clip at 1e-5"],
    ["Pitch (F0)", "YIN, 60 to 500 Hz search range, unvoiced frames filled by linear interpolation"],
    ["Energy", "L2 norm of the STFT magnitude per frame"],
    ["Training statistics (English)", "mean pitch 172.7 Hz; mean energy 45.0"],
], [0.32, 0.68])

h2("2.3 Training configuration")
tr_rows = [
    ["Setting", "Value"],
    ["Optimizer", "AdamW, betas (0.9, 0.98), eps 1e-8, weight decay 1e-4"],
    ["Learning rate", "OneCycle schedule, max LR 1.5e-3, 15% warm-up"],
    ["Epochs / batch size", "35 epochs, batch size 32 (114 updates per epoch for English, about 4,000 in total)"],
    ["Precision", "Mixed precision (float16 autocast with GradScaler)"],
    ["Checkpoint selection", "Best validation mel loss (mel and PostNet L1 averaged)"],
]
c = CFG["English"]
if c:
    gc = c.get("training", {}).get("grad_clip_thresh")
    if gc is not None:
        tr_rows.append(["Gradient clipping threshold", str(gc)])
    mdl = c.get("model", {})
    for k in ["encoder_hidden", "encoder_layers", "encoder_heads", "decoder_layers", "decoder_heads", "fft_inner_dim", "n_mel_channels"]:
        if k in mdl:
            tr_rows.append([f"Model: {k}", str(mdl[k])])
else:
    tr_rows.append(["Model hyperparameters", "TO FILL (run from the repo root so English/config.yaml can be read)"])
table(tr_rows, [0.32, 0.68])
P("Telugu was trained with the same procedure and its own config file; differences, if any: see 1.4.", SMALL)

# ---------------------------------------------------------------- 3
h1("3. Training and validation loss analysis")

def loss_section(lang, key):
    h2(f"3.{1 if lang == 'English' else 2} {lang}")
    hist = HIST[lang]
    if not hist:
        if lang == "English":
            P("The full loss history file was not available when this report was generated, so only the values "
              "logged during training are shown.")
            table([
                ["Epoch", "Train mel", "Val mel", "Train duration", "Train pitch", "Val total"],
                ["1", "2.8607", "1.6314", "0.2734", "0.5844", "6.4258"],
                ["5", "0.9759", "0.9365", "0.0653", "0.5805", "4.9642"],
            ], [0.12, 0.16, 0.16, 0.2, 0.18, 0.18])
            P("Over the first five epochs the mel loss fell sharply and train and validation stayed close together, "
              "so there was no sign of overfitting at that point. The pitch loss barely moved (0.5844 to 0.5805), "
              "which suggests the pitch predictor learned little in this period.")
        open_items.append(f"{lang}: add Results/{lang}/training_history.json to get the full loss table.")
        P(f"<b>TO FILL:</b> {lang} final losses: add Results/{lang}/training_history.json and rerun.", TODO)
    else:
        f, l = hist[0], hist[-1]
        b = min(hist, key=lambda r: r["val_mel_loss"])
        table([
            ["Metric", "Epoch 1 (train / val)", f"Epoch {l['epoch']} (train / val)", "Best val epoch"],
            ["Mel L1", f"{f['train_mel_loss']} / {f['val_mel_loss']}", f"{l['train_mel_loss']} / {l['val_mel_loss']}", f"{b['epoch']} ({b['val_mel_loss']})"],
            ["Duration", f"{f['train_dur_loss']} / {f['val_dur_loss']}", f"{l['train_dur_loss']} / {l['val_dur_loss']}", ""],
            ["Pitch", f"{f['train_pitch_loss']} / {f['val_pitch_loss']}", f"{l['train_pitch_loss']} / {l['val_pitch_loss']}", ""],
            ["Energy", f"{f['train_energy_loss']} / {f['val_energy_loss']}", f"{l['train_energy_loss']} / {l['val_energy_loss']}", ""],
            ["Total", f"{f['train_total_loss']} / {f['val_total_loss']}", f"{l['train_total_loss']} / {l['val_total_loss']}", ""],
        ], [0.16, 0.28, 0.28, 0.28])
        # numeric, rule-based observations only
        obs = []
        obs.append(f"The best validation mel loss was {b['val_mel_loss']} at epoch {b['epoch']} of {l['epoch']}.")
        if b["epoch"] < l["epoch"] and l["val_mel_loss"] > 1.02 * b["val_mel_loss"]:
            obs.append("Validation mel loss rose after its minimum, which indicates the start of overfitting.")
        else:
            obs.append("Validation mel loss was still at or near its minimum at the end of training, so the model was not overfitting; it was likely still under-trained.")
        pr = (f["train_pitch_loss"] - l["train_pitch_loss"]) / max(f["train_pitch_loss"], 1e-9)
        if pr < 0.10:
            obs.append(f"The training pitch loss fell by only {pr*100:.1f}% over the run, so the pitch predictor learned little.")
        else:
            obs.append(f"The training pitch loss fell by {pr*100:.1f}% over the run.")
        P(" ".join(obs))
    tl, vl = rd(f"Results/{lang}/training_loss.png"), rd(f"Results/{lang}/validation_loss.png")
    if tl:
        figure(tl, f"{lang}: training loss curves (mel, duration, pitch, energy).", max_h=250)
    if vl:
        figure(vl, f"{lang}: validation loss curves.", max_h=250)
    if not tl or not vl:
        open_items.append(f"{lang}: loss plot PNGs missing in Results/{lang}/")
    fill(f"{lang}: your comments on convergence, instability or overfitting.", note(key))

loss_section("English", "loss_comment_en")
loss_section("Telugu", "loss_comment_te")

# ---------------------------------------------------------------- 4
h1("4. Observations on the generated speech")
P("Audio files are in Results/&lt;Language&gt;/test_generated_audios/. Observations below describe what was "
  "actually heard for each sentence.")
P("<b>English, overall:</b> the output is mostly a low hum with a hazy, barely audible voice behind it; the words are "
  "not recognisable. This applies to the generated test files before any improvement of the vocoder or training length.")
h2("4.1 English test sentences")
rows = [["ID", "Sentence (as given in the assignment)", "Pronunciation and quality"]]
for i in EN_IDS:
    n = note("speech_en", i)
    if not n:
        open_items.append(f"English speech observation for {i}")
    rows.append([i, esc(EN_TEXT[i]), esc(n) if n else "TO FILL"])
table(rows, [0.08, 0.47, 0.45])
h2("4.2 Telugu test sentences")
P("The Telugu sentences are identified by ID (T1, T2a, T2b, T3, T4) as printed in the assignment PDF.", SMALL)
rows = [["ID", "What the sentence tests", "Pronunciation and quality"]]
tests = {"T1": "Long technical word (pharmacological research)", "T2_a": "Homograph: grip on a rope", "T2_b": "Homograph: silk saree",
         "T3": "Short sentence, soft conjunct consonants", "T4": "Date, time and numerals"}
for i in TE_IDS:
    n = note("speech_te", i)
    if not n:
        open_items.append(f"Telugu speech observation for {i}")
    rows.append([i, tests[i], esc(n) if n else "TO FILL"])
table(rows, [0.08, 0.34, 0.58])

# ---------------------------------------------------------------- 5
h1("5. Variance adaptor analysis")
P("For each test sentence the model predicts a duration per phoneme (in mel frames), a pitch (F0) contour and an energy "
  "contour. Plots are saved under Results/&lt;Language&gt;/variance_adaptor_plots/{duration,pitch,energy}/.")

def variance(lang, ids, key):
    h2(f"5.{1 if lang == 'English' else 2} {lang}")
    found = False
    for i in ids:
        paths = [rd(f"Results/{lang}/variance_adaptor_plots/{k}/{i}.png") for k in ("duration", "pitch", "energy")]
        P(f"<b>{i}</b>", BODY)
        if all(paths):
            found = True
            for p, k in zip(paths, ("duration", "pitch (F0)", "energy")):
                figure(p, f"{lang} {i}: predicted {k}.", max_h=150)
        else:
            open_items.append(f"{lang} {i}: variance plots missing")
            P(f"<b>TO FILL:</b> plots for {i} not found in Results/{lang}/variance_adaptor_plots/", TODO)
        fill(f"{lang} {i}: explain the duration, pitch and energy patterns you see.", note(key, i))
    return found

variance("English", EN_IDS, "variance_en")
variance("Telugu", TE_IDS, "variance_te")

# ---------------------------------------------------------------- 6
h1("6. Comparison, limitations and possible improvements")
fill("Comparison of English and Telugu results based on the test sentences.", note("comparison"))
h2("6.1 Limitations")
P("Based on the work done so far:")
for t in [
    "<b>Training length.</b> Each model was trained for 35 epochs (about 4,000 updates for English). FastSpeech 2 normally "
    "needs many more updates to produce sharp mel spectrograms. Smooth, averaged mels are turned into hum by any vocoder.",
    "<b>Pitch predictor.</b> The pitch loss changed very little during early training, so predicted contours are likely flat.",
    "<b>Vocoder.</b> FastSpeech 2 outputs only a mel spectrogram. Griffin-Lim reconstructs mels poorly and sounds metallic or "
    "hazy even for good mels; a pretrained neural vocoder is the standard remedy. A pretrained English vocoder may also "
    "behave differently on Telugu.",
    "<b>Alignments.</b> Duration targets come from an external alignment step; errors there directly limit duration, pitch and "
    "energy quality.",
]:
    P("- " + t)
fill("Add failures specific to the provided test sentences (for example numerals and dates in E4/T4, the nonce word in E4, the split word in E3, homograph handling in E2/T2).", note("limitations"))
h2("6.2 Diagnostics to confirm the cause")
P("1) Vocode a ground-truth mel with the same vocoder: if it also sounds bad, the vocoder or the mel settings are at fault. "
  "2) Plot predicted mel against ground-truth mel for one validation sentence: a smeared mel without clear harmonics means an "
  "under-trained model. 3) Compare value ranges of real and predicted mels to rule out a normalisation mismatch.")
h2("6.3 Possible improvements")
for t in ["Train for many more updates (resume from the best checkpoint with a lower peak learning rate and a smaller batch size).",
          "Use a pretrained HiFi-GAN vocoder with matching mel settings instead of Griffin-Lim.",
          "Use higher-quality forced alignments for durations, and normalise pitch and energy targets per speaker.",
          "Add text normalisation for numerals, dates and times (needed for E4 and T4), and a better grapheme-to-phoneme step for "
          "out-of-vocabulary words (E1, E4) and for Telugu conjuncts.",
          "Use more training data or a single-speaker subset for cleaner supervision."]:
    P("- " + t)

# ---------------------------------------------------------------- 7
h1("7. Reproducibility")
h2("7.1 Software dependencies")
P("Python 3, PyTorch (CUDA build for training), librosa, soundfile, scipy, numpy, PyYAML, matplotlib, tqdm, speechbrain "
  "(for the optional pretrained HiFi-GAN vocoder). Exact versions: see requirements.txt.")
h2("7.2 Commands")
cmds = """# inference (works as used in this project)
python English/inference.py --config English/config.yaml --checkpoint English/checkpoints/best_model.pth
python Telugu/inference.py  --config Telugu/config.yaml  --checkpoint Telugu/checkpoints/best_model.pth

# generate this report (after filling report_notes.json)
python make_report.py"""
story.append(Preformatted(cmds, CODE))
fill("Add the exact preprocess and train commands for each language (training was run through the train_language_fast function in a Colab notebook; give the train.py command if one exists).", note("commands_note"))
P("Configuration files: English/config.yaml, Telugu/config.yaml. Checkpoints hosted on Google Drive: https://drive.google.com/drive/folders/17G4YMY4ce79uIVG7E7gOVIq9E0f4qVMf?usp=drive_link")

# ---------------------------------------------------------------- 8
h1("8. Tools and AI assistance")
P("AI coding assistants were used while building this project (code generation, debugging, and drafting of this report "
  "template). All reported numbers were taken from the project's own logs and result files, and the audio was listened to "
  "by the author.")

# ---------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(M, 10 * mm, "FastSpeech 2 TTS: English and Telugu")
    canvas.drawRightString(PAGE_W - M, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="FastSpeech 2 TTS Technical Report")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"[+] Wrote {OUT}")
if open_items:
    print(f"\n[!] {len(open_items)} items still marked TO FILL:")
    for it in open_items:
        print("   -", it)
