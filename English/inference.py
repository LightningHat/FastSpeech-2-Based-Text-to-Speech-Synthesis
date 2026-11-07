"""
Inference & Variance Adaptor Analysis Script for FastSpeech 2 (English)
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.model import FastSpeech2
from core.audio import AudioProcessor
from core.text_en import text_to_sequence_en

TEST_SENTENCES_EN = [
    ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
    ("E2_a", "The orchestra's conduct during the ceremony was praised by the audience."),
    ("E2_b", "The researchers will conduct an experiment to evaluate the proposed method."),
    ("E3", "The rhyth  m of the orchestra changed unexpectedly."),
    ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.")
]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Analysis for FastSpeech 2 English")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Custom text to synthesize")
    parser.add_argument("--test", action="store_true", help="Run inference on all 5 benchmark test sentences")
    parser.add_argument("--p_control", type=float, default=1.0, help="Pitch control scaling factor")
    parser.add_argument("--e_control", type=float, default=1.0, help="Energy control scaling factor")
    parser.add_argument("--d_control", type=float, default=1.0, help="Duration control scaling factor")
    return parser.parse_args()

def plot_variance_predictions(sample_id, text, phonemes, durations, pitch, energy, hop_time, out_base_dir):
    """Generate high-resolution plots for predicted duration, pitch, and energy."""
    dur_dir = os.path.join(out_base_dir, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(out_base_dir, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(out_base_dir, "variance_adaptor_plots", "energy")
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)

    # 1. Duration Bar Plot
    plt.figure(figsize=(max(8, len(phonemes) * 0.28), 4.5))
    x_pos = np.arange(len(phonemes))
    bars = plt.bar(x_pos, durations, color="#3498db", edgecolor="#2980b9", width=0.7)
    plt.xticks(x_pos, phonemes, rotation=70, fontsize=8, ha="right")
    plt.title(f"Predicted Phoneme Duration - {sample_id}\n\"{text[:65]}...\"", fontsize=11, fontweight="bold")
    plt.xlabel("Phoneme Sequence", fontsize=10)
    plt.ylabel("Duration (Mel Frames)", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    dur_path = os.path.join(dur_dir, f"{sample_id}.png")
    plt.savefig(dur_path, dpi=300)
    plt.close()

    # 2. Pitch (F0) Contour Plot
    time_axis = np.arange(len(pitch)) * hop_time
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, pitch, color="#2ecc71", lw=2.2, label="Predicted Pitch (F0)")
    plt.title(f"Predicted Pitch Contour (F0) - {sample_id}", fontsize=11, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Fundamental Frequency F0 (Hz)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    pitch_path = os.path.join(pitch_dir, f"{sample_id}.png")
    plt.savefig(pitch_path, dpi=300)
    plt.close()

    # 3. Energy Contour Plot
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, energy, color="#e74c3c", lw=2.2, label="Predicted Energy (L2 Norm)")
    plt.title(f"Predicted Frame Energy - {sample_id}", fontsize=11, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Energy Amplitude", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    energy_path = os.path.join(energy_dir, f"{sample_id}.png")
    plt.savefig(energy_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        candidate = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(candidate):
            config_path = candidate

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(config_path))
    result_dir = config["path"]["result_dir"]
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(base_dir, result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] English Inference Device: {device}")

    # Initialize model
    model = FastSpeech2(
        vocab_size=config["model"]["vocab_size"],
        encoder_hidden=config["model"]["encoder_hidden"],
        encoder_layers=config["model"]["encoder_layers"],
        encoder_heads=config["model"]["encoder_heads"],
        decoder_layers=config["model"]["decoder_layers"],
        decoder_heads=config["model"]["decoder_heads"],
        fft_inner_dim=config["model"]["fft_inner_dim"],
        n_mel_channels=config["model"]["n_mel_channels"],
        max_seq_len=config["model"]["max_seq_len"],
        pitch_min=config["model"]["pitch_min"],
        pitch_max=config["model"]["pitch_max"],
        energy_min=config["model"]["energy_min"],
        energy_max=config["model"]["energy_max"],
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(base_dir, config["path"]["checkpoint_dir"], "best_model.pth")

    if os.path.exists(ckpt_path):
        print(f"[*] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        if "model_state_dict" in state:
            if "variance_adaptor.pitch_bins" in state["model_state_dict"]:
                model.variance_adaptor.pitch_bins.copy_(state["model_state_dict"]["variance_adaptor.pitch_bins"])
            if "variance_adaptor.energy_bins" in state["model_state_dict"]:
                model.variance_adaptor.energy_bins.copy_(state["model_state_dict"]["variance_adaptor.energy_bins"])
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found. Running with initialized weights for demonstration.")

    model.eval()

    ap = AudioProcessor(
        sampling_rate=config["audio"]["sampling_rate"],
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        win_length=config["audio"]["win_length"],
        n_mels=config["audio"]["n_mels"],
        fmin=config["audio"]["fmin"],
        fmax=config["audio"]["fmax"],
    )

    hop_time = config["audio"]["hop_length"] / config["audio"]["sampling_rate"]
    audio_out_dir = os.path.join(result_dir, "test_generated_audios")
    os.makedirs(audio_out_dir, exist_ok=True)

    items = [(f"custom_{i}", args.text)] if args.text else TEST_SENTENCES_EN

    print(f"[*] Generating speech for {len(items)} test sentences...")
    for sample_id, text in items:
        seq, phonemes = text_to_sequence_en(text)
        text_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        src_lens = torch.tensor([len(seq)], dtype=torch.long, device=device)

        with torch.no_grad():
            preds = model(
                text_tensor,
                src_lens,
                p_control=args.p_control,
                e_control=args.e_control,
                d_control=args.d_control,
            )

        postnet_mel = preds["postnet_output"][0].cpu().numpy()
        durations = preds["dur_rounded"][0].cpu().numpy()
        pitch = preds["pitch_pred"][0].cpu().numpy()
        energy = preds["energy_pred"][0].cpu().numpy()

        # Invert predicted Mel-spectrogram to audio waveform via Griffin-Lim
        wav_path = os.path.join(audio_out_dir, f"{sample_id}.wav")
        audio = ap.mel_to_audio(postnet_mel)

        sf.write(wav_path, audio, config["audio"]["sampling_rate"], subtype="PCM_16")
        print(f"[*] [{sample_id}] Generated audio: {wav_path} (Duration: {len(audio)/config['audio']['sampling_rate']:.2f}s)")

        # Generate variance adaptor plots
        plot_variance_predictions(sample_id, text, phonemes, durations, pitch, energy, hop_time, result_dir)

    print("[*] English inference and variance analysis complete!")

if __name__ == "__main__":
    main()
