"""
Training Script for FastSpeech 2 (English)
"""
import os
import sys
import json
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.model import FastSpeech2, FastSpeech2Loss
from core.dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 English Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()

def plot_loss_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    # Metrics
    train_mel = [h["train_mel_loss"] for h in history]
    train_dur = [h["train_dur_loss"] for h in history]
    train_pitch = [h["train_pitch_loss"] for h in history]
    train_energy = [h["train_energy_loss"] for h in history]

    val_mel = [h["val_mel_loss"] for h in history]
    val_dur = [h["val_dur_loss"] for h in history]
    val_pitch = [h["val_pitch_loss"] for h in history]
    val_energy = [h["val_energy_loss"] for h in history]

    # 1. Training Loss Plot (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    fig.suptitle("FastSpeech 2 (English) - Training Loss Convergence", fontsize=14, fontweight="bold", y=0.98)

    axes[0, 0].plot(epochs, train_mel, color="#2b6cb0", marker="o", markersize=4, lw=2, label="Train Mel Loss (L1)")
    axes[0, 0].set_title("(a) Mel Spectrogram Reconstruction Loss", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=10)
    axes[0, 0].set_ylabel("L1 Loss", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(frameon=True)

    axes[0, 1].plot(epochs, train_dur, color="#c53030", marker="s", markersize=4, lw=2, label="Train Duration Loss (MSE)")
    axes[0, 1].set_title("(b) Duration Predictor Loss", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=10)
    axes[0, 1].set_ylabel("MSE Loss (Log Frames)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(frameon=True)

    axes[1, 0].plot(epochs, train_pitch, color="#2f855a", marker="^", markersize=4, lw=2, label="Train Pitch Loss (MSE)")
    axes[1, 0].set_title("(c) Pitch (F0) Predictor Loss", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=10)
    axes[1, 0].set_ylabel("MSE Loss (Hz^2)", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(frameon=True)

    axes[1, 1].plot(epochs, train_energy, color="#805ad5", marker="d", markersize=4, lw=2, label="Train Energy Loss (MSE)")
    axes[1, 1].set_title("(d) Energy Predictor Loss", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=10)
    axes[1, 1].set_ylabel("MSE Loss (L2 Norm)", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(frameon=True)

    plt.tight_layout()
    train_plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(train_plot_path, dpi=300)
    plt.close()
    print(f"[*] Saved training loss plot to: {train_plot_path}")

    # 2. Validation Loss Plot (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    fig.suptitle("FastSpeech 2 (English) - Validation Loss Convergence", fontsize=14, fontweight="bold", y=0.98)

    axes[0, 0].plot(epochs, val_mel, color="#3182ce", marker="o", markersize=4, lw=2, linestyle="--", label="Val Mel Loss (L1)")
    axes[0, 0].set_title("(a) Mel Spectrogram Reconstruction Loss", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=10)
    axes[0, 0].set_ylabel("L1 Loss", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(frameon=True)

    axes[0, 1].plot(epochs, val_dur, color="#e53e3e", marker="s", markersize=4, lw=2, linestyle="--", label="Val Duration Loss (MSE)")
    axes[0, 1].set_title("(b) Duration Predictor Loss", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=10)
    axes[0, 1].set_ylabel("MSE Loss (Log Frames)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(frameon=True)

    axes[1, 0].plot(epochs, val_pitch, color="#38a169", marker="^", markersize=4, lw=2, linestyle="--", label="Val Pitch Loss (MSE)")
    axes[1, 0].set_title("(c) Pitch (F0) Predictor Loss", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=10)
    axes[1, 0].set_ylabel("MSE Loss (Hz^2)", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(frameon=True)

    axes[1, 1].plot(epochs, val_energy, color="#9f7aea", marker="d", markersize=4, lw=2, linestyle="--", label="Val Energy Loss (MSE)")
    axes[1, 1].set_title("(d) Energy Predictor Loss", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=10)
    axes[1, 1].set_ylabel("MSE Loss (L2 Norm)", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(frameon=True)

    plt.tight_layout()
    val_plot_path = os.path.join(output_dir, "validation_loss.png")
    plt.savefig(val_plot_path, dpi=300)
    plt.close()
    print(f"[*] Saved validation loss plot to: {val_plot_path}")

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
    epochs = args.epochs if args.epochs else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size else config["training"]["batch_size"]
    lr = args.lr if args.lr else config["training"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # Load stats if available
    processed_dir = config["path"]["processed_data_dir"]
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(base_dir, processed_dir)

    ckpt_dir = config["path"]["checkpoint_dir"]
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(base_dir, ckpt_dir)

    result_dir = config["path"]["result_dir"]
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(base_dir, result_dir)

    stats_file = os.path.join(processed_dir, "stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)
        pitch_min, pitch_max = stats["pitch"][:2]
        energy_min, energy_max = stats["energy"][:2]
    else:
        pitch_min, pitch_max = config["model"]["pitch_min"], config["model"]["pitch_max"]
        energy_min, energy_max = config["model"]["energy_min"], config["model"]["energy_max"]

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
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        energy_min=energy_min,
        energy_max=energy_max,
    ).to(device)

    criterion = FastSpeech2Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4)

    train_loader = get_dataloader(os.path.join(processed_dir, "train"), batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(os.path.join(processed_dir, "val"), batch_size=batch_size, shuffle=False)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"[*] Dataset samples - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print(f"[*] Starting training for {epochs} epochs...")

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_losses = {"total": 0.0, "mel": 0.0, "pitch": 0.0, "energy": 0.0, "dur": 0.0}
        n_train_batches = max(1, len(train_loader))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch in pbar:
            texts = batch["texts"].to(device)
            src_lens = batch["src_lens"].to(device)
            mels = batch["mels"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            pitches = batch["pitches"].to(device)
            energies = batch["energies"].to(device)
            durations = batch["durations"].to(device)

            targets = {
                "mels": mels,
                "pitches": pitches,
                "energies": energies,
                "durations": durations,
            }

            optimizer.zero_grad()
            preds = model(
                texts,
                src_lens,
                max_src_len=batch["max_src_len"],
                mels=mels,
                mel_lens=mel_lens,
                max_mel_len=batch["max_mel_len"],
                p_targets=pitches,
                e_targets=energies,
                d_targets=durations,
            )

            loss_dict = criterion(preds, targets)
            loss_dict["total_loss"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_thresh"])
            optimizer.step()

            train_losses["total"] += loss_dict["total_loss"].item()
            train_losses["mel"] += loss_dict["mel_loss"].item()
            train_losses["pitch"] += loss_dict["pitch_loss"].item()
            train_losses["energy"] += loss_dict["energy_loss"].item()
            train_losses["dur"] += loss_dict["dur_loss"].item()

            pbar.set_postfix({"loss": f"{loss_dict['total_loss'].item():.4f}"})

        for k in train_losses:
            train_losses[k] /= n_train_batches

        # Validation Phase
        model.eval()
        val_losses = {"total": 0.0, "mel": 0.0, "pitch": 0.0, "energy": 0.0, "dur": 0.0}
        n_val_batches = max(1, len(val_loader))

        with torch.no_grad():
            for batch in val_loader:
                texts = batch["texts"].to(device)
                src_lens = batch["src_lens"].to(device)
                mels = batch["mels"].to(device)
                mel_lens = batch["mel_lens"].to(device)
                pitches = batch["pitches"].to(device)
                energies = batch["energies"].to(device)
                durations = batch["durations"].to(device)

                targets = {
                    "mels": mels,
                    "pitches": pitches,
                    "energies": energies,
                    "durations": durations,
                }

                preds = model(
                    texts,
                    src_lens,
                    max_src_len=batch["max_src_len"],
                    mels=mels,
                    mel_lens=mel_lens,
                    max_mel_len=batch["max_mel_len"],
                    p_targets=pitches,
                    e_targets=energies,
                    d_targets=durations,
                )

                loss_dict = criterion(preds, targets)
                val_losses["total"] += loss_dict["total_loss"].item()
                val_losses["mel"] += loss_dict["mel_loss"].item()
                val_losses["pitch"] += loss_dict["pitch_loss"].item()
                val_losses["energy"] += loss_dict["energy_loss"].item()
                val_losses["dur"] += loss_dict["dur_loss"].item()

        for k in val_losses:
            val_losses[k] /= n_val_batches

        epoch_stats = {
            "epoch": epoch,
            "train_total_loss": train_losses["total"],
            "train_mel_loss": train_losses["mel"],
            "train_pitch_loss": train_losses["pitch"],
            "train_energy_loss": train_losses["energy"],
            "train_dur_loss": train_losses["dur"],
            "val_total_loss": val_losses["total"],
            "val_mel_loss": val_losses["mel"],
            "val_pitch_loss": val_losses["pitch"],
            "val_energy_loss": val_losses["energy"],
            "val_dur_loss": val_losses["dur"],
        }
        history.append(epoch_stats)

        print(f"[*] Epoch {epoch:02d} Summary: Train Total={train_losses['total']:.4f} | Val Total={val_losses['total']:.4f}")

        # Checkpoint Saving
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
        }

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(ckpt_state, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"[*] -> Saved new best model checkpoint! (Val Loss: {best_val_loss:.4f})")

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(ckpt_state, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pth"))

    # Save final history and generate loss plots
    with open(os.path.join(result_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_loss_curves(history, result_dir)
    print("[*] Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
