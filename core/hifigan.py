"""
HiFi-GAN Neural Vocoder Module for FastSpeech 2
Converts 80-channel log-mel spectrograms into high-fidelity 22,050 Hz speech waveforms.
"""

import os
import json
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HiFiGANGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(cfg.model_in_dim, cfg.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                cfg.upsample_initial_channel // (2**i),
                cfg.upsample_initial_channel // (2**(i+1)),
                k, u, padding=(k - u) // 2
            )))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg.upsample_initial_channel // (2**(i+1))
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @torch.no_grad()
    def infer(self, mel, device=None):
        """
        Synthesize audio waveform from mel-spectrogram.
        Args:
            mel: [T, 80] or [80, T] or [1, 80, T] numpy array or torch Tensor
        Returns:
            waveform: 1D float32 numpy array normalized to -1 dBFS
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel).float()

        if mel.dim() == 2:
            if mel.shape[1] == 80 and mel.shape[0] != 80:
                mel = mel.transpose(0, 1)  # [80, T]
            mel = mel.unsqueeze(0)  # [1, 80, T]
        elif mel.dim() == 3:
            if mel.shape[2] == 80 and mel.shape[1] != 80:
                mel = mel.transpose(1, 2)  # [1, 80, T]

        mel = mel.to(device)
        audio = self.forward(mel).squeeze().cpu().numpy()

        peak = np.max(np.abs(audio))
        if peak > 1e-4:
            audio = (audio / peak) * 0.89
        return audio.astype(np.float32)


DEFAULT_HIFIGAN_CONFIG = {
    "model_in_dim": 80,
    "upsample_initial_channel": 512,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}


def load_hifigan_vocoder(weights_path=None, device="cpu"):
    """
    Load pretrained HiFi-GAN universal vocoder.
    Automatically fetches weights from Hugging Face Hub if weights_path is not specified.
    """
    cfg = types.SimpleNamespace(**DEFAULT_HIFIGAN_CONFIG)
    vocoder = HiFiGANGenerator(cfg)

    if weights_path is None or not os.path.exists(weights_path):
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download("jaketae/hifigan-lj-v1", "pytorch_model.bin")
        except Exception as e:
            # Fallback to local cache check
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--jaketae--hifigan-lj-v1")
            for root, _, files in os.walk(cache_dir):
                if "pytorch_model.bin" in files:
                    weights_path = os.path.join(root, "pytorch_model.bin")
                    break

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        vocoder.load_state_dict(state)
        vocoder.remove_weight_norm()
        vocoder.to(device)
        vocoder.eval()
        print(f"[*] Loaded HiFi-GAN vocoder successfully from: {weights_path}")
        return vocoder
    else:
        raise FileNotFoundError(f"Could not locate HiFi-GAN weights at {weights_path}")
