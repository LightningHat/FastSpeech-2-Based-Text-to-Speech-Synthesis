import math
import numpy as np
import scipy.interpolate
import scipy.signal
import torch
import librosa
import soundfile as sf

class AudioProcessor:
    def __init__(
        self,
        sampling_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
        pitch_min=50.0,
        pitch_max=800.0,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

        self.mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)

    def get_mel_and_energy(self, audio):
        """Compute log-mel spectrogram [T, n_mels] and frame L2 energy [T] in a single STFT pass."""
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        audio = audio.astype(np.float32)

        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            pad_mode="reflect",
        )
        magnitudes = np.abs(stft)
        mel_spectrogram = np.dot(self.mel_basis, magnitudes)
        log_mel = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None)).T
        energy = np.linalg.norm(magnitudes, axis=0)
        return log_mel, energy

    def get_mel(self, audio):
        log_mel, _ = self.get_mel_and_energy(audio)
        return log_mel

    def get_energy(self, audio):
        _, energy = self.get_mel_and_energy(audio)
        return energy

    def get_pitch(self, audio):
        """Fast frame-level F0 extraction using YIN with linear unvoiced interpolation [T]."""
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        audio = audio.astype(np.float32)

        f0 = librosa.yin(
            audio,
            fmin=max(60.0, self.pitch_min),
            fmax=min(500.0, self.pitch_max),
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            frame_length=self.win_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        # Filter out boundary spikes
        valid = (f0 > 62.0) & (f0 < 480.0)
        nonzero_idx = np.where(valid)[0]
        if len(nonzero_idx) > 1:
            interp_fn = scipy.interpolate.interp1d(
                nonzero_idx,
                f0[nonzero_idx],
                kind="linear",
                bounds_error=False,
                fill_value=(f0[nonzero_idx[0]], f0[nonzero_idx[-1]]),
            )
            f0 = interp_fn(np.arange(len(f0)))
        elif len(nonzero_idx) == 1:
            f0 = np.full_like(f0, f0[nonzero_idx[0]])
        else:
            f0 = np.full_like(f0, 175.0)
        return f0.astype(np.float32)

    def mel_to_audio(self, mel, n_iter=64):
        """Invert log-mel spectrogram [T, n_mels] to waveform via NNLS-based Griffin-Lim."""
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().squeeze().cpu().numpy()
        mel = mel.T  # [n_mels, T]
        mel_power = np.exp(mel)

        audio = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=1.0,
            n_iter=n_iter,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        peak = np.max(np.abs(audio))
        if peak > 1e-4:
            audio = (audio / peak) * 0.89
        return audio.astype(np.float32)
