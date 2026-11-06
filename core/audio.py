import math
import numpy as np
import scipy
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

    def get_mel(self, audio):
        """Compute log-mel spectrogram [T, n_mels]."""
        # Ensure audio is 1D float
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        audio = audio.astype(np.float32)

        # STFT
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
        log_mel = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        return log_mel.T  # [T, n_mels]

    def get_energy(self, audio):
        """Compute frame-level energy (L2 norm of STFT magnitudes) [T]."""
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
        energy = np.linalg.norm(magnitudes, axis=0)
        return energy  # [T]

    def get_pitch(self, audio):
        """Extract frame-level F0 pitch contour with voiced interpolation [T]."""
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        audio = audio.astype(np.float32)

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.pitch_min,
                fmax=self.pitch_max,
                sr=self.sampling_rate,
                hop_length=self.hop_length,
                frame_length=self.win_length,
            )
        except Exception:
            f0 = librosa.yin(
                audio,
                fmin=self.pitch_min,
                fmax=self.pitch_max,
                sr=self.sampling_rate,
                hop_length=self.hop_length,
                frame_length=self.win_length,
            )
            voiced_flag = np.ones_like(f0, dtype=bool)

        # Interpolate unvoiced frames
        f0 = np.nan_to_num(f0)
        nonzero_idx = np.where(f0 > 0)[0]
        if len(nonzero_idx) > 0:
            interp_fn = scipy.interpolate.interp1d(
                nonzero_idx,
                f0[nonzero_idx],
                kind="linear",
                bounds_error=False,
                fill_value=(f0[nonzero_idx[0]], f0[nonzero_idx[-1]]),
            )
            f0 = interp_fn(np.arange(len(f0)))
        else:
            f0 = np.full_like(f0, (self.pitch_min + self.pitch_max) / 2.0)
        return f0  # [T]

    def mel_to_audio(self, mel, n_iter=60):
        """Invert mel-spectrogram to audio waveform via Griffin-Lim [audio_len]."""
        if isinstance(mel, torch.Tensor):
            mel = mel.squeeze().cpu().numpy()
        # mel is [T, n_mels] -> transpose to [n_mels, T]
        mel = mel.T
        mel_spectrogram = np.exp(mel)

        # Invert mel basis to approximate linear spectrogram
        inv_mel = np.linalg.pinv(self.mel_basis)
        linear_spec = np.maximum(1e-5, np.dot(inv_mel, mel_spectrogram))

        # Griffin-Lim phase reconstruction
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
        )
        return audio

