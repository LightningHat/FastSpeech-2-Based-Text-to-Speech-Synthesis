import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Transformer FFT Blocks & Attention
# =========================================================================

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoidal position encoding table."""
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(self.n_head, sz_b, len_q, self.d_v).permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network with 1D Convolutions."""
    def __init__(self, d_in, d_hid, kernel_size=(9, 1), dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2
        )
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(nn.Module):
    """Feed-Forward Transformer Block."""
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size=(9, 1), dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, kernel_size, dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        return enc_output, enc_slf_attn

# =========================================================================
# Variance Predictor & Length Regulator
# =========================================================================

class VariancePredictor(nn.Module):
    """Duration / Pitch / Energy Predictor."""
    def __init__(self, hidden_dim=256, filter_size=256, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(hidden_dim, filter_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(filter_size, filter_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, encoder_output, mask=None):
        out = encoder_output.transpose(1, 2)
        out = self.conv_1[0](out)
        out = self.conv_1[1](out)
        out = self.conv_1[2](out.transpose(1, 2)).transpose(1, 2)
        out = self.conv_1[3](out)

        out = self.conv_2[0](out)
        out = self.conv_2[1](out)
        out = self.conv_2[2](out.transpose(1, 2)).transpose(1, 2)
        out = self.conv_2[3](out)

        out = self.linear_layer(out.transpose(1, 2)).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class LengthRegulator(nn.Module):
    """Length Regulator expands phonemes to frame length based on durations."""
    def __init__(self):
        super().__init__()

    def LR(self, x, duration, max_len):
        output = []
        for batch, expand_target in zip(x, duration):
            expanded = []
            for phoneme, d in zip(batch, expand_target):
                d_val = int(round(float(d)))
                if d_val > 0:
                    expanded.append(phoneme.repeat(d_val, 1))
            if len(expanded) > 0:
                expanded = torch.cat(expanded, dim=0)
            else:
                expanded = batch[:1]
            output.append(expanded)

        if max_len is not None:
            output = [F.pad(item, (0, 0, 0, max(0, max_len - item.size(0))))[:max_len] for item in output]
        else:
            max_len = max(item.size(0) for item in output)
            output = [F.pad(item, (0, 0, 0, max_len - item.size(0))) for item in output]

        return torch.stack(output, dim=0)

    def forward(self, x, duration, max_len=None):
        return self.LR(x, duration, max_len)


class VarianceAdaptor(nn.Module):
    """Variance Adaptor combining Duration, Pitch, and Energy."""
    def __init__(self, hidden_dim=256, n_bins=256, pitch_min=50.0, pitch_max=800.0, energy_min=0.0, energy_max=100.0):
        super().__init__()
        self.duration_predictor = VariancePredictor(hidden_dim=hidden_dim)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(hidden_dim=hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim=hidden_dim)

        self.n_bins = n_bins
        # Linear bins for pitch & energy
        self.register_buffer("pitch_bins", torch.linspace(pitch_min, pitch_max, n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(energy_min, energy_max, n_bins - 1))

        self.pitch_embedding = nn.Embedding(n_bins, hidden_dim)
        self.energy_embedding = nn.Embedding(n_bins, hidden_dim)

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)

        if duration_target is not None:
            x_expanded = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - 1) * d_control, min=0
            )
            x_expanded = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(
                torch.sum(duration_rounded, dim=-1).long(), max_len=x_expanded.size(1)
            )

        pitch_prediction = self.pitch_predictor(x_expanded, mel_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_prediction = pitch_prediction * p_control
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_prediction, self.pitch_bins))

        energy_prediction = self.energy_predictor(x_expanded, mel_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))
        else:
            energy_prediction = energy_prediction * e_control
            energy_embedding = self.energy_embedding(torch.bucketize(energy_prediction, self.energy_bins))

        x_adapted = x_expanded + pitch_embedding + energy_embedding

        return (
            x_adapted,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_mask,
        )

# =========================================================================
# Encoder, Decoder, PostNet & FastSpeech2 Model
# =========================================================================

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = ids.unsqueeze(0).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
    return mask


class PostNet(nn.Module):
    """5-layer 1D Conv PostNet to refine predicted mel-spectrogram."""
    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=5, n_convs=5):
        super().__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mel_channels, postnet_embedding_dim, kernel_size=postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
            )
        )

        for _ in range(n_convs - 2):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_embedding_dim, n_mel_channels, kernel_size=postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
                nn.Dropout(0.5),
            )
        )

    def forward(self, x):
        # x: [B, T, n_mels] -> transpose to [B, n_mels, T]
        out = x.transpose(1, 2)
        residual = out
        for conv in self.convolutions:
            residual = conv(residual)
        return (out + residual).transpose(1, 2)


class FastSpeech2(nn.Module):
    """Complete FastSpeech 2 Architecture."""
    def __init__(
        self,
        vocab_size=300,
        encoder_hidden=256,
        encoder_layers=4,
        encoder_heads=2,
        decoder_layers=4,
        decoder_heads=2,
        fft_inner_dim=1024,
        n_mel_channels=80,
        max_seq_len=1000,
        pitch_min=50.0,
        pitch_max=800.0,
        energy_min=0.0,
        energy_max=100.0,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.n_mel_channels = n_mel_channels

        # Encoder
        self.src_word_emb = nn.Embedding(vocab_size, encoder_hidden, padding_idx=0)
        self.register_buffer("position_enc", get_sinusoid_encoding_table(max_seq_len + 1, encoder_hidden, padding_idx=0))

        self.encoder_layers = nn.ModuleList([
            FFTBlock(
                d_model=encoder_hidden,
                n_head=encoder_heads,
                d_k=encoder_hidden // encoder_heads,
                d_v=encoder_hidden // encoder_heads,
                d_inner=fft_inner_dim,
                kernel_size=(9, 1),
                dropout=0.2,
            )
            for _ in range(encoder_layers)
        ])

        # Variance Adaptor
        self.variance_adaptor = VarianceAdaptor(
            hidden_dim=encoder_hidden,
            n_bins=256,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            energy_min=energy_min,
            energy_max=energy_max,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([
            FFTBlock(
                d_model=encoder_hidden,
                n_head=decoder_heads,
                d_k=encoder_hidden // decoder_heads,
                d_v=encoder_hidden // decoder_heads,
                d_inner=fft_inner_dim,
                kernel_size=(9, 1),
                dropout=0.2,
            )
            for _ in range(decoder_layers)
        ])

        self.mel_linear = nn.Linear(encoder_hidden, n_mel_channels)
        self.postnet = PostNet(n_mel_channels=n_mel_channels)

    def forward(
        self,
        texts,
        src_lens,
        max_src_len=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_mask = get_mask_from_lengths(src_lens, max_src_len)
        mel_mask = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        # Encoder
        slf_attn_mask = src_mask.unsqueeze(1).expand(-1, src_mask.size(1), -1)
        enc_output = self.src_word_emb(texts) + self.position_enc[:texts.size(1), :].unsqueeze(0)

        for enc_layer in self.encoder_layers:
            enc_output, _ = enc_layer(enc_output, mask=src_mask, slf_attn_mask=slf_attn_mask)

        # Variance Adaptor
        (
            adapted_output,
            pitch_pred,
            energy_pred,
            log_dur_pred,
            dur_rounded,
            mel_mask,
        ) = self.variance_adaptor(
            enc_output,
            src_mask,
            mel_mask=mel_mask,
            max_len=max_mel_len,
            pitch_target=p_targets,
            energy_target=e_targets,
            duration_target=d_targets,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

        # Decoder
        dec_slf_attn_mask = mel_mask.unsqueeze(1).expand(-1, mel_mask.size(1), -1)
        dec_output = adapted_output + self.position_enc[:adapted_output.size(1), :].unsqueeze(0)

        for dec_layer in self.decoder_layers:
            dec_output, _ = dec_layer(dec_output, mask=mel_mask, slf_attn_mask=dec_slf_attn_mask)

        mel_output = self.mel_linear(dec_output)
        mel_output = mel_output.masked_fill(mel_mask.unsqueeze(-1), 0.0)

        postnet_output = self.postnet(mel_output)
        postnet_output = postnet_output.masked_fill(mel_mask.unsqueeze(-1), 0.0)

        return {
            "mel_output": mel_output,
            "postnet_output": postnet_output,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "log_dur_pred": log_dur_pred,
            "dur_rounded": dur_rounded,
            "src_mask": src_mask,
            "mel_mask": mel_mask,
        }


class FastSpeech2Loss(nn.Module):
    """FastSpeech 2 Multi-Task Loss."""
    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        mel_targets = targets["mels"]
        pitch_targets = targets["pitches"]
        energy_targets = targets["energies"]
        dur_targets = targets["durations"]

        mel_output = predictions["mel_output"]
        postnet_output = predictions["postnet_output"]
        pitch_pred = predictions["pitch_pred"]
        energy_pred = predictions["energy_pred"]
        log_dur_pred = predictions["log_dur_pred"]

        src_mask = ~predictions["src_mask"]
        mel_mask = ~predictions["mel_mask"]

        # Truncate to matching temporal bounds if needed
        T_mel = min(mel_output.size(1), mel_targets.size(1), mel_mask.size(1))
        mel_output = mel_output[:, :T_mel]
        postnet_output = postnet_output[:, :T_mel]
        mel_targets = mel_targets[:, :T_mel]
        mel_mask = mel_mask[:, :T_mel]
        pitch_pred = pitch_pred[:, :T_mel]
        pitch_targets = pitch_targets[:, :T_mel]
        energy_pred = energy_pred[:, :T_mel]
        energy_targets = energy_targets[:, :T_mel]

        # Mel losses
        mel_mask_3d = mel_mask.unsqueeze(-1)
        mel_loss = self.mae_loss(mel_output.masked_select(mel_mask_3d), mel_targets.masked_select(mel_mask_3d))
        postnet_loss = self.mae_loss(postnet_output.masked_select(mel_mask_3d), mel_targets.masked_select(mel_mask_3d))

        # Variance losses
        pitch_loss = self.mse_loss(pitch_pred.masked_select(mel_mask), pitch_targets.masked_select(mel_mask))
        energy_loss = self.mse_loss(energy_pred.masked_select(mel_mask), energy_targets.masked_select(mel_mask))

        log_dur_targets = torch.log(dur_targets.float() + 1.0)
        dur_loss = self.mse_loss(log_dur_pred.masked_select(src_mask), log_dur_targets.masked_select(src_mask))

        total_loss = mel_loss + postnet_loss + dur_loss + pitch_loss + energy_loss

        return {
            "total_loss": total_loss,
            "mel_loss": mel_loss,
            "postnet_loss": postnet_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "dur_loss": dur_loss,
        }

