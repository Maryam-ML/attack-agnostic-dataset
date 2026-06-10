"""
Stabilized LCNN for Audio Deepfake Detection
=======================================================================
Memory-Safe Edition + Cross-Fold Generalization Upgrades

Upgrades to stabilize EER across unseen folds:
  1. Attentive Statistics Pooling (ASP): Captures both mean and variance
     of temporal artifacts, crucial for unseen vocoders.
  2. Spatial Dropout (Dropout2d): Drops entire frequency bands during 
     training to prevent co-adaptation to fold-specific artifacts.
  3. Streamlined Temporal Head: Removed the heavy Transformer layer to 
     stop temporal pacing memorization. Single Bi-LSTM + ASP.
  4. Memory Fixes Retained: torch.amax and gradient checkpointing.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Primitives & Memory Fixes
# ─────────────────────────────────────────────────────────────────────────────

class MaxFeatureMap2D(nn.Module):
    """Memory-safe MFM using torch.amax"""
    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.max_dim
        if x.size(d) % 2 != 0:
            sys.exit(1)
        shape = list(x.size())
        shape[d] //= 2
        shape.insert(d, 2)
        return torch.amax(x.view(*shape), dim=d)


class BLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if output_dim % 2 != 0:
            sys.exit(1)
        self.lstm = nn.LSTM(
            input_dim, output_dim // 2,
            bidirectional=True, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Attention & Pooling
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FrequencyAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0),
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class AttentiveStatPool(nn.Module):
    """
    ASP: Learns where to attend in time, and returns the concatenated 
    weighted mean AND standard deviation of the features.
    Crucial for stabilizing results on Attack-Agnostic datasets.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)         # (B, T, 1)
        mean = (w * x).sum(dim=1)                      # (B, D)
        var = (w * (x - mean.unsqueeze(1))**2).sum(dim=1) 
        std = torch.sqrt(var.clamp(min=1e-9))          # (B, D)
        return torch.cat([mean, std], dim=1)           # (B, 2D)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Block
# ─────────────────────────────────────────────────────────────────────────────

class ConvMFMSE(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, 1, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MaxFeatureMap2D()
        self.se   = SEBlock(out_ch, reduction=16)
        
        # Spatial dropout drops entire feature maps, preventing feature co-adaptation
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.mfm(x)
        x = self.se(x)
        x = self.drop(x)
        return x


def spec_augment(x: torch.Tensor, freq_mask: int = 8, time_mask: int = 20) -> torch.Tensor:
    B, C, F, T = x.shape
    out = x.clone()
    for b in range(B):
        for _ in range(2):
            f0 = torch.randint(0, max(F - freq_mask, 1), (1,)).item()
            fw = torch.randint(1, freq_mask + 1, (1,)).item()
            out[b, :, f0:f0 + fw, :] = 0.0
        for _ in range(2):
            t0 = torch.randint(0, max(T - time_mask, 1), (1,)).item()
            tw = torch.randint(1, time_mask + 1, (1,)).item()
            out[b, :, :, t0:t0 + tw] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class LCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_ch           = kwargs.get("input_channels",   3)
        num_coeff       = kwargs.get("num_coefficients", 80)
        dropout         = kwargs.get("dropout",          0.4)
        self.use_aug    = kwargs.get("use_spec_augment", True)
        self.use_ckpt   = kwargs.get("use_checkpoint",   True)
        self.v_emd_dim  = 1

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, (5, 5), 1, (2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            MaxFeatureMap2D(),  
            nn.MaxPool2d(2, 2),
        )
        nn.init.kaiming_normal_(self.stem[0].weight, mode='fan_out', nonlinearity='relu')

        # Stages with Spatial Dropout to force generalization
        self.stage2 = nn.Sequential(
            ConvMFMSE(32, 32, dropout_rate=0.05),
            nn.BatchNorm2d(32, affine=False),
            ConvMFMSE(32, 48, dropout_rate=0.05),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48, affine=False),
        )

        self.stage3 = nn.Sequential(
            ConvMFMSE(48, 48, dropout_rate=0.1),
            nn.BatchNorm2d(48, affine=False),
            ConvMFMSE(48, 64, dropout_rate=0.1),
            nn.MaxPool2d(2, 2),
        )

        self.stage4 = nn.Sequential(
            ConvMFMSE(64, 64, dropout_rate=0.1),
            nn.BatchNorm2d(64, affine=False),
            ConvMFMSE(64, 32, dropout_rate=0.1),
            nn.BatchNorm2d(32, affine=False),
            ConvMFMSE(32, 32, dropout_rate=0.1),
            nn.MaxPool2d(2, 2),
        )

        self.freq_attn = FrequencyAttention(channels=32)

        # ── Streamlined Temporal Pipeline ──
        lstm_dim = (num_coeff // 16) * 32
        
        # Removed the heavy Transformer to prevent fold-memorization
        self.blstm = BLSTMLayer(lstm_dim, lstm_dim)
        
        # ASP doubles the dimension (concatenates mean + std)
        self.asp = AttentiveStatPool(lstm_dim)
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, self.v_emd_dim),
        )

    def _run_stem(self, x): return self.stem(x)
    def _run_stage2(self, x): return self.stage2(x)
    def _run_stage3(self, x): return self.stage3(x)
    def _run_stage4(self, x): return self.stage4(x)

    def _compute_score(self, logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit).squeeze(1)

    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.permute(0, 1, 3, 2)

        if self.training and self.use_aug:
            x = spec_augment(x)

        if self.use_ckpt and self.training:
            x = checkpoint(self._run_stem,   x, use_reentrant=False)
            x = checkpoint(self._run_stage2, x, use_reentrant=False)
            x = checkpoint(self._run_stage3, x, use_reentrant=False)
            x = checkpoint(self._run_stage4, x, use_reentrant=False)
        else:
            x = self._run_stem(x)
            x = self._run_stage2(x)
            x = self._run_stage3(x)
            x = self._run_stage4(x)

        x = self.freq_attn(x)

        x = x.permute(0, 2, 1, 3).contiguous()
        T_prime = x.size(1)
        x = x.view(B, T_prime, -1)

        # ── Streamlined Temporal Aggregation ──
        h = self.blstm(x)
        
        # Attentive Statistics Pooling (Mean + Std)
        pooled = self.asp(h) 

        return self.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compute_embedding(x)

if __name__ == "__main__":
    print("Definition of model")
    model = LCNN(input_channels=3, num_coefficients=80)
    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    print(output.shape)