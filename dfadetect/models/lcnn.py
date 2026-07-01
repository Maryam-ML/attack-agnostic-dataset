"""
LCNN + SE (lightweight) for LFCC, based on ASVspoof 2021 LFCC-LCNN baseline.
Adds Squeeze-and-Excitation blocks at key stages but keeps model small and fast.
"""

import sys
import torch
import torch.nn as torch_nn


# ----------------- Helper modules ----------------- #

class BLSTMLayer(torch_nn.Module):
    """Bi-directional LSTM layer for temporal modeling.

    Input:  (batch, length, dim_in)
    Output: (batch, length, dim_out)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            print(f"Output_dim of BLSTMLayer is {output_dim:d}")
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)

        self.l_blstm = torch_nn.LSTM(
            input_dim,
            output_dim // 2,
            bidirectional=True,
            batch_first=False,  # we permute manually
        )

    def forward(self, x):
        # x: (batch, length, dim) -> (length, batch, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # back to (batch, length, dim)
        return blstm_data.permute(1, 0, 2)


class MaxFeatureMap2D(torch_nn.Module):
    """Max Feature Map along a given dimension (default: channel)."""

    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print(f"MaxFeatureMap: maximize on {self.max_dim} dim")
            print(f"But input has {len(shape)} dimensions")
            sys.exit(1)

        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print(f"MaxFeatureMap: maximize on {self.max_dim} dim")
            print("But this dimension has an odd number of data")
            sys.exit(1)

        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batch, 2, channel//2, ...) and max over that dim
        m, _ = inputs.view(*shape).max(self.max_dim)
        return m


class SEBlock(torch_nn.Module):
    """Lightweight Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = torch_nn.AdaptiveAvgPool2d(1)
        self.fc = torch_nn.Sequential(
            torch_nn.Linear(channels, channels // reduction, bias=False),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(channels // reduction, channels, bias=False),
            torch_nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ----------------- LCNN + SE model ----------------- #

class LCNN(torch_nn.Module):
    """Lightweight LCNN with SE blocks for LFCC-based spoofing detection.

    input_channels: usually 3 (e.g., LFCC + delta + delta-delta)
    num_coefficients: LFCC dimension, e.g., 80
    """

    def __init__(self, **kwargs):
        super().__init__()
        dropout         = kwargs.get("dropout", 0.4)
        input_channels = kwargs.get("input_channels", 3)
        num_coefficients = kwargs.get("num_coefficients", 80)

        self.num_coefficients = num_coefficients
        self.v_emd_dim = 1  # single logit

        # Front-end convolution + MFM + SE
        # We slightly reduce some channels (96->80, 128->96) to stay light.

        self.m_transform = torch_nn.Sequential(
            # ---- Block 1 ----
            torch_nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),          # 64 -> 32 channels
            SEBlock(32),
            torch_nn.MaxPool2d((2, 2), (2, 2)),

            # ---- Block 2 ----
            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),          # 64 -> 32
            SEBlock(32),
            torch_nn.BatchNorm2d(32, affine=False),

            torch_nn.Conv2d(32, 80, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),          # 80 -> 40
            SEBlock(40),

            torch_nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.BatchNorm2d(40, affine=False),

            # ---- Block 3 ----
            torch_nn.Conv2d(40, 80, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),          # 80 -> 40
            SEBlock(40),
            torch_nn.BatchNorm2d(40, affine=False),

            torch_nn.Conv2d(40, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),          # 96 -> 48
            SEBlock(48),

            torch_nn.MaxPool2d((2, 2), (2, 2)),

            # ---- Block 4 ----
            torch_nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),          # 96 -> 48
            SEBlock(48),
            torch_nn.BatchNorm2d(48, affine=False),

            torch_nn.Conv2d(48, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),          # 64 -> 32
            SEBlock(32),
            torch_nn.BatchNorm2d(32, affine=False),

            # ---- Block 5 ----
            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),          # 64 -> 32
            SEBlock(32),
            torch_nn.BatchNorm2d(32, affine=False),

            torch_nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),          # 64 -> 32
            SEBlock(32),

            torch_nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Dropout(dropout),
        )

        # After 4 MaxPool2d (2x2) operations, frequency dimension is /16.
        # Last block channel count after MFM is 32.
        # So the BLSTM input dim is (num_coefficients // 16) * 32.
        blstm_dim = (self.num_coefficients // 16) * 32

        self.m_before_pooling = torch_nn.Sequential(
            BLSTMLayer(blstm_dim, blstm_dim),
            BLSTMLayer(blstm_dim, blstm_dim),
        )

        self.m_output_act = torch_nn.Linear(blstm_dim, self.v_emd_dim)

    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, num_coefficients, T)
        returns: (batch, v_emd_dim)
        """
        batch_size = x.shape[0]

        # Conv expects (batch, channel, freq, time)
        # Your original code permutes as (batch, 1, frame_length, fft_bin) → here we keep same logic.
        x = x.permute(0, 1, 3, 2)  # (B, C, T, F) -> Conv2d

        hidden_features = self.m_transform(x)  # (B, C', T', F')

        # (batch, channel, frame//N, feat_dim//N) -> (batch, frame//N, channel * feat_dim//N)
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]
        hidden_features = hidden_features.view(batch_size, frame_num, -1)

        # BLSTM + residual
        hidden_features_lstm = self.m_before_pooling(hidden_features)

        # Mean pool over time, pass through linear head
        tmp_emb = self.m_output_act((hidden_features_lstm + hidden_features).mean(1))

        return tmp_emb

    def _compute_score(self, feature_vec: torch.Tensor) -> torch.Tensor:
        # feature_vec: [batch, 1]
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_vec = self._compute_embedding(x)
        return feature_vec


if __name__ == "__main__":
    print("Definition of LCNN_SE model")
    model = LCNN(
    input_channels=3, 
    num_coefficients=80,
    dropout=0.4, 
    )
    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    print("Output shape:", output.shape)