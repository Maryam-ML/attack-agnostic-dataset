import sys
import torch
import torch.nn as torch_nn


class BLSTMLayer(torch_nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        self.l_blstm = torch_nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x):
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        return blstm_data.permute(1, 0, 2)


class MaxFeatureMap2D(torch_nn.Module):
    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


# ----------------- CBAM modules ----------------- #

class ChannelAttention(torch_nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = torch_nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch_nn.AdaptiveMaxPool2d(1)
        self.fc = torch_nn.Sequential(
            torch_nn.Linear(channels, max(channels // reduction, 1), bias=False),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(max(channels // reduction, 1), channels, bias=False),
        )
        self.sigmoid = torch_nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(torch_nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch_nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch_nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class CBAMBlock(torch_nn.Module):
    """Channel attention + spatial attention, applied sequentially."""
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


##############
## FOR MODEL
##############

class LCNN(torch_nn.Module):
    """LCNN baseline with CBAM inserted after every MaxFeatureMap2D activation."""

    def __init__(self, **kwargs):
        super().__init__()
        input_channels = kwargs.get("input_channels", 3)
        num_coefficients = kwargs.get("num_coefficients", 80)
        dropout = kwargs.get("dropout", 0.7)
        cbam_reduction = kwargs.get("cbam_reduction", 8)
        cbam_kernel = kwargs.get("cbam_kernel", 7)

        self.num_coefficients = num_coefficients
        self.v_emd_dim = 1

        self.m_transform = torch_nn.Sequential(
            # ---- Block 1 ----
            torch_nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),                       # 64 -> 32
            CBAMBlock(32, cbam_reduction, cbam_kernel),
            torch_nn.MaxPool2d((2, 2), (2, 2)),

            # ---- Block 2 ----
            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),                       # 64 -> 32
            CBAMBlock(32, cbam_reduction, cbam_kernel),
            torch_nn.BatchNorm2d(32, affine=False),

            torch_nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),                       # 96 -> 48
            CBAMBlock(48, cbam_reduction, cbam_kernel),

            torch_nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.BatchNorm2d(48, affine=False),

            # ---- Block 3 ----
            torch_nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),                       # 96 -> 48
            CBAMBlock(48, cbam_reduction, cbam_kernel),
            torch_nn.BatchNorm2d(48, affine=False),

            torch_nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),                       # 128 -> 64
            CBAMBlock(64, cbam_reduction, cbam_kernel),

            torch_nn.MaxPool2d((2, 2), (2, 2)),

            # ---- Block 4 ----
            torch_nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),                       # 128 -> 64
            CBAMBlock(64, cbam_reduction, cbam_kernel),
            torch_nn.BatchNorm2d(64, affine=False),

            torch_nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),                       # 64 -> 32
            CBAMBlock(32, cbam_reduction, cbam_kernel),
            torch_nn.BatchNorm2d(32, affine=False),

            # ---- Block 5 ----
            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),                       # 64 -> 32
            CBAMBlock(32, cbam_reduction, cbam_kernel),
            torch_nn.BatchNorm2d(32, affine=False),

            torch_nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),                       # 64 -> 32
            CBAMBlock(32, cbam_reduction, cbam_kernel),

            torch_nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.Dropout(dropout),
        )

        blstm_dim = (self.num_coefficients // 16) * 32

        self.m_before_pooling = torch_nn.Sequential(
            BLSTMLayer(blstm_dim, blstm_dim),
            BLSTMLayer(blstm_dim, blstm_dim),
        )

        self.m_output_act = torch_nn.Linear(blstm_dim, self.v_emd_dim)

    def _compute_embedding(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 1, 3, 2)
        hidden_features = self.m_transform(x)

        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]
        hidden_features = hidden_features.view(batch_size, frame_num, -1)

        hidden_features_lstm = self.m_before_pooling(hidden_features)
        tmp_emb = self.m_output_act((hidden_features_lstm + hidden_features).mean(1))
        return tmp_emb

    def _compute_score(self, feature_vec):
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        return self._compute_embedding(x)


if __name__ == "__main__":
    print("Definition of LCNN_CBAM model")
    model = LCNN(input_channels=1, num_coefficients=80, dropout=0.4)
    batch_size = 12
    mock_input = torch.rand((batch_size, 1, 80, 404))
    output = model(mock_input)
    print("Output shape:", output.shape)