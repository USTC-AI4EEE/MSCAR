"""
Transformer Baseline for TC Intensity Prediction

Reference: "Transformer-based tropical cyclone track and intensity forecasting"
           (Journal of Wind Engineering, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SpatialEncoder(nn.Module):
    """Extract features from IR and ERA5"""

    def __init__(self, ir_ch=1, era5_ch=69, hidden=128):
        super().__init__()

        self.ir_enc = nn.Sequential(
            nn.Conv2d(ir_ch, 32, 7, 2, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.era5_enc = nn.Sequential(
            nn.Conv2d(era5_ch, hidden, 3, 1, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, ir, era5):
        ir_f = self.ir_enc(ir).flatten(1)
        era5_f = self.era5_enc(era5).flatten(1)
        return torch.cat([ir_f, era5_f], dim=-1)


class TCTransformer(nn.Module):
    """
    Transformer for TC intensity prediction

    Uses spatial features + sequence embedding, then transformer encoder
    """

    def __init__(self, ir_ch=1, era5_ch=69, seq_dim=2, d_model=256,
                 n_heads=4, n_layers=3, d_ff=512, input_len=4, output_len=4, dropout=0.1):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.seq_dim = seq_dim

        self.spatial_enc = SpatialEncoder(ir_ch, era5_ch, 128)
        self.spatial_proj = nn.Linear(256, d_model // 2)  # 128 + 128 = 256
        self.seq_embed = nn.Linear(seq_dim, d_model // 2)
        self.feature_proj = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, input_len + output_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.output = nn.Sequential(
            nn.Linear(d_model * input_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len * seq_dim),
        )

    def forward(self, ir, era5, inp_seq, step=None):
        if step is None:
            step = self.output_len

        B, T = ir.shape[:2]

        # spatial features per timestep
        spatial = []
        for t in range(T):
            feat = self.spatial_enc(ir[:, t], era5[:, t])
            spatial.append(self.spatial_proj(feat))
        spatial = torch.stack(spatial, dim=1)  # (B, T, d_model/2)

        seq = self.seq_embed(inp_seq)  # (B, T, d_model/2)
        combined = self.feature_proj(torch.cat([spatial, seq], dim=-1))
        combined = self.pos_enc(combined)

        encoded = self.transformer(combined)
        flat = encoded.reshape(B, -1)
        out = self.output(flat).view(B, self.output_len, self.seq_dim)

        return out[:, :step]


if __name__ == '__main__':
    model = TCTransformer()
    ir = torch.randn(2, 4, 1, 140, 140)
    era5 = torch.randn(2, 4, 69, 40, 40)
    seq = torch.randn(2, 4, 2)
    out = model(ir, era5, seq)
    print(f"Output: {out.shape}")  # (2, 4, 2)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
