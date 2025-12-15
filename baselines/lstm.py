import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAggregator(nn.Module):
    """Extract and fuse features from IR and ERA5"""

    def __init__(self, ir_ch=1, era5_ch=69, hidden=128, out_dim=256):
        super().__init__()

        # IR encoder (satellite)
        self.ir_enc = nn.Sequential(
            nn.Conv2d(ir_ch, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden, 3, 2, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # ERA5 encoder (atmosphere)
        self.era5_enc = nn.Sequential(
            nn.Conv2d(era5_ch, hidden, 3, 1, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, ir, era5):
        ir_feat = self.ir_enc(ir).flatten(1)
        era5_feat = self.era5_enc(era5).flatten(1)
        return self.fuse(torch.cat([ir_feat, era5_feat], dim=1))


class LSTMEncoder(nn.Module):
    def __init__(self, in_dim, hidden, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out, (h, c)


class LSTMDecoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x, hidden):
        out, _ = self.lstm(x, hidden)
        return self.fc(out)


class LSTM(nn.Module):
    """
    Encoder-Decoder LSTM for TC intensity prediction

    Input: 4 time steps of IR + ERA5 + intensity
    Output: 4 time steps of intensity prediction
    """

    def __init__(self, ir_ch=1, era5_ch=69, seq_dim=2, hidden=256,
                 lstm_hidden=128, n_layers=2, input_len=4, output_len=4, dropout=0.1):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.seq_dim = seq_dim

        self.feature_agg = FeatureAggregator(ir_ch, era5_ch, 128, hidden)
        self.seq_embed = nn.Linear(seq_dim, hidden // 2)

        enc_in = hidden + hidden // 2
        self.encoder = LSTMEncoder(enc_in, lstm_hidden, n_layers, dropout)
        self.dec_embed = nn.Linear(seq_dim, lstm_hidden)
        self.decoder = LSTMDecoder(lstm_hidden, lstm_hidden, seq_dim, n_layers, dropout)

    def forward(self, ir, era5, inp_seq, step=None):
        """
        ir: (B, T, C, H, W)
        era5: (B, T, C, H, W)
        inp_seq: (B, T, seq_dim)
        """
        if step is None:
            step = self.output_len

        B, T = ir.shape[:2]

        # extract features per timestep
        feats = []
        for t in range(T):
            feat = self.feature_agg(ir[:, t], era5[:, t])
            feats.append(feat)
        spatial = torch.stack(feats, dim=1)  # (B, T, hidden)

        seq = self.seq_embed(inp_seq)  # (B, T, hidden/2)
        combined = torch.cat([spatial, seq], dim=-1)

        # encode
        _, (h, c) = self.encoder(combined)

        # decode with zeros as input (unconditional)
        dec_in = torch.zeros(B, step, self.decoder.lstm.input_size, device=ir.device)
        pred = self.decoder(dec_in, (h, c))

        return pred


if __name__ == '__main__':
    model = LSTM()
    ir = torch.randn(2, 4, 1, 140, 140)
    era5 = torch.randn(2, 4, 69, 40, 40)
    seq = torch.randn(2, 4, 2)
    out = model(ir, era5, seq)
    print(f"Output: {out.shape}")  # (2, 4, 2)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
