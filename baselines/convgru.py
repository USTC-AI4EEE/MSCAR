"""
ConvGRU Baseline for TC Intensity Prediction

Reference: "A neural network framework for fine-grained tropical cyclone
           intensity prediction" (Knowledge-Based Systems, 2022)

This is the best-performing variant from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """Single ConvGRU cell - captures spatial structure in recurrence"""

    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel // 2

        # reset and update gates
        self.conv_gates = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 2, kernel, padding=pad)
        # candidate hidden state
        self.conv_cand = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel, padding=pad)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)

        gates = self.conv_gates(combined)
        r, z = gates.chunk(2, dim=1)
        r, z = torch.sigmoid(r), torch.sigmoid(z)

        cand = torch.tanh(self.conv_cand(torch.cat([x, r * h_prev], dim=1)))
        h_new = (1 - z) * h_prev + z * cand
        return h_new

    def init_hidden(self, B, H, W, device):
        return torch.zeros(B, self.hidden_ch, H, W, device=device)


class ConvGRU(nn.Module):
    """Multi-layer ConvGRU"""

    def __init__(self, in_ch, hidden_ch, n_layers=1, kernel=3):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_ch = hidden_ch

        cells = []
        for i in range(n_layers):
            cells.append(ConvGRUCell(in_ch if i == 0 else hidden_ch, hidden_ch, kernel))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, hidden=None):
        """x: (B, T, C, H, W)"""
        B, T, C, H, W = x.shape

        if hidden is None:
            hidden = [cell.init_hidden(B, H, W, x.device) for cell in self.cells]

        outputs = []
        for t in range(T):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(x_t, hidden[i])
                x_t = hidden[i]
            outputs.append(x_t)

        return torch.stack(outputs, dim=1), hidden


class FeatureAggregator(nn.Module):
    """Spatial feature extraction from IR + ERA5"""

    def __init__(self, ir_ch=1, era5_ch=69, hidden=64, spatial=16):
        super().__init__()
        self.spatial = spatial

        self.ir_enc = nn.Sequential(
            nn.Conv2d(ir_ch, 32, 7, 2, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )

        self.era5_enc = nn.Sequential(
            nn.Conv2d(era5_ch, hidden, 3, 1, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(spatial)

    def forward(self, ir, era5):
        ir_f = self.pool(self.ir_enc(ir))
        era5_f = self.pool(self.era5_enc(era5))
        combined = torch.cat([self.alpha * ir_f, self.beta * era5_f], dim=1)
        return self.fuse(combined)


class FeatureEnhancer(nn.Module):
    """Attention-based feature enhancement between encoder and decoder"""

    def __init__(self, ch, spatial=16):
        super().__init__()
        self.conv_q = nn.Conv2d(ch, ch, 1)
        self.conv_k = nn.Conv2d(ch, ch, 1)
        self.conv_v = nn.Conv2d(ch, ch, 1)

        self.attn_conv = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1),
        )

        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, enc_out, dec_out):
        """
        enc_out: (B, T_enc, C, H, W)
        dec_out: (B, T_dec, C, H, W)
        """
        B, T_dec, C, H, W = dec_out.shape
        T_enc = enc_out.shape[1]

        enhanced = []
        for t in range(T_dec):
            q = self.conv_q(dec_out[:, t])

            # aggregate attention over encoder states
            attn_sum = 0
            for t_enc in range(T_enc):
                k = self.conv_k(enc_out[:, t_enc])
                v = self.conv_v(enc_out[:, t_enc])

                qk = torch.cat([q, k], dim=1)
                w = torch.sigmoid(self.attn_conv(qk))
                attn_sum = attn_sum + w * v

            attn_sum = attn_sum / T_enc
            h = self.norm1(q + attn_sum)
            h = self.norm2(h + self.out_conv(h))
            enhanced.append(h)

        return torch.stack(enhanced, dim=1)


class ConvGRU(nn.Module):
    """
    ConvGRU model for TC intensity prediction

    Uses spatial ConvGRU cells + feature enhancement attention
    """

    def __init__(self, ir_ch=1, era5_ch=69, seq_dim=2, hidden=64,
                 gru_hidden=64, n_layers=2, spatial=16, input_len=4, output_len=4, dropout=0.1):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.seq_dim = seq_dim
        self.spatial = spatial

        self.feature_agg = FeatureAggregator(ir_ch, era5_ch, hidden, spatial)

        # sequence embedding broadcast to spatial
        self.seq_embed = nn.Sequential(nn.Linear(seq_dim, hidden), nn.ReLU())
        self.combine = nn.Conv2d(hidden * 2, hidden, 1)

        # encoder-decoder
        self.encoder = ConvGRU_(hidden, gru_hidden, n_layers)
        self.decoder = ConvGRU_(gru_hidden, gru_hidden, n_layers)
        self.enhance = FeatureEnhancer(gru_hidden, spatial)

        # output head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, seq_dim),
        )

    def forward(self, ir, era5, inp_seq, step=None):
        if step is None:
            step = self.output_len

        B, T = ir.shape[:2]

        # extract spatial features per timestep
        feats = []
        for t in range(T):
            spatial_f = self.feature_agg(ir[:, t], era5[:, t])

            # broadcast sequence to spatial
            seq_f = self.seq_embed(inp_seq[:, t])
            seq_f = seq_f.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.spatial, self.spatial)

            combined = torch.cat([spatial_f, seq_f], dim=1)
            feats.append(self.combine(combined))

        feats = torch.stack(feats, dim=1)  # (B, T, C, H, W)

        # encode
        enc_out, hidden = self.encoder(feats)

        # decode with zeros
        dec_in = torch.zeros(B, step, hidden[0].shape[1], self.spatial, self.spatial, device=ir.device)
        dec_out, _ = self.decoder(dec_in, hidden)

        # enhance and predict
        enhanced = self.enhance(enc_out, dec_out)

        preds = []
        for t in range(step):
            preds.append(self.head(enhanced[:, t]))

        return torch.stack(preds, dim=1)


# internal ConvGRU class (renamed to avoid conflict)
class ConvGRU_(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_layers=1, kernel=3):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_ch = hidden_ch

        cells = []
        for i in range(n_layers):
            cells.append(ConvGRUCell(in_ch if i == 0 else hidden_ch, hidden_ch, kernel))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, hidden=None):
        B, T, C, H, W = x.shape
        if hidden is None:
            hidden = [cell.init_hidden(B, H, W, x.device) for cell in self.cells]

        outputs = []
        for t in range(T):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(x_t, hidden[i])
                x_t = hidden[i]
            outputs.append(x_t)

        return torch.stack(outputs, dim=1), hidden


if __name__ == '__main__':
    model = ConvGRU()
    ir = torch.randn(2, 4, 1, 140, 140)
    era5 = torch.randn(2, 4, 69, 40, 40)
    seq = torch.randn(2, 4, 2)
    out = model(ir, era5, seq)
    print(f"Output: {out.shape}")  # (2, 4, 2)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
