"""
MSCAR: Multi-Scale Causal Autoregressive Model for TC Intensity Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class Bottleneck(nn.Module):
    """Standard bottleneck block with residual connection"""
    expansion = 2

    def __init__(self, in_ch, out_ch=64, kernel_size=3, stride=1, padding=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction from IR and ERA5"""

    def __init__(self, ir_ch, era5_ch):
        super().__init__()
        # IR branch - deeper processing for higher resolution satellite images
        self.ir_conv1 = nn.Conv2d(ir_ch, 16, 7, 1, 3, bias=False)
        self.ir_bn1 = nn.BatchNorm2d(16)
        self.ir_layer1 = self._make_layer(16, 32, 2)
        self.ir_layer2 = self._make_layer(64, 128, 2, stride=2, down_stride=2)
        self.ir_layer3 = self._make_layer(256, 128, 2, stride=2, down_stride=2)

        # ERA5 branch - simpler processing for lower resolution reanalysis
        self.era5_conv1 = nn.Conv2d(era5_ch, 128, 7, 1, 3, bias=False)
        self.era5_bn1 = nn.BatchNorm2d(128)
        self.era5_layer = self._make_layer(128, 128, 1, kernel_size=6, padding=0,
                                           downsample_kernel=6, down_stride=1)

        # fusion and lateral connections
        self.fuse = nn.Conv2d(512, 256, 1)
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.lateral1 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(256, 256, 1)
        self.lateral3 = nn.Conv2d(64, 256, 1)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, blocks, kernel_size=3, stride=1, padding=1,
                    downsample_kernel=1, down_stride=1):
        downsample = None
        if stride != 1 or in_ch != Bottleneck.expansion * out_ch:
            downsample = nn.Conv2d(in_ch, Bottleneck.expansion * out_ch,
                                   downsample_kernel, down_stride, bias=False)
        layers = [Bottleneck(in_ch, out_ch, kernel_size, stride, padding, downsample)]
        in_ch = out_ch * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_ch, out_ch, padding=padding))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, ir, era5):
        # IR forward pass
        ir = self.ir_bn1(self.ir_conv1(ir))
        c2 = self.ir_layer1(ir)
        c3 = self.ir_layer2(c2)
        c4 = self.ir_layer3(c3)

        # ERA5 forward pass
        era5 = self.era5_bn1(self.era5_conv1(era5))
        era5_feat = self.era5_layer(era5)

        # build pyramid
        c5 = torch.cat([c4, era5_feat], dim=1)
        p5 = self.fuse(c5)
        p4 = self.smooth1(self._upsample_add(p5, self.lateral1(c4)))
        p3 = self.smooth2(self._upsample_add(p4, self.lateral2(c3)))
        p2 = self.smooth3(self._upsample_add(p3, self.lateral3(c2)))
        return p2, p3, p4, p5


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross attention between image features and sequence features"""

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * heads, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim * heads, dim), nn.Dropout(dropout))

    def forward(self, img_feat, seq_feat):
        # query from sequence, key/value from image
        q = self.to_qkv(seq_feat)
        k = self.to_qkv(img_feat)
        v = self.to_qkv(img_feat)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossTransformer(nn.Module):
    """Bidirectional cross-attention between image and sequence"""

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(dim, heads, dropout),
                FeedForward(dim, mlp_dim, dropout),
                CrossAttention(dim, heads, dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))
        self.final_attn = CrossAttention(dim, heads, dropout)
        self.final_ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, img_emb, seq_emb):
        for attn_s, ff_s, attn_i, ff_i in self.layers:
            seq_emb = attn_s(img_emb, seq_emb) + seq_emb
            seq_emb = ff_s(seq_emb) + seq_emb
            img_emb = attn_i(seq_emb, img_emb) + img_emb
            img_emb = ff_i(img_emb) + img_emb
        seq_emb = self.final_attn(img_emb, seq_emb) + seq_emb
        seq_emb = self.final_ff(seq_emb) + seq_emb
        return self.norm(seq_emb)


class SpatioTemporalEncoder(nn.Module):
    """Encodes spatial features over time with causal attention"""

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.cross_transformer = CrossTransformer(dim, depth, heads, mlp_dim, dropout)

    def forward(self, img_emb, seq_emb):
        """
        Args:
            img_emb: (B, T, N, dim) spatial tokens over time
            seq_emb: (B, T, 1, dim) sequence embedding over time
        Returns:
            (B, K, dim) aggregated temporal features where K = T*(T+1)/2
        """
        B, T = img_emb.shape[:2]

        # process t=0
        fusion = torch.cat([img_emb[:, 0], seq_emb[:, 0]], dim=1)
        out = self.cross_transformer(fusion, seq_emb[:, 0])

        # causal aggregation: for each t, attend to all t' <= t
        for t in range(1, T):
            for t_prev in range(t + 1):
                fusion = torch.cat([img_emb[:, t_prev], seq_emb[:, t_prev], seq_emb[:, t]], dim=1)
                out_t = self.cross_transformer(img_emb[:, t_prev], seq_emb[:, t])
                out = torch.cat([out, out_t], dim=1)
        return out


class ARDecoder(nn.Module):
    """Autoregressive decoder - predicts one step at a time"""

    def __init__(self, dim, depth, heads, mlp_dim, seq_len, hidden_dim, seq_dim=2, dropout=0.):
        super().__init__()
        self.cross_transformer = CrossTransformer(dim, depth, heads, mlp_dim, dropout)
        self.seq_att_num = seq_len * (seq_len + 1) // 2
        self.head = nn.Sequential(
            nn.LayerNorm(dim * self.seq_att_num),
            nn.Linear(dim * self.seq_att_num, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_dim),
        )
        self.seq_embed = nn.Linear(seq_dim, dim)

    def forward(self, hidden, last_seq, steps):
        """
        Decode `steps` time steps autoregressively
        Args:
            hidden: (B, K, dim) encoder output
            last_seq: (B, seq_dim) last known intensity value
            steps: number of steps to predict
        """
        pred = last_seq
        for _ in range(steps):
            seq_emb = self.seq_embed(pred).unsqueeze(1)
            cond = torch.cat([seq_emb, hidden], dim=1)
            hidden = self.cross_transformer(cond, hidden)
            flat = hidden.reshape(hidden.shape[0], -1)
            pred = self.head(flat) + pred
        return pred.unsqueeze(1)


class MSCAR(nn.Module):
    """
    Multi-Scale Causal Autoregressive Model

    Predicts tropical cyclone intensity step-by-step using:
    - FPN for multi-scale feature extraction
    - Spatio-temporal transformer for encoding
    - Autoregressive decoder for sequential prediction
    """

    def __init__(self, ir_size=(140, 140), era5_size=(40, 40), patch_size=(5, 5),
                 ir_ch=1, era5_ch=69, dim=128, seq_dim=2, seq_len=4, pred_len=4,
                 depth=3, heads=4, mlp_dim=32, dropout=0.1, ar_hidden=256):
        super().__init__()

        patch_h, patch_w = patch_size
        ih, iw = ir_size

        # feature extraction
        self.fpn = FPN(ir_ch, era5_ch)

        # patch embedding for each pyramid level
        patch_dim = 256 * patch_h * patch_w
        self.to_patch_p2 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)
        self.to_patch_p3 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)
        self.to_patch_p4 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)
        self.to_patch_p5 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)

        # positional embedding
        num_patches = ((ih // patch_h) * (iw // patch_w) +
                       (ih // patch_h // 2) * (iw // patch_w // 2) +
                       2 * (ih // patch_h // 4) * (iw // patch_w // 4))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))

        self.dropout = nn.Dropout(dropout)
        self.seq_embed = nn.Linear(seq_dim, dim)

        # encoder and decoder
        self.encoder = SpatioTemporalEncoder(dim, depth, heads, mlp_dim, dropout)
        self.decoder = ARDecoder(dim, depth, heads, mlp_dim, seq_len, ar_hidden, seq_dim, dropout)

        self.pred_len = pred_len

    def _make_patch_embed(self, ph, pw, patch_dim, dim):
        from einops.layers.torch import Rearrange
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, ir, era5, inp_label, step):
        """
        Args:
            ir: (B, T, C, H, W) satellite IR images
            era5: (B, T, C, H, W) ERA5 reanalysis data
            inp_label: (B, T, 2) input intensity sequence [wind, pressure]
            step: number of steps to predict (1 to pred_len)
        Returns:
            pred: (B, 1, 2) predicted intensity at step
        """
        B, T = ir.shape[:2]

        # flatten time for FPN processing
        ir_flat = ir.reshape(B * T, *ir.shape[2:])
        era5_flat = era5.reshape(B * T, *era5.shape[2:])
        p2, p3, p4, p5 = self.fpn(ir_flat, era5_flat)

        # patch embedding
        p2 = self.to_patch_p2(p2)
        p3 = self.to_patch_p3(p3)
        p4 = self.to_patch_p4(p4)
        p5 = self.to_patch_p5(p5)
        tokens = torch.cat([p2, p3, p4, p5], dim=1)
        tokens = tokens.reshape(B, T, tokens.shape[-2], tokens.shape[-1])

        # add cls token and positional embedding
        cls = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=B, t=T)
        tokens = torch.cat([cls, tokens], dim=-2)
        tokens = self.dropout(tokens + self.pos_embed)

        # sequence embedding
        seq = self.seq_embed(inp_label).unsqueeze(-2)

        # encode and decode
        hidden = self.encoder(tokens, seq)
        pred = self.decoder(hidden.clone(), inp_label[:, -1].clone(), step)

        return pred


if __name__ == "__main__":
    # quick test
    model = MSCAR(patch_size=(7, 7), ar_hidden=128)
    ir = torch.randn(2, 4, 1, 140, 140)
    era5 = torch.randn(2, 4, 69, 40, 40)
    seq = torch.randn(2, 4, 2)
    out = model(ir, era5, seq, step=1)
    print(f"Output shape: {out.shape}")  # should be (2, 1, 2)
