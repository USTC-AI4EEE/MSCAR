"""
MSCAR: Multi-source Multi-Scale Constrained AutoRegressive model for TC intensity forecasting.
MSCAR：多源多尺度约束自回归热带气旋强度预报模型。

Encoder (FPN + tokens) -> CC Attention (temporal constraint) -> AR Decoder.
编码器（FPN + token）-> 约束交叉注意力（时间约束）-> 自回归解码器。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Bottleneck(nn.Module):
    """Bottleneck residual block.
    带残差连接的瓶颈块。"""
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
    """Feature pyramid network fusing IR and ERA5 into multi-scale features.
    特征金字塔网络，融合红外与 ERA5 得到多尺度特征。"""

    def __init__(self, ir_ch, era5_ch):
        super().__init__()
        # IR branch: deeper, for high-resolution satellite images
        # 红外分支：较深，处理高分辨率卫星图像
        self.ir_conv1 = nn.Conv2d(ir_ch, 16, 7, 1, 3, bias=False)
        self.ir_bn1 = nn.BatchNorm2d(16)
        self.ir_layer1 = self._make_layer(16, 32, 2)
        self.ir_layer2 = self._make_layer(64, 128, 2, stride=2, down_stride=2)
        self.ir_layer3 = self._make_layer(256, 128, 2, stride=2, down_stride=2)

        # ERA5 branch: shallow, for low-resolution reanalysis
        # ERA5 分支：较浅，处理低分辨率再分析场
        self.era5_conv1 = nn.Conv2d(era5_ch, 128, 7, 1, 3, bias=False)
        self.era5_bn1 = nn.BatchNorm2d(128)
        self.era5_layer = self._make_layer(128, 128, 1, kernel_size=6, padding=0,
                                           downsample_kernel=6, down_stride=1)

        # Fusion and lateral connections
        # 融合层与横向连接
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
        # IR: 140 -> 140 / 70 / 35
        # 红外：140 -> 140 / 70 / 35
        ir = self.ir_bn1(self.ir_conv1(ir))
        c2 = self.ir_layer1(ir)
        c3 = self.ir_layer2(c2)
        c4 = self.ir_layer3(c3)

        # ERA5: 40 -> 35, aligned with the coarsest IR level
        # ERA5：40 -> 35，与红外最粗尺度对齐
        era5 = self.era5_bn1(self.era5_conv1(era5))
        era5_feat = self.era5_layer(era5)

        # Top-down pyramid p5 -> p2
        # 自顶向下构建金字塔 p5 -> p2
        c5 = torch.cat([c4, era5_feat], dim=1)
        p5 = self.fuse(c5)
        p4 = self.smooth1(self._upsample_add(p5, self.lateral1(c4)))
        p3 = self.smooth2(self._upsample_add(p4, self.lateral2(c3)))
        p2 = self.smooth3(self._upsample_add(p3, self.lateral3(c2)))
        return p2, p3, p4, p5


class FeedForward(nn.Module):
    """Pre-norm MLP.
    前置 LayerNorm 的 MLP。"""

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
    """Cross attention: query from the second input, key/value from the first.
    交叉注意力：query 来自第二个输入，key/value 来自第一个输入。"""

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * heads, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim * heads, dim), nn.Dropout(dropout))

    def forward(self, img_feat, seq_feat):
        # One shared projection for q, k and v (as in the original model)
        # q、k、v 共用同一投影（与原始模型一致）
        q = self.to_qkv(seq_feat)
        k = self.to_qkv(img_feat)
        v = self.to_qkv(img_feat)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossTransformer(nn.Module):
    """Alternating cross attention between spatial and intensity tokens.
    空间 token 与强度 token 之间交替进行交叉注意力。"""

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
            # Intensity tokens attend to spatial tokens / 强度 token 关注空间 token
            seq_emb = attn_s(img_emb, seq_emb) + seq_emb
            seq_emb = ff_s(seq_emb) + seq_emb
            # Spatial tokens attend to intensity tokens / 空间 token 关注强度 token
            img_emb = attn_i(seq_emb, img_emb) + img_emb
            img_emb = ff_i(img_emb) + img_emb
        seq_emb = self.final_attn(img_emb, seq_emb) + seq_emb
        seq_emb = self.final_ff(seq_emb) + seq_emb
        return self.norm(seq_emb)


class SpatioTemporalEncoder(nn.Module):
    """CC Attention: intensity at time t attends to spatial tokens at times p <= t.
    约束交叉注意力：t 时刻的强度只关注 p <= t 时刻的空间 token。"""

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.cross_transformer = CrossTransformer(dim, depth, heads, mlp_dim, dropout)

    def forward(self, img_emb, seq_emb):
        """
        Args:
            img_emb: (B, T, N, dim) spatial tokens / 空间 token
            seq_emb: (B, T, 1, dim) intensity tokens / 强度 token
        Returns:
            (B, K, dim) hidden state, K = T*(T+1)/2 (lower-triangular pairs)
            (B, K, dim) 隐状态，K = T*(T+1)/2（下三角时间对）
        """
        B, T = img_emb.shape[:2]

        # t = 0: spatial tokens and intensity token are fused together
        # t = 0：空间 token 与强度 token 拼接融合
        fusion = torch.cat([img_emb[:, 0], seq_emb[:, 0]], dim=1)
        out = self.cross_transformer(fusion, seq_emb[:, 0])

        # Pairs (t, p) with p <= t, computed in one batch; order matches the paper
        # 所有 p <= t 的时间对 (t, p) 批量计算，顺序与论文一致
        pairs = [(t, p) for t in range(1, T) for p in range(t + 1)]
        if pairs:
            images = torch.stack([img_emb[:, p] for t, p in pairs], dim=1)
            queries = torch.stack([seq_emb[:, t] for t, p in pairs], dim=1)
            count = len(pairs)
            fused = self.cross_transformer(images.flatten(0, 1), queries.flatten(0, 1))
            out = torch.cat([out, fused.reshape(B, count, -1)], dim=1)
        return out


class ARDecoder(nn.Module):
    """Autoregressive decoder: one 6h step per iteration, residual on the previous intensity.
    自回归解码器：每次迭代预测一个 6h 步，在上一步强度上做残差预测。"""

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

    def forward(self, hidden, last_seq, steps, return_sequence=False):
        """
        Args:
            hidden: (B, K, dim) encoder output / 编码器输出
            last_seq: (B, 2) last observed intensity / 最后一个观测强度
            steps: number of 6h steps / 预测的 6h 步数
            return_sequence: return all steps (B, steps, 2), else the last one (B, 1, 2)
                             返回所有步 (B, steps, 2)，否则只返回最后一步 (B, 1, 2)
        """
        if steps < 1:
            raise ValueError('steps must be positive')
        pred = last_seq
        predictions = []
        for _ in range(steps):
            # Feed back the previous prediction / 将上一步预测作为输入
            seq_emb = self.seq_embed(pred).unsqueeze(1)
            cond = torch.cat([seq_emb, hidden], dim=1)
            hidden = self.cross_transformer(cond, hidden)
            flat = hidden.reshape(hidden.shape[0], -1)
            pred = self.head(flat) + pred
            predictions.append(pred)
        return torch.stack(predictions, dim=1) if return_sequence else pred.unsqueeze(1)


class MSCAR(nn.Module):
    """
    MSCAR: forecast TC intensity [wind, pressure] from the past 4 time steps.
    MSCAR：根据过去 4 个时刻预报热带气旋强度 [风速, 气压]。

    pred_len=4 -> 24h (conference), pred_len=20 -> 120h (journal).
    pred_len=4 对应 24h（会议版），pred_len=20 对应 120h（期刊版）。
    """

    def __init__(self, ir_size=(140, 140), era5_size=(40, 40), patch_size=(5, 5),
                 ir_ch=1, era5_ch=69, dim=128, seq_dim=2, seq_len=4, pred_len=4,
                 depth=3, heads=4, mlp_dim=32, dropout=0.1, ar_hidden=256):
        super().__init__()

        # Sizes may come from YAML lists / 尺寸参数可能来自 YAML 列表
        ir_size, era5_size, patch_size = tuple(ir_size), tuple(era5_size), tuple(patch_size)
        patch_h, patch_w = patch_size
        ih, iw = ir_size

        # Multi-scale feature extraction / 多尺度特征提取
        self.fpn = FPN(ir_ch, era5_ch)

        # Patch embedding for each pyramid level / 每个金字塔层的 patch 嵌入
        patch_dim = 256 * patch_h * patch_w
        self.to_patch_p2 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)
        self.to_patch_p3 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)
        self.to_patch_p4 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)
        self.to_patch_p5 = self._make_patch_embed(patch_h, patch_w, patch_dim, dim)

        # Positional embedding over all levels (+1 cls token)
        # 所有层的位置编码（+1 个 cls token）
        num_patches = ((ih // patch_h) * (iw // patch_w) +
                       (ih // patch_h // 2) * (iw // patch_w // 2) +
                       2 * (ih // patch_h // 4) * (iw // patch_w // 4))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))

        self.dropout = nn.Dropout(dropout)
        self.seq_embed = nn.Linear(seq_dim, dim)

        # Encoder and decoder / 编码器与解码器
        self.encoder = SpatioTemporalEncoder(dim, depth, heads, mlp_dim, dropout)
        self.decoder = ARDecoder(dim, depth, heads, mlp_dim, seq_len, ar_hidden, seq_dim, dropout)

        self.pred_len = pred_len

    def _make_patch_embed(self, ph, pw, patch_dim, dim):
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def encode(self, ir, era5, inp_label):
        """
        Args:
            ir: (B, T, 1, 140, 140) IR images / 红外图像
            era5: (B, T, 69, 40, 40) ERA5 fields / ERA5 场
            inp_label: (B, T, 2) past intensity [wind, pressure] / 历史强度 [风速, 气压]
        Returns:
            hidden: (B, K, dim) / 隐状态
        """
        B, T = ir.shape[:2]

        # Merge batch and time for the FPN / 合并批次与时间维送入 FPN
        ir_flat = ir.reshape(B * T, *ir.shape[2:])
        era5_flat = era5.reshape(B * T, *era5.shape[2:])
        p2, p3, p4, p5 = self.fpn(ir_flat, era5_flat)

        # Patch embedding / patch 嵌入
        p2 = self.to_patch_p2(p2)
        p3 = self.to_patch_p3(p3)
        p4 = self.to_patch_p4(p4)
        p5 = self.to_patch_p5(p5)
        tokens = torch.cat([p2, p3, p4, p5], dim=1)
        tokens = tokens.reshape(B, T, tokens.shape[-2], tokens.shape[-1])

        # Add cls token and positional embedding / 加入 cls token 与位置编码
        cls = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=B, t=T)
        tokens = torch.cat([cls, tokens], dim=-2)
        tokens = self.dropout(tokens + self.pos_embed)

        # Intensity tokens / 强度 token
        seq = self.seq_embed(inp_label).unsqueeze(-2)

        # CC Attention / 约束交叉注意力
        return self.encoder(tokens, seq)

    def forward(self, ir, era5, inp_label, step=None, return_sequence=False):
        """
        Encode once, then roll out `step` steps autoregressively (past inputs only).
        Returns (B, 1, 2) for the last step, or (B, step, 2) if return_sequence=True.
        编码一次后自回归预报 `step` 步（只使用历史输入）。
        默认返回最后一步 (B, 1, 2)；return_sequence=True 时返回全部 (B, step, 2)。
        """
        steps = self.pred_len if step is None else int(step)
        hidden = self.encode(ir, era5, inp_label)
        return self.decoder(hidden, inp_label[:, -1], steps, return_sequence)


if __name__ == "__main__":
    # Shape check / 形状检查
    model = MSCAR(pred_len=20).eval()
    ir = torch.randn(2, 4, 1, 140, 140)
    era5 = torch.randn(2, 4, 69, 40, 40)
    seq = torch.randn(2, 4, 2)
    print(model(ir, era5, seq, return_sequence=True).shape)  # (2, 20, 2)
