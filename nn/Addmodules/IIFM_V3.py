import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv  import *
# ------------------------
# Scale module
# ------------------------
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return x * self.weight

# ------------------------
# Depthwise conv
# ------------------------
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=bias)
    def forward(self, x):
        return self.depthwise(x)

# ------------------------
# Lightweight Adaptive Filter
# ------------------------
class LightweightAdaptiveFilter(nn.Module):
    def __init__(self, c, max_k=5, reduction=8):
        super().__init__()
        self.c = c
        self.max_k = max_k
        red = max(1, c // reduction)
        self.kernel_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, (max_k + 1)//2, 1, bias=False)
        )
        self.depthwise_convs = nn.ModuleList([
            DepthwiseConv(c, kernel_size=k, padding=k//2) for k in range(1, max_k+1, 2)
        ])
    def forward(self, feat):
        B, c, H, W = feat.shape
        kernels = self.kernel_pred(feat).view(B, -1)
        kernels = F.softmax(kernels, dim=1)
        outs = []
        for idx, conv in enumerate(self.depthwise_convs):
            out_k = conv(feat)
            outs.append(out_k * kernels[:, idx].view(B, 1, 1, 1))
        return sum(outs)

# ------------------------
# DynamicTPA (lightweight)
# ------------------------
class DynamicTPA_FGCA(nn.Module):
    def __init__(self, c1, c2, max_rank):
        super().__init__()
        self.max_rank = max_rank
        self.A = Conv(c1, max_rank, 1, act=False)
        self.B = Conv(c1, max_rank, 1, act=False)
        self.output = nn.Conv2d(max_rank * max_rank, c2, kernel_size=1, bias=True)

        # === FGCA风格的方向特征提取 ===
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽方向池化（关注垂直结构）
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高方向池化（关注水平结构）
        self.fuse_hw = Conv(c1, c1, 1, act=True)   # 融合方向信息
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1?? FGCA风格方向感知
        x_h = self.pool_h(x)  # [B,C,H,1]
        x_w = self.pool_w(x).permute(0,1,3,2)  # [B,C,W,1]
        hw_feat = torch.cat([x_h, x_w], dim=2)
        hw_feat = self.fuse_hw(hw_feat)
        x_enhanced = x * hw_feat.sigmoid()  # 空间方向增强后的特征

        # 2?? 动态rank控制（融合方向感知）
        gate_val = self.gate(x_enhanced).mean()
        rank = int(gate_val.item() * self.max_rank)
        rank = max(1, min(rank, self.max_rank))

        # 3?? Outer Product 注意力
        a = self.A(x_enhanced)[:, :rank]
        b = self.B(x_enhanced)[:, :rank]
        ab = torch.einsum('bihw,bjhw->bijhw', a, b)
        ab = ab.reshape(B, rank * rank, H, W)

        weight = self.output.weight[:, :rank * rank]
        bias = self.output.bias
        out = F.conv2d(ab, weight, bias)
        return out

# ------------------------
# WTConv2d (lazy wavelet conv)
# ------------------------
class WTConv2d(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super().__init__()
        self._in_ch = in_channels
        self._out_ch = out_channels if out_channels is not None else in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.wt_levels = wt_levels
        self.wt_type = wt_type
        self.initialized = False
    def _init_layers(self, in_channels, out_channels=None):
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_conv = nn.Conv2d(in_channels, out_channels, self.kernel_size,
                                   padding=self.kernel_size//2, stride=1, bias=self.bias)
        self.dw = DepthwiseConv(in_channels, kernel_size=3, padding=1, bias=False)
        self.scale = _ScaleModule([1, out_channels, 1, 1], init_scale=0.1)
        if self.stride > 1:
            self.do_stride = lambda x_in: F.avg_pool2d(x_in, kernel_size=self.stride, stride=self.stride)
        else:
            self.do_stride = None
        self.initialized = True
    def forward(self, x):
        if not self.initialized:
            ch = x.shape[1]
            if ch <= 0:
                return x
            self._init_layers(ch, ch)
        tag = self.dw(x)
        tag = self.scale(tag)
        x_out = self.base_conv(x)
        x = x_out + tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x

# ------------------------
# IIFM_V3 Full version (weak-light auto-adaptive)
# ------------------------
class IIFM_V3_Feat(nn.Module):
    def __init__(self, c1, c2=None, fusion="replace", reduction=8,
                 max_rank=8, filter_k=5, wt_levels=1, wt_type='db1'):
        super().__init__()
        assert fusion in ("replace", "add", "concat")
        self.fusion = fusion
        self.c1, self.c2 = c1, c2

        # 基础特征提取
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.in1 = nn.InstanceNorm2d(c2, affine=True)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(c2, affine=True)

        # 光照 mask
        red = max(1, c2 // reduction)
        self.mask_branch = nn.Sequential(
            nn.Conv2d(c2, red, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(red, c2, 1, bias=False),
            nn.Tanh()
        )

        # 局部修复
        self.restore = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        )

        # DynamicTPA + AdaptiveFilter
        self.dynamic_tpa = DynamicTPA_FGCA(c2, c2, max_rank)
        self.adaptive_filter = LightweightAdaptiveFilter(c2, max_k=filter_k, reduction=reduction)

        # 小波增强
        self.restore_wt = WTConv2d(in_channels=None, out_channels=None,
                                   wt_levels=wt_levels, wt_type=wt_type)

        # 输出融合
        if fusion == "concat":
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(c1 + c2, c2, 1, bias=False),
                nn.InstanceNorm2d(c2, affine=True),
                nn.SiLU()
            )
        elif fusion == "add" and c1 != c2:
            self.project = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.project = None

        # 弱光增强可训练参数
        self.weak_gamma = nn.Parameter(torch.ones(1, c2, 1, 1) * 1.5)

    def forward(self, x):
        out = self.act(self.in1(self.conv1(x)))
        out = self.act(self.in2(self.conv2(out)))

        mask = self.mask_branch(out)
        mask_pos = torch.clamp(mask, min=0)
        mask_neg = torch.clamp(-mask, min=0)
        mask_neg = mask_neg ** (1 + torch.sigmoid(self.weak_gamma))
        out_weak = out * (1 + mask_neg)

        # 局部修复 + WT
        local = self.restore(out_weak)
        wt_out = self.restore_wt(local)

        # DynamicTPA gate 弱光加权 + 防 NaN
        gate = self.dynamic_tpa.gate(wt_out)
        gate_weighted = gate * (1 + mask_neg)
        gate_weighted = torch.nan_to_num(gate_weighted, nan=0.5, posinf=1.0, neginf=0.0)
        gate_val = gate_weighted.mean().clamp(0.0, 1.0).item()
        rank = max(1, min(int(gate_val * self.dynamic_tpa.max_rank), self.dynamic_tpa.max_rank))

        a = self.dynamic_tpa.A(wt_out)[:, :rank]
        b = self.dynamic_tpa.B(wt_out)[:, :rank]
        ab = torch.einsum('bihw,bjhw->bijhw', a, b).reshape(wt_out.shape[0], rank*rank, wt_out.shape[2], wt_out.shape[3])
        weight = self.dynamic_tpa.output.weight[:, :rank*rank, :, :]
        bias = self.dynamic_tpa.output.bias
        att_out = F.conv2d(ab, weight, bias)

        filt_out = self.adaptive_filter(wt_out)
        fused = att_out + filt_out

        out_final = out * (1 + mask_neg) + fused * mask_pos

        if self.fusion == "replace":
            return out_final
        elif self.fusion == "add":
            xp = self.project(x) if self.project is not None else x
            return xp + out_final
        else:
            return self.fuse_conv(torch.cat([x, out_final], dim=1))

# ------------------------
# quick sanity test
# ------------------------
if __name__ == "__main__":
    B, C_in, H, W = 2, 32, 64, 64
    module = IIFM_V3_Feat_AutoWeak(C_in, c2=32, fusion='replace')
    x = torch.randn(B, C_in, H, W)
    y = module(x)
    print('in', x.shape, 'out', y.shape)
    assert not torch.isnan(y).any()
    print('sanity check passed')
