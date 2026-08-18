# iifm_wt_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pywt

# ------------------------
# Small helper: Scale module
# ------------------------
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        # dims: a list/tuple for parameter shape, e.g. [1, C, 1, 1]
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return x * self.weight

# ------------------------
# Wavelet filter creation + forward/backward helpers
# ------------------------
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float32):
    """
    Create 2D separable wavelet filters for conv (dec / rec).
    Returns dec_filters and rec_filters shaped for conv2d usage (see below).
    """
    w = pywt.Wavelet(wave)
    # dec (analysis) filter coefficients
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)  # low pass
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)  # high pass
    # form 2D separable filters: LL, LH, HL, HH (order chosen as in earlier code)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)  # [4, K, K]

    # Repeat for channels
    dec_filters = dec_filters[None].repeat(in_size, 1, 1, 1)  # [in_size, 4, K, K]
    # reshape to conv weight shape [in_size*4, 1, K, K] for grouped convs (we will use groups=in_size)
    # but we will reformat inside transform functions

    # rec (synthesis) filters (reverse/appropriate orientation)
    # use reconstruction coeffs reversed appropriately
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype)
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    """
    x: [B, C, H, W]
    filters: dec_filters from create_wavelet_filter with shape [C, 4, K, K]
    Output: [B, C, 4, H//2, W//2]
    Implementation: per-channel grouped conv2d with stride=2.
    """
    B, C, H, W = x.shape
    # filters currently [C, 4, K, K] -> reshape to [C*4, 1, K, K] and use groups=C
    C_f, four, K, K2 = filters.shape
    assert C_f == C
    f = filters.view(C * 4, 1, K, K2).to(x.device).to(x.dtype)
    pad = (K // 2 - 1, K // 2 - 1)
    # conv2d with groups=C, input channels=C, weight channels=C*4: use grouped conv by splitting input channels
    # Trick: use F.conv2d with groups=C by reshaping weight to [C*4, 1, K, K] and setting groups=C
    out = F.conv2d(x, f, bias=None, stride=2, padding=pad, groups=C)
    # out shape: [B, C*4, H//2, W//2]
    out = out.view(B, C, 4, out.shape[-2], out.shape[-1])
    return out

def inverse_wavelet_transform(x, filters):
    """
    x: [B, C, 4, H_half, W_half]
    filters: rec_filters shape [C, 4, K, K]
    Returns: [B, C, H, W] via grouped conv_transpose
    """
    B, C, four, Hh, Wh = x.shape
    assert four == 4
    K = filters.shape[2]
    f = filters.view(C * 4, 1, K, K).to(x.device).to(x.dtype)
    inp = x.view(B, C * 4, Hh, Wh)
    pad = (K // 2 - 1, K // 2 - 1)
    out = F.conv_transpose2d(inp, f, bias=None, stride=2, padding=pad, groups=C)
    return out

class WaveletTransform(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.filters = filters
        with torch.no_grad():
            x = wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = inverse_wavelet_transform(grad_output, ctx.filters)
        # second return None for filters
        return grad, None

class InverseWaveletTransform(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.filters = filters
        with torch.no_grad():
            x = inverse_wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = wavelet_transform(grad_output, ctx.filters)
        return grad, None

def wavelet_transform_init(filters):
    def apply(x):
        return WaveletTransform.apply(x, filters)
    return apply

def inverse_wavelet_transform_init(filters):
    def apply(x):
        return InverseWaveletTransform.apply(x, filters)
    return apply

# ------------------------
# DynamicTPA (as you defined)
# ------------------------
class DynamicTPA(nn.Module):
    def __init__(self, c1, c2, max_rank):
        super(DynamicTPA, self).__init__()
        self.max_rank = max_rank
        self.A = nn.Conv2d(c1, max_rank, 1, bias=False)
        self.B = nn.Conv2d(c1, max_rank, 1, bias=False)
        self.output = nn.Conv2d(max_rank * max_rank, c2, kernel_size=1, bias=True)
        self.gate = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        B, C, H, W = x.shape
        gate_value = self.gate(x).squeeze()  # [B, C] or [C]
        if gate_value.numel() == 0 or torch.isnan(gate_value).any() or torch.isinf(gate_value).any():
            mean_gate = torch.tensor(0.5, device=x.device)
        else:
            gate_value = torch.clamp(gate_value, -10, 10)
            gate_value = torch.sigmoid(gate_value)
            mean_gate = gate_value.mean()
        rank = int(mean_gate.item() * self.max_rank)
        rank = max(1, min(rank, self.max_rank))
        a = self.A(x)[:, :rank]  # [B, r, H, W]
        b = self.B(x)[:, :rank]
        ab = torch.einsum('bihw,bjhw->bijhw', a, b)  # [B, r, r, H, W]
        ab = ab.reshape(B, rank * rank, H, W)
        # weight cropping in forward (output expects in_channels = rank*rank)
        weight = self.output.weight[:, :rank * rank, :, :]
        bias = self.output.bias
        out = F.conv2d(ab, weight, bias)
        return out

# ------------------------
# LightweightAdaptiveFilter (depthwise small-k convs)
# ------------------------
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
    def forward(self, x):
        return self.depthwise(x)

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
            nn.Conv2d(red, (max_k + 1)//2, 1, bias=False)  # predict weight for odd kernels 1,3,5...
        )
        self.depthwise_convs = nn.ModuleList([
            DepthwiseConv(c, kernel_size=k, padding=k//2) for k in range(1, max_k+1, 2)
        ])

    def forward(self, feat):
        B, c, H, W = feat.shape
        kernels = self.kernel_pred(feat).view(B, -1)  # [B, n_k]
        kernels = F.softmax(kernels, dim=1)
        outs = []
        for idx, conv in enumerate(self.depthwise_convs):
            out_k = conv(feat)
            outs.append(out_k * kernels[:, idx].view(B, 1, 1, 1))
        return sum(outs)

# ------------------------
# WTConv2d with lazy initialization (compatible with YOLO dummy forward)
# ------------------------
class WTConv2d(nn.Module):
    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        """
        If in_channels/out_channels are None -> lazy init on first forward.
        This avoids YOLO's dummy-forward channel mismatch.
        """
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
        assert in_channels == out_channels, "WTConv2d currently expects in_channels == out_channels"

        self.in_channels = in_channels
        self.out_channels = out_channels

        dec_filters, rec_filters = create_wavelet_filter(self.wt_type, in_channels, in_channels, dtype=torch.float32)
        # dec_filters: [in_channels, 4, K, K], rec_filters similar
        self.wt_filter = nn.Parameter(dec_filters, requires_grad=False)
        self.iwt_filter = nn.Parameter(rec_filters, requires_grad=False)

        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        # base conv as standard conv (not grouped) for robustness
        self.base_conv = nn.Conv2d(in_channels, in_channels, self.kernel_size,
                                   padding=self.kernel_size//2, stride=1, bias=self.bias, groups=1)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # wavelet-level convs (we keep grouped convs per-subband to be efficient)
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, self.kernel_size,
                       padding=self.kernel_size//2, stride=1, groups=in_channels*4, bias=False)
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels*4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            # simple average pooling for downsampling to avoid grouped convs mismatch
            self.do_stride = lambda x_in: F.avg_pool2d(x_in, kernel_size=self.stride, stride=self.stride)
        else:
            self.do_stride = None

        self.initialized = True

    def forward(self, x):
        # x shape: [B, C, H, W]
        if not self.initialized:
            # If YOLO does dummy forward with placeholder small channel,
            # still initialize to given x.shape[1] (works for both dummy and real)
            ch = x.shape[1]
            if ch <= 0:
                # fallback: avoid crash, return x
                return x
            self._init_layers(ch, ch)

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)  # [B, C, 4, H//2, W//2]
            curr_x_ll = curr_x[:, :, 0, :, :]  # LL

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  # [B, C*4, H//2, W//2]
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)  # [B, C, 4, H, W]
            next_x_ll = self.iwt_function(curr_x)  # inverse
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

# ------------------------
# Final IIFM_V2 module (integrated)
# ------------------------
class IIFM_V2(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        fusion="replace",
        reduction=8,
        max_rank=8,
        filter_k=5,
        wt_levels=1,
        wt_type="db1",
        use_in=True,
        use_mask=True,
        use_restore=True,
        use_wt=True,
        use_adaptive_fusion=True,
        use_inner_dtpa=False,
    ):        
        super().__init__()
        assert fusion in ("replace", "add", "concat")
        self.fusion = fusion
        self.c1, self.c2, self.use_mask,self.use_restore,self.use_wt, = c1, c2,use_mask,use_restore,use_wt

        # Basic convs (use normal convs to avoid channels mismatch during YOLO dummy forward)
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.in1 = nn.InstanceNorm2d(c2)#todo Instance
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(c2)#todo
        red = max(1, c2 // reduction)
        self.mask_branch = nn.Sequential(
            nn.Conv2d(c2, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, c2, 1, bias=False),
            nn.Sigmoid()
        )

        # restore (local convs)
        self.restore = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),#todo
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        )

        # attention + filtering
        #self.dynamic_tpa = DynamicTPA(c2, c2, max_rank)
        #self.adaptive_filter = LightweightAdaptiveFilter(c2, max_k=filter_k, reduction=reduction)

        # WTConv2d lazy (do not force in_channels here)
        self.restore_wt = WTConv2d(in_channels=None, out_channels=None)

        # fusion outputs
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

    def forward(self, x):
        # base features
        out = self.act(self.in1(self.conv1(x)))
        out = self.act(self.in2(self.conv2(out)))

        # luminance mask
        #gray = out.mean(dim=1, keepdim=True)
        #mask = self.mask_branch(out)
        #mask = torch.ones_like(out)*0.5
        # restore via local convs then wavelet processing
        #restored_local = self.restore(out)
        #wt_out = self.restore_wt(restored_local)  # WTConv2d handles lazy init

        # attention + adaptive filter on WT output
        #att_out = self.dynamic_tpa(wt_out)
        #filt_out = self.adaptive_filter(wt_out)

        #fused = att_out
        #fused = wt_out
        mask = self.mask_branch(out) if self.use_mask else torch.full_like(out, 0.5)
        restored_local = self.restore(out) if self.use_restore else out
        wt_out = self.restore_wt(restored_local) if self.use_wt else restored_local
        fused = wt_out
        out = (1 - mask) * out + mask * fused

        if self.fusion == "replace":
            return out
        elif self.fusion == "add":
            xp = self.project(x) if self.project is not None else x
            return xp + out
        else:
            return self.fuse_conv(torch.cat([x, out], dim=1))


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# 1?? Lightweight Illumination-Aware Enhancement
# ------------------------
class LowLightEnhanceLW_V2(nn.Module):
    #"""
    #Lightweight Illumination-Aware Enhancement
    #- 弱光增强 + 强光抑制
    #"""
    def __init__(self, c, reduction=8, use_groupnorm=False):
        super().__init__()
        mid = max(1, c // reduction)

        # 局部特征增强
        self.depthwise = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        if use_groupnorm:
            self.norm = nn.GroupNorm(4, c)
        else:
            self.norm = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

        # 通道注意权重
        self.enhance = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False),
            nn.Sigmoid()
        )

        # 参数控制
        self.scale = 0.8      # 增强幅度
        self.balance = 1.0    # 亮暗敏感度

    def forward(self, x):
        # 局部特征
        base = self.act(self.norm(self.depthwise(x)))
        weight = self.enhance(base)

        # 全局亮度估计
        illum = x.mean(dim=(2, 3), keepdim=True)

        # 亮度响应函数：暗区正、亮区负
        illum_weight = torch.tanh(self.balance * (1.0 - 2.0 * illum))

        # 自适应增强/抑制
        out = x * (1 + self.scale * illum_weight * weight)
        return out


# ------------------------
# 2?? IIFM_V4 with Illumination Adaptive Enhancement
# ------------------------
class IIFM_V4(nn.Module):
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

        # 强光遮罩分支
        red = max(1, c2 // reduction)
        self.mask_branch = nn.Sequential(
            nn.Conv2d(c2, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, c2, 1, bias=False),
            nn.Sigmoid()
        )

        # 局部修复分支
        self.restore = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        )

        # 动态注意与自适应滤波
        self.dynamic_tpa = DynamicTPA(c2, c2, max_rank)
        self.adaptive_filter = LightweightAdaptiveFilter(c2, max_k=filter_k, reduction=reduction)

        # 小波卷积（延迟初始化）
        self.restore_wt = WTConv2d(in_channels=None, out_channels=None)

        # ? 光照自适应增强
        self.low_light_enhance = LowLightEnhanceLW_V2(c2, reduction=reduction, use_groupnorm=True)

        # 融合输出层
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

    def forward(self, x):
        # ===== 基础特征提取 =====
        out = self.act(self.in1(self.conv1(x)))
        out = self.act(self.in2(self.conv2(out)))

        # ===== 强光区域检测 =====
        mask = self.mask_branch(out)

        # ===== 局部修复 + 小波滤波 =====
        restored_local = self.restore(out)
        wt_out = self.restore_wt(restored_local)

        # ===== 动态注意 + 自适应滤波 =====
        att_out = self.dynamic_tpa(wt_out)
        filt_out = self.adaptive_filter(wt_out)
        fused = att_out + filt_out

        # ===== 光照自适应增强 =====
        illum = x.mean(dim=(2,3), keepdim=True)
        illum_weight = torch.clamp(1.2 - 2 * illum, 0, 1)

        # 强光区域修复 + 弱光区域增强
        out = (1 - mask) * out + mask * fused
        out = out + illum_weight * self.low_light_enhance(out)

        # ===== 输出融合模式 =====
        if self.fusion == "replace":
            return out
        elif self.fusion == "add":
            xp = self.project(x) if self.project is not None else x
            return xp + out
        else:
            return self.fuse_conv(torch.cat([x, out], dim=1))
# ------------------------
# Quick test function (run locally)
# ------------------------
if __name__ == "__main__":
    # Quick sanity check: create IIFM and do a dummy forward (same shape used in YOLO builder)
    m = IIFM_V2(c1=128, c2=128, fusion="replace", wt_levels=1)
    x = torch.zeros(2, 128, 80, 80)  # same shape as YOLO dummy forward in builder
    y = m(x)
    print("IIFM_V2 forward ok, out shape:", y.shape)
