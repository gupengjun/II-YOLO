class DynamicTPA(nn.Module):
    def __init__(self, c1, c2, max_rank):
        super(DynamicTPA, self).__init__()
        self.max_rank = max_rank
        self.A = Conv(c1, max_rank, 1, act=False)
        self.B = Conv(c1, max_rank, 1, act=False)
        self.output = nn.Conv2d(max_rank * max_rank, c2, kernel_size=1, bias=True)
        self.gate = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        B, C, H, W = x.shape
        gate_value = self.gate(x).squeeze()  
        if gate_value.numel() == 0 or torch.isnan(gate_value).any() or torch.isinf(gate_value).any():
            print("Warning: gate_value contains NaN or Inf, fallback to 0.5")
            mean_gate = torch.tensor(0.5, device=x.device)
        else:
            gate_value = torch.clamp(gate_value, -10, 10)
            gate_value = torch.sigmoid(gate_value)
            mean_gate = gate_value.mean()
        rank = int(mean_gate.item() * self.max_rank)
        rank = max(1, min(rank, self.max_rank))  # ÏŞ¶¨ÔÚ [1, max_rank]
        a = self.A(x)[:, :rank]  # [B, r, H, W]
        b = self.B(x)[:, :rank]  # [B, r, H, W]
        ab = torch.einsum('bihw,bjhw->bijhw', a, b)  # [B, r, r, H, W]
        ab = ab.reshape(B, rank * rank, H, W)
        weight = self.output.weight[:, :rank * rank]  # [c2, r*r, 1, 1]
        bias = self.output.bias
        out = F.conv2d(ab, weight, bias)

        return out
