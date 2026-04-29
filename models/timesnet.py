"""
TimesNet for BTC HFT regression.

Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
           Time Series Analysis" (ICLR 2023)

Key design:
- Temporal 2D Variation: uses FFT to identify dominant periods, then reshapes
  the 1D sequence into a 2D space (period × n_rows) and applies 2D Inception
  convolutions to capture both intra-period and inter-period patterns.
- Series Stationarization: normalizes input per sequence/channel, then
  re-introduces the statistics at the output (regression only).
- Multivariate: all 56 features are projected to d_model jointly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    2D Inception block: parallel convolutions at multiple kernel sizes.
    Uses GroupNorm (stable with variable batch sizes during HFT eval).
    """

    def __init__(self, d_model: int, num_kernels: int = 3):
        super().__init__()
        # Kernel sizes: 1×1, 3×3, 5×5  (one per branch up to num_kernels)
        kernel_sizes = [2 * i + 1 for i in range(num_kernels)]
        groups = min(8, d_model)   # GroupNorm groups; d_model must be divisible by this

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    d_model, d_model,
                    kernel_size=ks,
                    padding=ks // 2,   # 'same' padding
                    groups=groups,     # depth-wise-style per-channel
                    bias=False,
                ),
                nn.GroupNorm(groups, d_model),
                nn.GELU(),
            )
            for ks in kernel_sizes
        ])

        # 1×1 conv to fuse branches back to d_model
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model * num_kernels, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(groups, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, H, W)  →  (B, D, H, W)"""
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fuse(out)


class TimesBlock(nn.Module):
    """
    Core TimesNet block.

    Algorithm:
      1. FFT over the time axis → identify top-k dominant periods.
      2. For each period p: pad to multiple of p, reshape to 2D (p × T/p),
         apply InceptionBlock, reshape back, truncate to T.
      3. Weighted average (FFT amplitude as weights) over the k branches.
      4. Add residual, LayerNorm, then a position-wise FFN.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        top_k: int = 5,
        num_kernels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.top_k     = top_k

        self.inception = InceptionBlock(d_model, num_kernels)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _find_top_k_periods(self, x: torch.Tensor):
        """
        Compute FFT over time dim, average amplitudes across batch and channels,
        and return top-k period lengths and their softmax-normalised weights.

        x: (B, T, D)
        Returns:
            periods     : list of int, length k
            amp_weights : (k,) float tensor on same device as x
        """
        B, T, D = x.shape
        xf  = torch.fft.rfft(x, dim=1)            # (B, T//2+1, D)
        amp = xf.abs().mean(0).mean(-1)            # (T//2+1,)  avg over batch & channels
        amp[0] = 0.0                               # zero out DC

        k   = min(self.top_k, amp.shape[0] - 1)
        topk_amp, topk_idx = torch.topk(amp, k)   # (k,)

        # frequency-index → period (round, clamp to [2, T//2])
        periods = []
        for idx in topk_idx:
            f = max(1, idx.item())
            p = max(2, round(T / f))
            p = min(p, T // 2)
            periods.append(int(p))

        amp_weights = F.softmax(topk_amp.detach(), dim=0)  # (k,)
        return periods, amp_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  (B, T, D)"""
        B, T, D = x.shape
        periods, amp_weights = self._find_top_k_periods(x)

        branches = []
        for period in periods:
            T_pad = math.ceil(T / period) * period   # smallest multiple of period >= T

            # Pad along time axis
            if T_pad > T:
                x_pad = F.pad(
                    x.permute(0, 2, 1), (0, T_pad - T)  # pad last dim of (B, D, T)
                ).permute(0, 2, 1)                         # (B, T_pad, D)
            else:
                x_pad = x

            n_cols = T_pad // period   # number of periods (width dimension)

            # Reshape to 2D: (B, T_pad, D) → (B, D, period, n_cols)
            # period = height (intra-period), n_cols = width (inter-period)
            x_2d = (
                x_pad
                .reshape(B, n_cols, period, D)  # (B, n_cols, period, D)
                .permute(0, 3, 2, 1)            # (B, D, period, n_cols)
            )

            # 2D Inception convolutions
            x_2d = self.inception(x_2d)         # (B, D, period, n_cols)

            # Reshape back to 1D sequence
            x_out = (
                x_2d
                .permute(0, 3, 2, 1)            # (B, n_cols, period, D)
                .reshape(B, T_pad, D)            # (B, T_pad, D)
            )[:, :T, :]                          # (B, T, D)  truncate padding

            branches.append(x_out)

        # Weighted sum over branches
        stacked = torch.stack(branches, dim=-1)   # (B, T, D, k)
        w       = amp_weights.view(1, 1, 1, -1)  # (1, 1, 1, k)
        x_agg   = (stacked * w).sum(-1)           # (B, T, D)

        # Residual connections + layer norm
        x = self.norm1(x + x_agg)
        x = self.norm2(x + self.ff(x))
        return x


class TimesNet(nn.Module):
    """
    TimesNet for BTC HFT regression.

    Input:  (batch, seq_len, n_features)
    Output: (batch,)           — scalar regression (n_classes=1)
            (batch, n_classes)  — class logits      (n_classes>1, not rescaled)

    The de-stationarization rescaling (out * sigma_F01 + mu_F01) is applied
    only in regression mode, matching the NS-Transformer convention.
    """

    def __init__(
        self,
        n_features:  int = 56,
        seq_len:     int = 512,
        d_model:     int = 128,
        n_layers:    int = 3,
        d_ff:        int = None,
        top_k:       int = 5,
        num_kernels: int = 3,
        dropout:     float = 0.1,
        head_dropout: float = 0.1,
        n_classes:   int = 1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.seq_len     = seq_len
        self.d_model     = d_model
        self.n_classes   = n_classes
        self.eps         = 1e-5

        _d_ff = d_ff or d_model * 4

        # Input projection + positional embedding
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Stack of TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, _d_ff, top_k, num_kernels, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Regression or classification head
        out_dim = n_classes if n_classes > 1 else 1
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch,) for regression, (batch, n_classes) for classification
        """
        B, L, C = x.shape

        # Series stationarization (per-sequence, per-channel)
        mu    = x.mean(dim=1, keepdim=True)                                    # (B, 1, C)
        sigma = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt() # (B, 1, C)
        x_norm = (x - mu) / sigma                                              # (B, L, C)

        # Project to model dimension + add positional embedding
        h = self.input_proj(x_norm) + self.pos_embed[:, :L, :]  # (B, L, D)

        # Stack of 2D-variation blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Pool: last token + sequence mean (same as NS-Transformer)
        h_pool = h[:, -1, :] + h.mean(dim=1)   # (B, D)

        out = self.head(h_pool)                 # (B, out_dim)

        if self.n_classes == 1:
            out = out.squeeze(-1)               # (B,)
            # De-stationarize using F01 (first feature channel) statistics
            out = out * sigma[:, 0, 0] + mu[:, 0, 0]

        return out                              # (B,) or (B, n_classes)
