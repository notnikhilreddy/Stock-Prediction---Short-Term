"""
Non-Stationary Transformer for BTC HFT prediction.

Reference: Liu et al., "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"
           (NeurIPS 2022)

Key design:
- De-stationary Attention: computes per-sequence (mu, sigma) per channel, uses them
  as conditioning to re-scale attention output.
- Series Stationarization: normalizes input by subtracting mean and dividing by std,
  then reverses at output.
- Multi-variate: all channels cross-attend — no channel independence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    """Simple series decomposition: moving average as trend, remainder as seasonal."""

    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)

    def forward(self, x):
        """x: (B, L, C) -> trend (B, L, C), seasonal (B, L, C)"""
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DeStationaryAttention(nn.Module):
    """
    Multi-head attention with de-stationary conditioning.
    After standard attention, re-introduces non-stationarity via learned
    projections of the per-sequence (tau, delta) statistics.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # De-stationary projections: map (tau, delta) to re-scaling factors
        self.tau_proj = nn.Linear(1, n_heads, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, tau, delta):
        """
        Q, K, V: (B, L, d_model)
        tau: (B, 1) — per-sequence std
        delta: (B, d_model) — per-sequence mean projected
        """
        B, L, D = Q.shape
        H = self.n_heads

        q = self.W_Q(Q).view(B, L, H, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        k = self.W_K(K).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(B, L, H, self.d_k).transpose(1, 2)

        # Standard scaled dot-product attention
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, L, L)

        # De-stationary re-scaling via tau
        tau_weight = self.tau_proj(tau).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        attn = attn * tau_weight

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # De-stationary shift via delta
        delta_out = self.delta_proj(delta).unsqueeze(1)  # (B, 1, D)
        out = out + delta_out

        return self.out_proj(out)


class NSTransformerBlock(nn.Module):
    """Single NS-Transformer encoder block."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, tau, delta):
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, tau, delta)
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class NSTransformer(nn.Module):
    """
    Non-Stationary Transformer for BTC HFT prediction.

    Input:  (batch, seq_len, n_features)
    Output: (batch,)           when n_classes=1  — scalar regression
            (batch, n_classes)  when n_classes>1  — class logits

    Note: the de-stationarization re-scaling (out * sigma + mu) is only applied
    in regression mode. Scaling raw logits by input statistics would corrupt
    the classification loss.
    """

    def __init__(
        self,
        n_features=56,
        seq_len=512,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=None,
        dropout=0.1,
        head_dropout=0.1,
        n_classes=1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len    = seq_len
        self.d_model    = d_model
        self.n_classes  = n_classes
        self.eps        = 1e-5

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Mean/std projection for de-stationary conditioning
        self.mean_proj = nn.Linear(n_features, d_model)
        self.std_proj  = nn.Linear(n_features, 1)

        self.blocks = nn.ModuleList([
            NSTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        out_dim = n_classes if n_classes > 1 else 1
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: (batch,) for regression, (batch, n_classes) for classification
        """
        B, L, C = x.shape

        # Series stationarization
        mu    = x.mean(dim=1, keepdim=True)                                       # (B, 1, C)
        sigma = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()    # (B, 1, C)
        x_norm = (x - mu) / sigma

        # De-stationary conditioning statistics
        mu_flat    = mu.squeeze(1)       # (B, C)
        sigma_flat = sigma.squeeze(1)    # (B, C)
        tau   = self.std_proj(sigma_flat)    # (B, 1)
        delta = self.mean_proj(mu_flat)      # (B, d_model)

        # Project to d_model
        h = self.input_proj(x_norm) + self.pos_embed[:, :L, :]

        # Transformer blocks with de-stationary attention
        for block in self.blocks:
            h = block(h, tau, delta)

        h = self.norm(h)

        # Pool over sequence: last token + mean
        h_pool = h[:, -1, :] + h.mean(dim=1)   # (B, d_model)

        out = self.head(h_pool)   # (B, out_dim)

        if self.n_classes == 1:
            out = out.squeeze(-1)                      # (B,)
            # Re-scale by input return statistics (regression only)
            out = out * sigma[:, 0, 0] + mu[:, 0, 0]

        return out                                     # (B,) or (B, n_classes)
