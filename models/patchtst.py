"""
PatchTST with Channel Independence and RevIN normalization.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

Key design:
- Channel Independence: each of the 56 features is processed as a separate univariate
  time series through a shared Transformer backbone.
- RevIN: per-channel reversible instance normalization over the lookback window.
- Patching: the lookback window is divided into non-overlapping patches that become
  the token sequence for self-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022)."""

    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, mode):
        """
        x: (batch, seq_len, n_features)
        mode: 'norm' or 'denorm'
        """
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std + self.mean
        return x


class PatchEmbedding(nn.Module):
    """Convert a 1D time series into a sequence of patch tokens."""

    def __init__(self, patch_len, d_model, stride=None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x: (batch * n_features, seq_len)
        returns: (batch * n_features, n_patches, d_model)
        """
        # Unfold into patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B*C, n_patches, patch_len)
        return self.proj(x)


class PatchTSTEncoder(nn.Module):
    """Standard Transformer encoder for patch tokens."""

    def __init__(self, d_model, n_heads, n_layers, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)


class PatchTST(nn.Module):
    """
    PatchTST with Channel Independence for BTC HFT prediction.

    Input:  (batch, seq_len, n_features)
    Output: (batch,)          when n_classes=1  — scalar regression
            (batch, n_classes) when n_classes>1  — class logits
    """

    def __init__(
        self,
        n_features=56,
        seq_len=512,
        patch_len=16,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=None,
        dropout=0.1,
        head_dropout=0.1,
        n_classes=1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len    = seq_len
        self.patch_len  = patch_len
        self.n_classes  = n_classes

        n_patches = seq_len // patch_len

        self.revin       = RevIN(n_features)
        self.patch_embed = PatchEmbedding(patch_len, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.encoder     = PatchTSTEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.head_norm   = nn.LayerNorm(d_model)
        self.flatten     = nn.Flatten(start_dim=-2)
        out_dim = n_classes if n_classes > 1 else 1
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: (batch,) for regression, (batch, n_classes) for classification
        """
        B, L, C = x.shape

        # RevIN normalize
        x = self.revin(x, "norm")

        # Channel independence: reshape to (B*C, L)
        x = x.permute(0, 2, 1).reshape(B * C, L)

        # Patch embedding
        x = self.patch_embed(x)  # (B*C, n_patches, d_model)
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)  # (B*C, n_patches, d_model)

        # Reshape back: (B, C, n_patches, d_model)
        x = x.reshape(B, C, -1, x.shape[-1])

        # Average across channels (CI aggregation)
        x = x.mean(dim=1)  # (B, n_patches, d_model)

        x = self.head_norm(x)
        x = self.flatten(x)         # (B, n_patches * d_model)
        x = self.head(x)            # (B, out_dim)

        if self.n_classes == 1:
            return x.squeeze(-1)    # (B,)
        return x                    # (B, n_classes)
