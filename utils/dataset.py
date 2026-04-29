"""
BTCDataset: Sliding-window dataset for time series Transformer training.
Loads pre-computed feature parquets as numpy memmaps for efficient multi-worker DataLoader access.

Supports two task types:
  - "regression":     returns float target (raw log return)
  - "classification": bins target into {0=Down, 1=Neutral, 2=Up} using per-sample spread as boundary
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLS = [
    "F01_log_return", "F02_spread_bps", "F03_micro_dev_bps", "F04_rvol_20", "F05_spread_z",
    "F06_d_ask_1", "F07_d_ask_2", "F08_d_ask_3", "F09_d_ask_4", "F10_d_ask_5",
    "F11_d_bid_1", "F12_d_bid_2", "F13_d_bid_3", "F14_d_bid_4", "F15_d_bid_5",
    "F16_v_ask_1", "F17_v_ask_2", "F18_v_ask_3", "F19_v_ask_4", "F20_v_ask_5",
    "F21_v_ask_6", "F22_v_ask_7", "F23_v_ask_8", "F24_v_ask_9", "F25_v_ask_10",
    "F26_v_bid_1", "F27_v_bid_2", "F28_v_bid_3", "F29_v_bid_4", "F30_v_bid_5",
    "F31_v_bid_6", "F32_v_bid_7", "F33_v_bid_8", "F34_v_bid_9", "F35_v_bid_10",
    "F36_bid_slope", "F37_ask_slope",
    "F38_obi_l1", "F39_obi_near", "F40_obi_deep", "F41_wobi", "F42_dobi",
    "F43_ofi_bid", "F44_ofi_ask", "F45_cofi_10",
    "F46_tfi", "F47_log_vol", "F48_log_ticks", "F49_vwap_dev_bps", "F50_micro_vol_bps", "F51_aggr_ratio",
    "F52_obi_z", "F53_mom16", "F54_mom32",
    "F55_obi_tfi_div", "F56_book_asym_slope",
]

TARGET_COLS = ["y_1", "y_5", "y_10"]

# F02_spread_bps is column index 1 in the feature array
SPREAD_COL = 1

# Class labels for classification mode
CLASS_DOWN    = 0
CLASS_NEUTRAL = 1
CLASS_UP      = 2
N_CLASSES     = 3


class BTCDataset(Dataset):
    """Sliding-window dataset over pre-computed features stored as numpy memmap."""

    def __init__(self, data_path, seq_len=512, stride=1, target_idx=0,
                 task_type="regression"):
        """
        Args:
            data_path:  path to parquet or directory containing *_features.npy / *_targets.npy
            seq_len:    lookback window length in ticks
            stride:     step between consecutive windows (1 = every tick)
            target_idx: which target column to use (0=y_1, 1=y_5, 2=y_10)
            task_type:  "regression" returns float log-return target;
                        "classification" bins target into {0=Down, 1=Neutral, 2=Up}
                        using the per-sample F02_spread_bps value as the boundary.
        """
        self.seq_len    = seq_len
        self.stride     = stride
        self.target_idx = target_idx
        self.task_type  = task_type

        npy_features = data_path.replace(".parquet", "_features.npy")
        npy_targets  = data_path.replace(".parquet", "_targets.npy")

        if os.path.exists(npy_features) and os.path.exists(npy_targets):
            self.features = np.load(npy_features, mmap_mode="r")
            self.targets  = np.load(npy_targets,  mmap_mode="r")
        else:
            df = pd.read_parquet(data_path)
            feat_vals = df[FEATURE_COLS].values.astype(np.float32)
            tgt_vals  = df[TARGET_COLS].values.astype(np.float32)
            np.save(npy_features, feat_vals)
            np.save(npy_targets,  tgt_vals)
            self.features = np.load(npy_features, mmap_mode="r")
            self.targets  = np.load(npy_targets,  mmap_mode="r")

        self.n_samples  = max(0, (len(self.features) - self.seq_len) // self.stride)
        self.n_features = self.features.shape[1]

        # Pre-compute class weights for classification (used by CrossEntropyLoss)
        self._class_weights = None
        if task_type == "classification":
            self._class_weights = self._compute_class_weights()

    def _compute_class_weights(self, n_sample=500_000):
        """
        Compute balanced class weights: w_c = N / (n_classes * count_c).

        Reads the first n_sample rows as a single contiguous slice — fast on
        network-mounted memmaps (one sequential read vs millions of random seeks).
        500K rows is more than enough for a stable class-frequency estimate.

        Returns a float32 tensor of shape (N_CLASSES,).
        """
        n      = min(n_sample, len(self.features))
        raw_y  = np.array(self.targets[:n, self.target_idx], dtype=np.float64)
        spread = np.array(self.features[:n, SPREAD_COL],    dtype=np.float64) / 10000.0

        labels = np.where(raw_y < -spread, CLASS_DOWN,
                 np.where(raw_y >  spread, CLASS_UP, CLASS_NEUTRAL)).astype(np.int64)

        counts  = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
        n_total = counts.sum()
        weights = n_total / (N_CLASSES * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def class_weights(self):
        """Float32 tensor of shape (N_CLASSES,) for use in CrossEntropyLoss."""
        return self._class_weights

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end   = start + self.seq_len
        x     = torch.from_numpy(self.features[start:end].copy())

        if self.task_type == "classification":
            raw_y       = float(self.targets[end - 1, self.target_idx])
            spread_frac = float(self.features[end - 1, SPREAD_COL]) / 10000.0
            if raw_y < -spread_frac:
                label = CLASS_DOWN
            elif raw_y > spread_frac:
                label = CLASS_UP
            else:
                label = CLASS_NEUTRAL
            y = torch.tensor(label, dtype=torch.long)
        else:
            y = torch.tensor(self.targets[end - 1, self.target_idx], dtype=torch.float32)

        return x, y


def create_dataloaders(data_dir, seq_len=512, batch_size=256, stride=1,
                       target_idx=0, num_workers=4, task_type="regression"):
    """Create train/val/test DataLoaders from processed parquets."""
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.parquet")
        if not os.path.exists(path):
            continue
        ds = BTCDataset(path, seq_len=seq_len, stride=stride,
                        target_idx=target_idx, task_type=task_type)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
        )
    return loaders
