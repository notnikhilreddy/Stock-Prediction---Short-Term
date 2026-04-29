#!/usr/bin/env python3
"""
Step 2 — Model Training
========================
Trains one of three architectures (PatchTST, NS-Transformer, TimesNet) on the
pre-computed 56-feature BTC-USDT dataset.

Supports two task types:
    regression      — MSELoss, primary metric = val_MAE  (lower is better)
    classification  — CrossEntropyLoss, primary metric = val_DA  (higher is better)

Features:
  - Multi-GPU training via DistributedDataParallel (torchrun)
  - OneCycle LR schedule with configurable warmup
  - Mixed-precision (AMP) on CUDA
  - Automatic resume from last checkpoint on preemption
  - Global best tracking: best checkpoint per (architecture, task type) is copied
    to results/_global_best/ for use by backtest.py

Usage (single GPU / CPU):
    python 02_train.py --config configs/patchtst_small.yaml

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 02_train.py --config configs/nstransformer_small.yaml

Config reference: see configs/*.yaml (all fields and their defaults are in DEFAULT_CONFIG below).
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

from models.patchtst      import PatchTST
from models.ns_transformer import NSTransformer
from models.timesnet      import TimesNet
from utils.dataset        import BTCDataset, FEATURE_COLS, TARGET_COLS
from utils.metrics        import evaluate_all, evaluate_all_cls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "model":            "patchtst",    # "patchtst" | "nstransformer" | "timesnet"
    "task_type":        "regression",  # "regression" | "classification"
    "n_classes":        1,             # 1 = regression output; 3 = 3-class classification
    "n_features":       56,            # number of input features (fixed)
    "seq_len":          512,           # lookback window length in ticks (100 ms each)
    "patch_len":        16,            # PatchTST: patch length (seq_len / patch_len = n_patches)
    "d_model":          256,           # Transformer hidden dimension
    "n_heads":          8,             # number of attention heads
    "n_layers":         6,             # number of Transformer blocks
    "d_ff":             None,          # FFN dim (None = 4 × d_model)
    "dropout":          0.1,
    "head_dropout":     0.1,
    # ── TimesNet only ────────────────────────────────────────────────────────
    "top_k":            5,             # dominant FFT periods to keep
    "num_kernels":      3,             # InceptionBlock kernel sizes (1, 3, 5)
    # ── Training ─────────────────────────────────────────────────────────────
    "target_idx":       0,             # 0=y_1 (100ms), 1=y_5 (500ms), 2=y_10 (1s)
    "batch_size":       256,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "warmup_epochs":    3,
    "max_epochs":       200,
    "patience":         15,            # early stopping patience
    "grad_clip":        1.0,
    "stride":           1,             # dataset window stride (1 = every tick)
    "num_workers":      4,
    "use_amp":          True,
    "wall_time_hours":  23.5,          # hard wall-time limit (triggers graceful stop)
    # ── Paths ────────────────────────────────────────────────────────────────
    "data_dir":         "../data-processed",
    "results_dir":      "results",
    "experiment_name":  "default",
}


def load_config(path):
    """Load a YAML config and coerce numeric strings to float."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ["lr", "weight_decay", "dropout", "head_dropout", "grad_clip", "wall_time_hours"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = float(cfg[key])
    return cfg


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg):
    """Instantiate the model specified in cfg["model"]."""
    model_type = cfg["model"].lower()
    n_classes  = cfg.get("n_classes", 1)

    if model_type == "patchtst":
        return PatchTST(
            n_features=cfg["n_features"],  seq_len=cfg["seq_len"],
            patch_len=cfg.get("patch_len", 16),
            d_model=cfg["d_model"],        n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],      d_ff=cfg.get("d_ff"),
            dropout=cfg["dropout"],        head_dropout=cfg.get("head_dropout", 0.1),
            n_classes=n_classes,
        )
    elif model_type in ("nstransformer", "ns_transformer"):
        return NSTransformer(
            n_features=cfg["n_features"],  seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],        n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],      d_ff=cfg.get("d_ff"),
            dropout=cfg["dropout"],        head_dropout=cfg.get("head_dropout", 0.1),
            n_classes=n_classes,
        )
    elif model_type == "timesnet":
        return TimesNet(
            n_features=cfg["n_features"],  seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],        n_layers=cfg["n_layers"],
            d_ff=cfg.get("d_ff"),          top_k=cfg.get("top_k", 5),
            num_kernels=cfg.get("num_kernels", 3),
            dropout=cfg["dropout"],        head_dropout=cfg.get("head_dropout", 0.1),
            n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")


# ---------------------------------------------------------------------------
# Global best tracking
# ---------------------------------------------------------------------------

def _global_best_key(model_type, task_type):
    """Return the file stem for the global-best tracker of a given arch + task."""
    suffix = "cls" if task_type == "classification" else "reg"
    return f"{model_type}_{suffix}_best"


def update_global_best(model_type, primary_value, experiment_name,
                       checkpoint_path, results_dir, task_type="regression"):
    """
    Copy checkpoint to results/_global_best/ if it beats the current global best
    for this (architecture, task_type) combination.

    Regression:     primary_value = val_MAE  (lower is better)
    Classification: primary_value = val_DA   (higher is better)
    """
    gdir = os.path.join(results_dir, "_global_best")
    os.makedirs(gdir, exist_ok=True)

    key          = _global_best_key(model_type, task_type)
    tracker_path = os.path.join(gdir, f"{key}.json")
    ckpt_dst     = os.path.join(gdir, f"{key}.pt")

    higher_is_better = (task_type == "classification")
    current_best     = float("-inf") if higher_is_better else float("inf")

    if os.path.exists(tracker_path):
        try:
            with open(tracker_path) as f:
                current_best = json.load(f).get("primary_value", current_best)
        except (json.JSONDecodeError, KeyError):
            pass

    is_new_best = (primary_value > current_best) if higher_is_better \
                  else (primary_value < current_best)

    if is_new_best:
        import shutil
        shutil.copy2(checkpoint_path, ckpt_dst)
        metric_name = "val_da" if higher_is_better else "val_mae"
        with open(tracker_path, "w") as f:
            json.dump({
                "primary_value": float(primary_value),
                "metric":        metric_name,
                "task_type":     task_type,
                "experiment":    experiment_name,
                "checkpoint":    ckpt_dst,
                "updated_at":    datetime.now().isoformat(),
            }, f, indent=2)
        log.info(f"  ** NEW GLOBAL BEST {model_type} [{task_type}]: "
                 f"{metric_name}={primary_value:.6f} ({experiment_name})")
        return True
    return False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Self-contained training loop with checkpointing, early stopping,
    mixed-precision, and optional DDP multi-GPU support.
    """

    def __init__(self, cfg, local_rank=0, world_size=1):
        self.cfg          = cfg
        self.local_rank   = local_rank
        self.world_size   = world_size
        self.is_main      = (local_rank == 0)
        self.distributed  = (world_size > 1)
        self.start_time   = time.time()
        self.wall_limit   = cfg.get("wall_time_hours", 23.5) * 3600
        self.task_type    = cfg.get("task_type", "regression")

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Output directory
        self.exp_dir = os.path.join(cfg["results_dir"], cfg["experiment_name"])
        if self.is_main:
            os.makedirs(self.exp_dir, exist_ok=True)

        self._setup_data()

        # Model + optional DDP wrapping
        self.model = build_model(cfg).to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[local_rank])

        # Optimizer + OneCycle LR
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        warmup_steps = cfg["warmup_epochs"] * len(self.train_loader)
        total_steps  = cfg["max_epochs"]    * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg["lr"],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
            anneal_strategy="cos",
        )

        self.use_amp = cfg.get("use_amp", True) and self.device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Loss function
        if self.task_type == "classification":
            class_weights = self.train_ds.class_weights
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.MSELoss()

        # Primary metric tracking (what "best" means per task)
        self._higher_is_better = (self.task_type == "classification")
        self.best_primary       = float("-inf") if self._higher_is_better else float("inf")
        self.patience_counter   = 0
        self.start_epoch        = 0
        self.epoch              = 0
        self.train_log          = []

        self._try_resume()

        if self.is_main:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            log.info(f"Model:     {cfg['model']} | Params: {n_params:,}")
            log.info(f"Task:      {self.task_type} | n_classes={cfg.get('n_classes', 1)}")
            log.info(f"Device:    {self.device} | World size: {world_size}")
            log.info(f"Train:     {len(self.train_ds):,} samples | Val: {len(self.val_ds):,}")
            if self.start_epoch > 0:
                log.info(f"Resumed from epoch {self.start_epoch}")

    def _is_better(self, new_val, old_val):
        return new_val > old_val if self._higher_is_better else new_val < old_val

    def _setup_data(self):
        """Build BTCDataset objects and DataLoaders for train/val/test."""
        cfg       = self.cfg
        task_type = cfg.get("task_type", "regression")
        data_dir  = cfg["data_dir"]

        self.train_ds = BTCDataset(
            os.path.join(data_dir, "train.parquet"),
            seq_len=cfg["seq_len"], stride=cfg["stride"],
            target_idx=cfg["target_idx"], task_type=task_type,
        )
        self.val_ds = BTCDataset(
            os.path.join(data_dir, "val.parquet"),
            seq_len=cfg["seq_len"], stride=cfg["stride"],
            target_idx=cfg["target_idx"], task_type=task_type,
        )
        self.test_ds = BTCDataset(
            os.path.join(data_dir, "test.parquet"),
            seq_len=cfg["seq_len"], stride=cfg["stride"],
            target_idx=cfg["target_idx"], task_type=task_type,
        )

        if self.distributed:
            train_sampler = DistributedSampler(self.train_ds, shuffle=True)
            val_sampler   = DistributedSampler(self.val_ds,   shuffle=False)
            test_sampler  = DistributedSampler(self.test_ds,  shuffle=False)
        else:
            train_sampler = val_sampler = test_sampler = None

        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg["batch_size"],
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg["batch_size"],
            sampler=val_sampler, shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=cfg["batch_size"],
            sampler=test_sampler, shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )
        self.train_sampler = train_sampler

    def _try_resume(self):
        """Load checkpoint_last.pt if it exists (graceful preemption recovery)."""
        last_path = os.path.join(self.exp_dir, "checkpoint_last.pt")
        if not os.path.exists(last_path):
            return
        try:
            ckpt = torch.load(last_path, map_location=self.device)
            model_to_load = self.model.module if self.distributed else self.model
            model_to_load.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.best_primary     = ckpt.get("best_primary", self.best_primary)
            self.start_epoch      = ckpt.get("epoch", 0) + 1
            self.train_log        = ckpt.get("train_log", [])
            self.patience_counter = ckpt.get("patience_counter", 0)
            if self.is_main:
                log.info(f"Resumed from checkpoint at epoch {self.start_epoch}")
        except Exception as e:
            if self.is_main:
                log.warning(f"Failed to resume: {e}")

    def _time_remaining(self):
        return self.wall_limit - (time.time() - self.start_time)

    def train_epoch(self):
        """Run one full training epoch. Returns mean loss, or None if wall time hit."""
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

        total_loss    = 0.0
        n_batches     = 0
        total_batches = len(self.train_loader)
        epoch_start   = time.time()
        log_interval  = max(1, total_batches // 10)

        for x, y in self.train_loader:
            if self._time_remaining() < 300:
                log.info("Wall time limit approaching — stopping training loop.")
                return None

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                self.scaler.scale(loss).backward()
                if self.cfg["grad_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                if self.cfg["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            n_batches  += 1

            if self.is_main and n_batches % log_interval == 0:
                elapsed = time.time() - epoch_start
                rate    = n_batches / elapsed
                eta     = (total_batches - n_batches) / rate if rate > 0 else 0
                log.info(
                    f"  [{n_batches}/{total_batches} ({100*n_batches/total_batches:.0f}%)] "
                    f"loss={total_loss/n_batches:.6f} | {rate:.1f} b/s | ETA {eta/60:.0f}m"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        """Run evaluation on a DataLoader and return a metrics dict."""
        self.model.eval()
        all_preds, all_targets, all_spreads = [], [], []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.use_amp:
                with autocast(device_type="cuda"):
                    pred = self.model(x)
            else:
                pred = self.model(x)
            all_preds.append(pred.float().cpu())
            all_targets.append(y.cpu())
            # F02_spread_bps is column index 1; take last timestep for spread cost
            all_spreads.append(x[:, -1, 1].float().cpu())

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        if self.task_type == "classification":
            return evaluate_all_cls(preds, targets)
        else:
            spread_frac = torch.cat(all_spreads) / 10000.0
            return evaluate_all(preds, targets, spread_frac)

    def _primary_value(self, metrics):
        """Extract the scalar used for model selection from a metrics dict."""
        if self.task_type == "classification":
            return metrics.get("da", float("-inf"))
        return metrics.get("mae", float("inf"))

    def _log_epoch(self, epoch, total_epochs, train_loss, val_metrics, epoch_time):
        lr   = self.optimizer.param_groups[0]["lr"]
        wall = (time.time() - self.start_time) / 3600
        if self.task_type == "classification":
            log.info(
                f"Epoch {epoch+1}/{total_epochs} | loss={train_loss:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | da={val_metrics['da']:.4f} | "
                f"ordinal_ic={val_metrics['ordinal_ic']:.4f} | "
                f"lr={lr:.2e} | {epoch_time/60:.1f}min | wall={wall:.1f}h"
            )
        else:
            net_ic_val = val_metrics.get("net_ic", float("nan"))
            log.info(
                f"Epoch {epoch+1}/{total_epochs} | loss={train_loss:.6f} | "
                f"val_mae={val_metrics['mae']:.6f} | val_da={val_metrics['da']:.4f} | "
                f"val_ic={val_metrics['ic']:.4f} | val_net_ic={net_ic_val:.4f} | "
                f"lr={lr:.2e} | {epoch_time/60:.1f}min | wall={wall:.1f}h"
            )

    def save_checkpoint(self, tag="best"):
        if not self.is_main:
            return
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        ckpt = {
            "epoch":                self.epoch,
            "model_state_dict":     model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_primary":         self.best_primary,
            "patience_counter":     self.patience_counter,
            "train_log":            self.train_log,
            "config":               self.cfg,
        }
        path = os.path.join(self.exp_dir, f"checkpoint_{tag}.pt")
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, tag="best"):
        path = os.path.join(self.exp_dir, f"checkpoint_{tag}.pt")
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        if self.distributed:
            self.model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
        return True

    def run(self):
        cfg = self.cfg
        if self.is_main:
            log.info(f"Starting training: {cfg['experiment_name']}")
            log.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

        for epoch in range(self.start_epoch, cfg["max_epochs"]):
            self.epoch = epoch

            if self._time_remaining() < 600:
                if self.is_main:
                    log.info("Approaching wall time — finalizing.")
                break

            epoch_start = time.time()
            train_loss  = self.train_epoch()
            if train_loss is None:
                break

            val_metrics = self.evaluate(self.val_loader)
            epoch_time  = time.time() - epoch_start
            primary_val = self._primary_value(val_metrics)

            if self.is_main:
                self._log_epoch(epoch, cfg["max_epochs"], train_loss, val_metrics, epoch_time)
                self.train_log.append({
                    "epoch":        epoch + 1,
                    "train_loss":   train_loss,
                    **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                    "lr":           self.optimizer.param_groups[0]["lr"],
                    "epoch_time_s": epoch_time,
                })

            # Always overwrite last checkpoint (for resumption)
            self.save_checkpoint("last")

            # Update best checkpoint and early stopping counter
            if self._is_better(primary_val, self.best_primary):
                self.best_primary     = primary_val
                self.patience_counter = 0
                best_path = self.save_checkpoint("best")
                if self.is_main:
                    metric_name = "DA" if self._higher_is_better else "MAE"
                    log.info(f"  -> New best {metric_name}: {primary_val:.6f}")
                    update_global_best(
                        cfg["model"].lower(), primary_val,
                        cfg["experiment_name"], best_path,
                        cfg["results_dir"], task_type=self.task_type,
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg["patience"]:
                    if self.is_main:
                        log.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Persist incremental progress for monitoring
            if self.is_main:
                with open(os.path.join(self.exp_dir, "progress.json"), "w") as f:
                    json.dump({
                        "experiment":       cfg["experiment_name"],
                        "model":            cfg["model"],
                        "task_type":        self.task_type,
                        "current_epoch":    epoch + 1,
                        "best_primary":     float(self.best_primary),
                        "patience_counter": self.patience_counter,
                        "train_log":        self.train_log,
                        "config":           cfg,
                    }, f, indent=2, default=str)

        # Final evaluation on held-out test set using best checkpoint
        if not self.load_checkpoint("best"):
            log.warning("No best checkpoint found; evaluating current model weights.")
        test_metrics = self.evaluate(self.test_loader)
        val_metrics  = self.evaluate(self.val_loader)

        if self.is_main:
            log.info("=" * 60)
            log.info(f"FINAL RESULTS — {cfg['experiment_name']}")
            if self.task_type == "classification":
                log.info(f"  Val:  acc={val_metrics['accuracy']:.4f}  "
                         f"da={val_metrics['da']:.4f}  "
                         f"ordinal_ic={val_metrics['ordinal_ic']:.4f}")
                log.info(f"  Test: acc={test_metrics['accuracy']:.4f}  "
                         f"da={test_metrics['da']:.4f}  "
                         f"ordinal_ic={test_metrics['ordinal_ic']:.4f}")
            else:
                log.info(f"  Val:  MAE={val_metrics['mae']:.6f}  "
                         f"DA={val_metrics['da']:.4f}  "
                         f"IC={val_metrics['ic']:.4f}  "
                         f"net_IC={val_metrics.get('net_ic', float('nan')):.4f}")
                log.info(f"  Test: MAE={test_metrics['mae']:.6f}  "
                         f"DA={test_metrics['da']:.4f}  "
                         f"IC={test_metrics['ic']:.4f}  "
                         f"net_IC={test_metrics.get('net_ic', float('nan')):.4f}")
            log.info("=" * 60)

            results = {
                "experiment":   cfg["experiment_name"],
                "model":        cfg["model"],
                "task_type":    self.task_type,
                "best_epoch":   self.epoch + 1 - self.patience_counter,
                "total_epochs": self.epoch + 1,
                "val":          {k: float(v) for k, v in val_metrics.items()},
                "test":         {k: float(v) for k, v in test_metrics.items()},
                "config":       cfg,
                "train_log":    self.train_log,
            }
            results_path = os.path.join(self.exp_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            log.info(f"Results saved → {results_path}")

            # Final global best update (in case the last loaded epoch is best)
            update_global_best(
                cfg["model"].lower(),
                self._primary_value(val_metrics),
                cfg["experiment_name"],
                os.path.join(self.exp_dir, "checkpoint_best.pt"),
                cfg["results_dir"],
                task_type=self.task_type,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC HFT Model Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (see configs/*.yaml)")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **load_config(args.config)}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Resolve relative data_dir against the script's location
    if not os.path.isabs(cfg["data_dir"]):
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        cfg["data_dir"] = os.path.normpath(os.path.join(script_dir, cfg["data_dir"]))

    trainer = Trainer(cfg, local_rank=local_rank, world_size=world_size)
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
