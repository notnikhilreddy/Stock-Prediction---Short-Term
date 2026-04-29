#!/usr/bin/env python3
"""
Step 3 — Backtest
==================
Loads a trained model checkpoint, runs sliding-window inference over the test
set, and simulates a threshold-based long/short trading strategy with realistic
bid-ask spread costs.

Outputs per experiment:
    results/backtest/<experiment>_h<horizon>_backtest.json  — full metrics sweep
    results/backtest/<experiment>_h<horizon>_backtest.png   — 5-panel dashboard

Supports both regression (scalar signal) and classification (P(Up)−P(Down) signal).

Usage:
    # Single model, single horizon
    python backtest.py \\
        --checkpoint results/_global_best/nstransformer_reg_best.pt \\
        --data-dir ../data-processed --horizon 1

    # Compare two models side by side
    python backtest.py \\
        --checkpoint results/_global_best/nstransformer_reg_best.pt \\
        --compare   results/_global_best/patchtst_reg_best.pt \\
        --data-dir ../data-processed --horizon 10

    # Run all horizons for both NST and PatchTST regression global bests
    python backtest.py --all --data-dir ../data-processed

    # Run classification models across all horizons
    python backtest.py --all-cls --data-dir ../data-processed

    # Run all three architectures and produce a 3-way comparison plot
    python backtest.py --all-three --data-dir ../data-processed

Inference stride:
    --stride 1   full coverage (slow; use on GPU / high-memory node)
    --stride 64  every 64th tick (fast; good for quick checks)
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.patchtst import PatchTST
from models.ns_transformer import NSTransformer
from models.timesnet import TimesNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# F02_spread_bps is column index 1 in the 56-feature array
SPREAD_COL = 1
# Target column indices in targets array: y_1=0, y_5=1, y_10=2
HORIZON_TO_IDX = {1: 0, 5: 1, 10: 2}
HORIZON_TO_STEPS = {1: 1, 5: 5, 10: 10}

# Regression: thresholds as multiples of entry spread (0 = trade everything)
DEFAULT_THRESHOLDS_REG = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Classification: thresholds on |P(Up) - P(Down)| probability margin
DEFAULT_THRESHOLDS_CLS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

DEFAULT_THRESHOLDS = DEFAULT_THRESHOLDS_REG  # backward compat alias


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path, device):
    """Load a model and its config from a .pt checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    model_type = cfg.get("model", "nstransformer").lower()
    n_classes  = cfg.get("n_classes", 1)

    if model_type == "patchtst":
        model = PatchTST(
            n_features=cfg.get("n_features", 56),
            seq_len=cfg.get("seq_len", 512),
            patch_len=cfg.get("patch_len", 16),
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 3),
            d_ff=cfg.get("d_ff"),
            dropout=cfg.get("dropout", 0.1),
            head_dropout=cfg.get("head_dropout", 0.1),
            n_classes=n_classes,
        )
    elif model_type in ("nstransformer", "ns_transformer"):
        model = NSTransformer(
            n_features=cfg.get("n_features", 56),
            seq_len=cfg.get("seq_len", 512),
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            d_ff=cfg.get("d_ff"),
            dropout=cfg.get("dropout", 0.1),
            head_dropout=cfg.get("head_dropout", 0.1),
            n_classes=n_classes,
        )
    elif model_type == "timesnet":
        model = TimesNet(
            n_features=cfg.get("n_features", 56),
            seq_len=cfg.get("seq_len", 512),
            d_model=cfg.get("d_model", 128),
            n_layers=cfg.get("n_layers", 3),
            d_ff=cfg.get("d_ff"),
            top_k=cfg.get("top_k", 5),
            num_kernels=cfg.get("num_kernels", 3),
            dropout=cfg.get("dropout", 0.1),
            head_dropout=cfg.get("head_dropout", 0.1),
            n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    task_type  = cfg.get("task_type", "regression")
    experiment = cfg.get("experiment_name", os.path.basename(checkpoint_path))
    log.info(f"Loaded {model_type} [{experiment}] task={task_type} n_classes={n_classes} "
             f"from {checkpoint_path}")
    return model, cfg, experiment


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Runs sliding-window inference over a test set and simulates
    a threshold-based trading strategy using spread costs.

    Supports both regression (scalar output) and classification (n_classes logits)
    checkpoints. For classification, the signal is P(Up) - P(Down) and thresholds
    operate on the probability margin rather than spread multiples.
    """

    def __init__(self, checkpoint_path, data_dir, device="cpu"):
        self.device   = torch.device(device)
        self.data_dir = data_dir

        self.model, self.cfg, self.experiment = load_model_from_checkpoint(
            checkpoint_path, self.device
        )
        self.seq_len  = self.cfg.get("seq_len", 512)
        self.task_type = self.cfg.get("task_type", "regression")
        self.n_classes = self.cfg.get("n_classes", 1)

        log.info(f"Loading test data from {data_dir} ...")
        feat_path = os.path.join(data_dir, "test_features.npy")
        tgt_path  = os.path.join(data_dir, "test_targets.npy")
        self.features = np.load(feat_path, mmap_mode="r")   # [N, 56]
        self.targets  = np.load(tgt_path,  mmap_mode="r")   # [N, 3]
        log.info(f"Test set: {len(self.features):,} rows, "
                 f"{self.features.shape[1]} features")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_inference_on_positions(self, positions, batch_size, target_idx):
        """
        Run forward passes for a given int32 array of window-end row indices.
        target_idx is unused for the forward pass but kept for API symmetry.
        """
        import torch.nn.functional as F

        n_windows = len(positions)
        if n_windows == 0:
            return np.array([], dtype=np.float32)

        signals = np.empty(n_windows, dtype=np.float32)
        for start in range(0, n_windows, batch_size):
            batch_pos = positions[start: start + batch_size]
            batch_x   = np.stack([
                self.features[int(p) - self.seq_len: int(p)].copy()
                for p in batch_pos
            ], axis=0)
            x_t  = torch.from_numpy(batch_x).to(self.device)
            out  = self.model(x_t).float().cpu()

            if self.task_type == "classification" and self.n_classes > 1:
                probs = F.softmax(out, dim=-1)
                sig   = (probs[:, 2] - probs[:, 0]).numpy()
            else:
                sig = out.numpy().ravel()

            signals[start: start + len(sig)] = sig

            if n_windows > 0 and (start // batch_size) % 20 == 0:
                pct = 100 * start / n_windows
                log.info(f"  Inference {pct:.0f}% ({start:,}/{n_windows:,})")

        log.info(
            "Inference complete. Signal range: "
            f"[{signals.min():.3e}, {signals.max():.3e}]"
        )
        return signals

    @torch.no_grad()
    def run_inference(self, target_idx, batch_size=512, stride=64):
        """
        Slide a window of length seq_len over the test features and collect
        model predictions aligned with target positions.

        stride: step between consecutive windows (1 = every tick, 64 = every 64th).
                Use stride=1 on GPU/bigmem nodes for full coverage.
                Use stride=64 on login nodes / quick checks.

        Returns:
            positions : int array   [M]      — row index for each window end
            signals   : float array [M]      — scalar trading signal:
                        regression:     raw model output
                        classification: P(Up) - P(Down)  (in [-1, 1])
        """
        N = len(self.features)
        if N <= self.seq_len:
            raise ValueError("Test set smaller than seq_len")

        positions = np.arange(self.seq_len, N, stride, dtype=np.int32)
        n_windows = len(positions)
        log.info(f"Running inference: {n_windows:,} windows "
                 f"(stride={stride}, batch_size={batch_size}, task={self.task_type})")
        signals = self._run_inference_on_positions(positions, batch_size, target_idx)
        return positions, signals

    @torch.no_grad()
    def run_inference_shard(self, target_idx, batch_size=512, stride=64,
                          shard_id=0, num_shards=1):
        """
        Same sliding-window inference as run_inference, but only for a contiguous
        slice of window indices (for parallel SLURM array jobs).

        Shard i covers window indices [i * chunk, min((i+1)*chunk, n_windows)).
        """
        N = len(self.features)
        if N <= self.seq_len:
            raise ValueError("Test set smaller than seq_len")

        positions = np.arange(self.seq_len, N, stride, dtype=np.int32)
        n_windows = len(positions)
        chunk = (n_windows + num_shards - 1) // num_shards
        lo = shard_id * chunk
        hi = min(lo + chunk, n_windows)

        log.info(f"Shard {shard_id}/{num_shards}: windows [{lo:,}, {hi:,}) "
                 f"of {n_windows:,} (stride={stride}, batch_size={batch_size})")

        if lo >= hi:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        sub_positions = positions[lo:hi]
        signals = self._run_inference_on_positions(sub_positions, batch_size, target_idx)
        return sub_positions, signals

    # ------------------------------------------------------------------
    # Strategy simulation
    # ------------------------------------------------------------------

    def simulate(self, positions, signals, target_idx, horizon_steps,
                 threshold_multiplier=1.0):
        """
        Simulate threshold-based long/short trading.

        Regression mode:
          Trade condition: |signal| > threshold_multiplier × spread_entry
          (threshold_multiplier is a spread multiple, e.g. 1.0 = 1× spread)

        Classification mode:
          Trade condition: |signal| > threshold_multiplier
          (signal = P(Up)-P(Down) in [-1,1], threshold_multiplier is a prob margin)

        Net return per trade:
          Long:  y_K[t] - spread_entry/2 - spread_exit/2
          Short: -y_K[t] - spread_entry/2 - spread_exit/2
        """
        N = len(self.features)
        n = len(positions)

        y_K          = self.targets[positions, target_idx]
        spread_entry = self.features[positions, SPREAD_COL] / 10000.0

        exit_pos        = np.minimum(positions + horizon_steps, N - 1)
        spread_exit     = self.features[exit_pos, SPREAD_COL] / 10000.0
        round_trip_cost = spread_entry / 2.0 + spread_exit / 2.0

        if self.task_type == "classification" and self.n_classes > 1:
            threshold = threshold_multiplier            # fixed prob-margin cutoff
        else:
            threshold = threshold_multiplier * spread_entry   # spread-relative cutoff

        take_trade = np.abs(signals) > threshold
        go_long    = signals > 0

        net_returns = np.where(
            go_long,
            y_K - round_trip_cost,
            -y_K - round_trip_cost,
        )

        selected_returns = net_returns[take_trade]
        selected_signals = signals[take_trade]
        selected_pos     = positions[take_trade]
        trade_dir        = np.where(go_long[take_trade], 1, -1)

        return {
            "n_candidates":        int(n),
            "n_trades":            int(take_trade.sum()),
            "returns":             selected_returns,
            "predictions":         selected_signals,
            "positions":           selected_pos,
            "directions":          trade_dir,
            "threshold_multiplier":threshold_multiplier,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(sim_result):
        """Compute all trading metrics from simulate() output."""
        from utils.metrics import trading_metrics, ic as pearson_ic

        returns = sim_result["returns"]
        m = trading_metrics(returns)

        # Net IC on the traded subset: |pred| vs net_return
        if len(returns) > 1:
            preds_t  = torch.from_numpy(sim_result["predictions"].astype(np.float32))
            rets_t   = torch.from_numpy(returns.astype(np.float32))
            abs_pred = torch.abs(preds_t)
            m["net_ic_traded"] = pearson_ic(abs_pred, rets_t)
        else:
            m["net_ic_traded"] = float("nan")

        m["threshold_multiplier"] = sim_result["threshold_multiplier"]
        m["n_candidates"]         = sim_result["n_candidates"]
        m["coverage"]             = (sim_result["n_trades"] / sim_result["n_candidates"]
                                     if sim_result["n_candidates"] > 0 else 0.0)
        return m

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------

    def threshold_sweep(self, positions, predictions, target_idx, horizon_steps,
                        thresholds=None):
        """
        Simulate strategy at each threshold and return list of metric dicts.
        """
        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS

        results = []
        for thr in thresholds:
            sim = self.simulate(positions, predictions, target_idx,
                                horizon_steps, threshold_multiplier=thr)
            m = self.compute_metrics(sim)
            results.append((thr, sim, m))
            log.info(
                f"  thr={thr:.2f}x | trades={m['n_trades']:,} "
                f"({100*m['coverage']:.1f}%) | "
                f"hit={100*m.get('hit_rate', 0):.1f}% | "
                f"mean_ret={m.get('mean_return', 0):.3e} | "
                f"sharpe={m.get('sharpe', 0):.3f} | "
                f"pf={m.get('profit_factor', 0):.2f}"
            )
        return results

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def finalize_sweep(self, positions, predictions, horizon, stride, out_dir,
                       thresholds=None):
        """
        Threshold sweep → best Sharpe → JSON + PNG. Used by run() and by
        sharded merge after concatenating per-shard predictions.
        """
        target_idx    = HORIZON_TO_IDX[horizon]
        horizon_steps = HORIZON_TO_STEPS[horizon]

        if thresholds is None:
            if self.task_type == "classification" and self.n_classes > 1:
                thresholds = DEFAULT_THRESHOLDS_CLS
            else:
                thresholds = DEFAULT_THRESHOLDS_REG

        log.info(
            f"=== Backtest finalize: {self.experiment} | h{horizon} | stride={stride} "
            f"| task={self.task_type} | n_windows={len(positions):,} ==="
        )

        log.info("Running threshold sweep ...")
        sweep = self.threshold_sweep(positions, predictions, target_idx,
                                     horizon_steps, thresholds)

        valid = [(thr, sim, m) for thr, sim, m in sweep
                 if m["n_trades"] >= 30 and not np.isnan(m.get("sharpe", np.nan))]
        best_thr, best_sim, best_m = max(valid, key=lambda x: x[2]["sharpe"]) if valid \
            else sweep[0]
        log.info(f"Optimal threshold: {best_thr:.2f}x spread "
                 f"(Sharpe={best_m.get('sharpe', 0):.3f})")

        if out_dir is None:
            out_dir = os.path.join(SCRIPT_DIR, "results", "backtest")
        os.makedirs(out_dir, exist_ok=True)

        report = {
            "experiment": self.experiment,
            "horizon": horizon,
            "stride": stride,
            "generated_at": datetime.now().isoformat(),
            "optimal_threshold": best_thr,
            "optimal_metrics": best_m,
            "sweep": [m for _, _, m in sweep],
        }
        for entry in report["sweep"]:
            entry.pop("returns", None)

        json_path = os.path.join(out_dir, f"{self.experiment}_h{horizon}_backtest.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        log.info(f"Metrics saved → {json_path}")

        png_path = self._plot_dashboard(sweep, best_thr, best_sim, best_m,
                                        horizon, out_dir)
        return sweep, png_path

    def run(self, horizon=1, thresholds=None, out_dir=None,
            stride=64, batch_size=512):
        """
        End-to-end: inference → sweep → report → plot.
        Returns the sweep results list and the path to the dashboard PNG.

        stride: window stride for inference (1=every tick, 64=every 64th).
                Use 1 on GPU/bigmem nodes; 64 for quick login-node checks.
        """
        target_idx = HORIZON_TO_IDX[horizon]

        if thresholds is None:
            if self.task_type == "classification" and self.n_classes > 1:
                thresholds = DEFAULT_THRESHOLDS_CLS
            else:
                thresholds = DEFAULT_THRESHOLDS_REG

        log.info(f"=== Backtest: {self.experiment} | h{horizon} | stride={stride} "
                 f"| task={self.task_type} ===")
        positions, predictions = self.run_inference(
            target_idx, batch_size=batch_size, stride=stride
        )

        if out_dir is None:
            out_dir = os.path.join(SCRIPT_DIR, "results", "backtest")
        return self.finalize_sweep(positions, predictions, horizon, stride,
                                   out_dir, thresholds)

    # ------------------------------------------------------------------
    # Dashboard plot
    # ------------------------------------------------------------------

    def _plot_dashboard(self, sweep, best_thr, best_sim, best_m, horizon, out_dir):
        thresholds = [t for t, _, _ in sweep]
        metrics    = [m for _, _, m in sweep]

        sharpes     = [m.get("sharpe",       0)   for m in metrics]
        hit_rates   = [m.get("hit_rate",     0)   for m in metrics]
        n_trades    = [m.get("n_trades",     0)   for m in metrics]
        mean_rets   = [m.get("mean_return",  0)   for m in metrics]
        sortinos    = [m.get("sortino",      0)   for m in metrics]

        best_returns = best_sim["returns"]
        cum_pnl      = np.cumsum(best_returns) if len(best_returns) else np.array([0.0])
        running_max  = np.maximum.accumulate(cum_pnl)
        drawdown     = running_max - cum_pnl

        fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32,
                                top=0.93, bottom=0.05, left=0.07, right=0.96)

        txt_kw   = dict(color="#e6edf3", fontfamily="monospace")
        title_kw = dict(color="#58a6ff", fontsize=11, fontweight="bold", pad=8)
        bg_col   = "#161b22"
        grid_col = "#21262d"
        accent   = "#58a6ff"

        fig.text(0.5, 0.965,
                 f"Real-World Backtest — {self.experiment} | h{horizon} "
                 f"({horizon * 100}ms horizon)",
                 ha="center", fontsize=16, fontweight="bold",
                 color="#58a6ff", fontfamily="monospace")
        fig.text(0.5, 0.952,
                 f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
                 f"Optimal threshold: {best_thr:.2f}× spread  |  "
                 f"Trades: {best_m.get('n_trades', 0):,}  |  "
                 f"Sharpe: {best_m.get('sharpe', 0):.3f}  |  "
                 f"Hit rate: {100*best_m.get('hit_rate', 0):.1f}%",
                 ha="center", fontsize=9, color="#8b949e", fontfamily="monospace")

        def style(ax, title):
            ax.set_facecolor(bg_col)
            ax.set_title(title, **title_kw)
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            ax.xaxis.grid(True, color=grid_col, linewidth=0.6)
            ax.yaxis.grid(True, color=grid_col, linewidth=0.6)
            ax.set_axisbelow(True)

        # --- Panel 1: Cumulative P&L at optimal threshold ---
        ax1 = fig.add_subplot(gs[0, :])
        style(ax1, f"Cumulative P&L — threshold={best_thr:.2f}× spread "
              f"| {len(best_returns):,} trades | Total={cum_pnl[-1]:.4f}")
        if len(cum_pnl) > 1:
            x_axis = np.arange(len(cum_pnl))
            ax1.plot(x_axis, cum_pnl, color=accent, linewidth=1.2, label="Cumulative P&L")
            ax1.fill_between(x_axis, cum_pnl, 0,
                             where=cum_pnl >= 0, alpha=0.15, color="#3fb950")
            ax1.fill_between(x_axis, cum_pnl, 0,
                             where=cum_pnl < 0,  alpha=0.15, color="#f85149")
            ax1.axhline(0, color="#30363d", linewidth=1)
        ax1.set_xlabel("Trade #", **txt_kw, fontsize=9)
        ax1.set_ylabel("Cumulative log-return", **txt_kw, fontsize=9)

        # --- Panel 2: Sharpe ratio vs threshold ---
        ax2 = fig.add_subplot(gs[1, 0])
        style(ax2, "Sharpe Ratio vs Threshold (× spread)")
        ax2.plot(thresholds, sharpes, color=accent, marker="o", markersize=5,
                 linewidth=1.5)
        ax2.axvline(best_thr, color="#f0883e", linewidth=1.5, linestyle="--",
                    label=f"Optimal ({best_thr:.2f}×)")
        ax2.axhline(0, color="#30363d", linewidth=1)
        ax2.axhline(1, color="#3fb950", linewidth=0.8, linestyle=":", label="Sharpe=1")
        ax2.set_xlabel("Threshold (× spread)", **txt_kw, fontsize=9)
        ax2.set_ylabel("Sharpe ratio", **txt_kw, fontsize=9)
        ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3")

        # --- Panel 3: Hit rate vs threshold ---
        ax3 = fig.add_subplot(gs[1, 1])
        style(ax3, "Hit Rate vs Threshold")
        ax3.plot(thresholds, [h * 100 for h in hit_rates],
                 color="#3fb950", marker="s", markersize=5, linewidth=1.5)
        ax3.axvline(best_thr, color="#f0883e", linewidth=1.5, linestyle="--")
        ax3.axhline(50, color="#30363d", linewidth=1, linestyle=":", label="50% (random)")
        ax3.set_xlabel("Threshold (× spread)", **txt_kw, fontsize=9)
        ax3.set_ylabel("Hit rate (%)", **txt_kw, fontsize=9)
        ax3.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3")

        # --- Panel 4: Trade count vs threshold ---
        ax4 = fig.add_subplot(gs[2, 0])
        style(ax4, "Trade Count vs Threshold")
        ax4.plot(thresholds, n_trades, color="#ffd166", marker="^",
                 markersize=5, linewidth=1.5)
        ax4.axvline(best_thr, color="#f0883e", linewidth=1.5, linestyle="--",
                    label=f"Optimal ({best_thr:.2f}×)")
        ax4.set_xlabel("Threshold (× spread)", **txt_kw, fontsize=9)
        ax4.set_ylabel("Number of trades", **txt_kw, fontsize=9)
        ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3")

        # --- Panel 5: Drawdown curve ---
        ax5 = fig.add_subplot(gs[2, 1])
        style(ax5, f"Drawdown Curve — threshold={best_thr:.2f}×")
        if len(drawdown) > 1:
            ax5.fill_between(np.arange(len(drawdown)), -drawdown, 0,
                             alpha=0.6, color="#f85149")
            ax5.plot(np.arange(len(drawdown)), -drawdown,
                     color="#f85149", linewidth=0.8)
        ax5.set_xlabel("Trade #", **txt_kw, fontsize=9)
        ax5.set_ylabel("Drawdown (log-return)", **txt_kw, fontsize=9)
        ax5.axhline(0, color="#30363d", linewidth=1)

        png_path = os.path.join(out_dir,
                                f"{self.experiment}_h{horizon}_backtest.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"Dashboard saved → {png_path}")
        return png_path


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(ckpt_a, ckpt_b, data_dir, horizon, device, out_dir,
                   stride=64, batch_size=512):
    """Run backtest for two models and produce a side-by-side comparison plot."""
    eng_a = BacktestEngine(ckpt_a, data_dir, device)
    sweep_a, _ = eng_a.run(horizon=horizon, out_dir=out_dir,
                            stride=stride, batch_size=batch_size)

    eng_b = BacktestEngine(ckpt_b, data_dir, device)
    sweep_b, _ = eng_b.run(horizon=horizon, out_dir=out_dir,
                            stride=stride, batch_size=batch_size)

    _plot_comparison(eng_a.experiment, sweep_a,
                     eng_b.experiment, sweep_b,
                     horizon, out_dir)


def _compare_three(names, ckpts, data_dir, horizon, device, out_dir,
                   stride=64, batch_size=512):
    """Run backtest for up to 3 models and produce a side-by-side comparison plot."""
    sweeps = []
    for ckpt in ckpts:
        eng = BacktestEngine(ckpt, data_dir, device)
        sweep, _ = eng.run(horizon=horizon, out_dir=out_dir,
                           stride=stride, batch_size=batch_size)
        sweeps.append((eng.experiment, sweep))

    _plot_comparison_multi(sweeps, horizon, out_dir)


def _plot_comparison_multi(sweeps, horizon, out_dir):
    """
    Multi-model comparison plot (2 or 3 architectures).
    sweeps: list of (name, sweep_list) tuples.
    """
    colors = ["#58a6ff", "#f85149", "#3fb950", "#ffd166"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), facecolor="#0d1117")
    fig.suptitle(
        f"Architecture Comparison — h{horizon} ({horizon*100}ms)",
        color="#58a6ff", fontsize=14, fontweight="bold", fontfamily="monospace",
    )

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#58a6ff", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.xaxis.grid(True, color="#21262d", linewidth=0.6)
        ax.yaxis.grid(True, color="#21262d", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel, color="#e6edf3", fontfamily="monospace", fontsize=9)
        ax.set_ylabel(ylabel, color="#e6edf3", fontfamily="monospace", fontsize=9)

    style(axes[0], "Sharpe Ratio vs Threshold", "Threshold (× spread)", "Sharpe ratio")
    style(axes[1], "Hit Rate vs Threshold",     "Threshold (× spread)", "Hit rate (%)")
    style(axes[2], "Trade Count vs Threshold",  "Threshold (× spread)", "# Trades")

    for i, (name, sweep) in enumerate(sweeps):
        thr = [t for t, _, _ in sweep]
        sh  = [m.get("sharpe", 0)          for _, _, m in sweep]
        hr  = [100 * m.get("hit_rate", 0)  for _, _, m in sweep]
        nt  = [m.get("n_trades", 0)        for _, _, m in sweep]
        col = colors[i % len(colors)]

        best_valid = [(t, m) for t, _, m in sweep
                      if m["n_trades"] >= 30 and not np.isnan(m.get("sharpe", float("nan")))]
        best_thr = max(best_valid, key=lambda x: x[1]["sharpe"])[0] if best_valid else thr[0]
        best_sh  = max((m.get("sharpe", 0) for t, m in best_valid), default=0)
        label = f"{name} (best Sharpe={best_sh:.3f})"

        for ax, vals in zip(axes, [sh, hr, nt]):
            ax.plot(thr, vals, marker="o", markersize=4, linewidth=1.5, color=col, label=label)
            ax.axvline(best_thr, color=col, linewidth=1, linestyle="--", alpha=0.5)

    for ax in axes:
        ax.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    axes[0].axhline(0, color="#30363d", linewidth=1)
    axes[0].axhline(1, color="#3fb950", linewidth=0.8, linestyle=":", alpha=0.6)
    axes[1].axhline(50, color="#30363d", linewidth=1, linestyle=":", alpha=0.6)

    name_slug = "_".join(n.replace(" ", "").replace("-", "") for n, _ in sweeps)
    fname    = f"comparison_{name_slug}_h{horizon}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"3-way comparison plot saved → {out_path}")


def _plot_comparison(name_a, sweep_a, name_b, sweep_b, horizon, out_dir):
    thr_a = [t for t, _, _ in sweep_a]
    thr_b = [t for t, _, _ in sweep_b]
    sh_a  = [m.get("sharpe", 0) for _, _, m in sweep_a]
    sh_b  = [m.get("sharpe", 0) for _, _, m in sweep_b]
    hr_a  = [100 * m.get("hit_rate", 0) for _, _, m in sweep_a]
    hr_b  = [100 * m.get("hit_rate", 0) for _, _, m in sweep_b]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d1117")
    fig.suptitle(f"Architecture Comparison — h{horizon} ({horizon*100}ms)",
                 color="#58a6ff", fontsize=14, fontweight="bold",
                 fontfamily="monospace")

    for ax, data_pairs, ylabel, title in [
        (axes[0],
         [(thr_a, sh_a, name_a, "#58a6ff"), (thr_b, sh_b, name_b, "#f85149")],
         "Sharpe ratio", "Sharpe vs Threshold"),
        (axes[1],
         [(thr_a, hr_a, name_a, "#58a6ff"), (thr_b, hr_b, name_b, "#f85149")],
         "Hit rate (%)", "Hit Rate vs Threshold"),
    ]:
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#58a6ff", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.xaxis.grid(True, color="#21262d", linewidth=0.6)
        ax.yaxis.grid(True, color="#21262d", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Threshold (× spread)", color="#e6edf3",
                      fontfamily="monospace", fontsize=9)
        ax.set_ylabel(ylabel, color="#e6edf3", fontfamily="monospace", fontsize=9)
        for thr, vals, label, col in data_pairs:
            ax.plot(thr, vals, marker="o", markersize=4, linewidth=1.5,
                    color=col, label=label)
        ax.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                  labelcolor="#e6edf3")

    fname = f"comparison_{name_a}_vs_{name_b}_h{horizon}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Comparison plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_infer_shard(checkpoint, data_dir, device, shard_dir, shard_id,
                      num_shards, horizon, stride, batch_size):
    """
    Run inference for one shard and write shard_XXXX.npz plus meta.json
    (from shard 0 only) for later merge_sharded_backtest().
    """
    eng = BacktestEngine(checkpoint, data_dir, device)
    target_idx = HORIZON_TO_IDX[horizon]
    N = len(eng.features)
    positions_full = np.arange(eng.seq_len, N, stride, dtype=np.int32)
    n_windows = int(len(positions_full))

    os.makedirs(shard_dir, exist_ok=True)
    if shard_id == 0:
        meta = {
            "checkpoint": checkpoint,
            "data_dir": data_dir,
            "experiment": eng.experiment,
            "horizon": int(horizon),
            "stride": int(stride),
            "num_shards": int(num_shards),
            "n_windows": n_windows,
            "task_type": eng.task_type,
            "n_classes": int(eng.n_classes),
            "seq_len": int(eng.seq_len),
        }
        with open(os.path.join(shard_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    pos, sig = eng.run_inference_shard(
        target_idx,
        batch_size=batch_size,
        stride=stride,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    out_path = os.path.join(shard_dir, f"shard_{shard_id:04d}.npz")
    np.savez_compressed(out_path, positions=pos, signals=sig)
    log.info(f"Wrote {out_path} ({len(pos):,} windows)")


def merge_sharded_backtest(checkpoint, data_dir, shard_dir, horizon, out_dir,
                           device="cpu"):
    """
    Load meta.json + all shard_*.npz, concatenate predictions, run threshold
    sweep and write the same JSON/PNG as a monolithic backtest run.
    """
    meta_path = os.path.join(shard_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    K = int(meta["num_shards"])
    parts_p, parts_s = [], []
    for i in range(K):
        zp = os.path.join(shard_dir, f"shard_{i:04d}.npz")
        if not os.path.isfile(zp):
            raise FileNotFoundError(f"Missing shard file {zp}")
        z = np.load(zp)
        parts_p.append(np.asarray(z["positions"], dtype=np.int32))
        parts_s.append(np.asarray(z["signals"], dtype=np.float32))

    positions = np.concatenate(parts_p)
    predictions = np.concatenate(parts_s)
    if int(meta["n_windows"]) != len(positions):
        raise ValueError(
            f"Shard merge: meta n_windows={meta['n_windows']} != {len(positions)}"
        )

    stride = int(meta["stride"])
    eng = BacktestEngine(checkpoint, data_dir, device)
    return eng.finalize_sweep(positions, predictions, horizon, stride, out_dir)


def _find_global_best(results_dir, arch, task="reg"):
    """
    Return path to global best checkpoint.

    arch: "nstransformer" or "patchtst"
    task: "reg" (regression) or "cls" (classification)

    Falls back to legacy filename if task-specific file not found.
    """
    p = os.path.join(results_dir, "_global_best", f"{arch}_{task}_best.pt")
    if os.path.exists(p):
        return p
    # Legacy fallback for checkpoints created before the task-type split
    legacy = os.path.join(results_dir, "_global_best", f"{arch}_best.pt")
    return legacy if os.path.exists(legacy) else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-world backtest for BTC HFT models"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Second checkpoint to compare against")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(SCRIPT_DIR), "data-processed"),
                        help="Directory containing test_features.npy and test_targets.npy")
    parser.add_argument("--horizon", type=int, default=None, choices=[1, 5, 10],
                        help="Prediction horizon in steps (1=100ms, 5=500ms, 10=1s). "
                             "Defaults to 1 for single-model runs; when used with --all-three "
                             "or --all-timesnet restricts to that horizon only.")
    parser.add_argument("--all", action="store_true",
                        help="Run all horizons for both NST and PTST regression global bests")
    parser.add_argument("--all-cls", action="store_true",
                        help="Run all horizons for both NST and PTST classification global bests")
    parser.add_argument("--all-timesnet", action="store_true",
                        help="Run all horizons for TimesNet regression global best")
    parser.add_argument("--all-three", action="store_true",
                        help="Run all horizons for all three architectures (NST, PTST, TimesNet) "
                             "and produce a 3-way comparison plot")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "results", "backtest"),
                        help="Output directory for reports and plots")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda:0)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Inference batch size")
    parser.add_argument("--stride", type=int, default=64,
                        help="Window stride for inference (1=full coverage/slow, 64=fast/approximate)")
    parser.add_argument(
        "--infer-shard", action="store_true",
        help="Write one inference shard to --shard-dir (enables parallelized inference)",
    )
    parser.add_argument(
        "--merge-shards", action="store_true",
        help="Merge shard_*.npz under --shard-dir and run threshold sweep",
    )
    parser.add_argument(
        "--shard-dir", type=str, default=None,
        help="Directory for sharded .npz files + meta.json",
    )
    parser.add_argument(
        "--shard-id", type=int, default=-1,
        help="Shard index for --infer-shard (default: reads SLURM_ARRAY_TASK_ID or 0)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards for --infer-shard",
    )
    args = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.infer_shard:
        if not args.checkpoint:
            parser.error("--infer-shard requires --checkpoint")
        if not args.shard_dir:
            parser.error("--infer-shard requires --shard-dir")
        sid = args.shard_id
        if sid < 0:
            sid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if args.num_shards < 1:
            parser.error("--num-shards must be >= 1")
        write_infer_shard(
            args.checkpoint,
            args.data_dir,
            args.device,
            args.shard_dir,
            sid,
            args.num_shards,
            args.horizon,
            args.stride,
            args.batch_size,
        )
        return

    if args.merge_shards:
        if not args.checkpoint:
            parser.error("--merge-shards requires --checkpoint")
        if not args.shard_dir:
            parser.error("--merge-shards requires --shard-dir")
        merge_sharded_backtest(
            args.checkpoint,
            args.data_dir,
            args.shard_dir,
            args.horizon,
            args.out_dir,
            device=args.device,
        )
        return

    if args.all or args.all_cls:
        task = "cls" if args.all_cls else "reg"
        nst_ckpt  = _find_global_best(results_dir, "nstransformer", task)
        ptst_ckpt = _find_global_best(results_dir, "patchtst",      task)
        for horizon in [1, 5, 10]:
            if nst_ckpt:
                eng = BacktestEngine(nst_ckpt, args.data_dir, args.device)
                eng.run(horizon=horizon, out_dir=args.out_dir,
                        stride=args.stride, batch_size=args.batch_size)
            if ptst_ckpt:
                eng = BacktestEngine(ptst_ckpt, args.data_dir, args.device)
                eng.run(horizon=horizon, out_dir=args.out_dir,
                        stride=args.stride, batch_size=args.batch_size)
            if nst_ckpt and ptst_ckpt:
                compare_models(nst_ckpt, ptst_ckpt, args.data_dir,
                               horizon, args.device, args.out_dir,
                               stride=args.stride, batch_size=args.batch_size)
        return

    if args.all_timesnet:
        tnet_ckpt = _find_global_best(results_dir, "timesnet", "reg")
        if not tnet_ckpt:
            log.error("No TimesNet global best checkpoint found.")
            sys.exit(1)
        horizons = [args.horizon] if args.horizon else [1, 5, 10]
        for horizon in horizons:
            eng = BacktestEngine(tnet_ckpt, args.data_dir, args.device)
            eng.run(horizon=horizon, out_dir=args.out_dir,
                    stride=args.stride, batch_size=args.batch_size)
        return

    if args.all_three:
        nst_ckpt  = _find_global_best(results_dir, "nstransformer", "reg")
        ptst_ckpt = _find_global_best(results_dir, "patchtst",      "reg")
        tnet_ckpt = _find_global_best(results_dir, "timesnet",      "reg")
        horizons = [args.horizon] if args.horizon else [1, 5, 10]
        for horizon in horizons:
            for ckpt in (nst_ckpt, ptst_ckpt, tnet_ckpt):
                if ckpt:
                    eng = BacktestEngine(ckpt, args.data_dir, args.device)
                    eng.run(horizon=horizon, out_dir=args.out_dir,
                            stride=args.stride, batch_size=args.batch_size)
            present = [(n, c) for n, c in [
                ("NS-Transformer", nst_ckpt),
                ("PatchTST",       ptst_ckpt),
                ("TimesNet",       tnet_ckpt),
            ] if c]
            if len(present) >= 2:
                names   = [n for n, _ in present]
                ckpts   = [c for _, c in present]
                _compare_three(names, ckpts, args.data_dir, horizon,
                               args.device, args.out_dir,
                               stride=args.stride, batch_size=args.batch_size)
        return

    if args.checkpoint is None:
        args.checkpoint = _find_global_best(results_dir, "nstransformer")
        if args.checkpoint is None:
            parser.error("No --checkpoint specified and no global best found.")

    horizon = args.horizon if args.horizon is not None else 1

    if args.compare:
        compare_models(args.checkpoint, args.compare,
                       args.data_dir, horizon, args.device, args.out_dir,
                       stride=args.stride, batch_size=args.batch_size)
    else:
        eng = BacktestEngine(args.checkpoint, args.data_dir, args.device)
        eng.run(horizon=horizon, out_dir=args.out_dir,
                stride=args.stride, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
