# BTC-USDT HFT Price Prediction Pipeline

End-to-end pipeline for training and backtesting deep learning models that
predict short-horizon BTC-USDT mid-price returns from L10 order book +
trade flow data at **100 ms resolution**.

---

## Overview

The pipeline has four sequential steps:

```
Raw parquets  →  [01] Feature Engineering  →  train / val / test parquets
                 [02] Training             →  checkpoint_best.pt
                 [03] Backtest             →  JSON report + PNG dashboard
                 [04] Plot Results         →  results_summary.png
```

Three model architectures are implemented and benchmarked:

| Architecture | Key mechanism | Params (small) |
|---|---|---|
| **NS-Transformer** | De-stationary attention re-scales Q/K via per-sequence μ, σ | ~1.5M |
| **PatchTST** | Channel-independent patching; each feature attends to its own patches | ~2.1M |
| **TimesNet** | FFT → dominant periods → 2D Inception convolutions | ~1.8M |

---

## Repository Layout

```
train-pipeline/
├── 01_feature_engineering.py   # raw parquets → 56-feature parquets
├── 02_train.py                 # model training (single or multi-GPU)
├── backtest.py                 # inference + trading strategy simulation
├── plot_results.py             # multi-panel results dashboard
│
├── models/
│   ├── ns_transformer.py       # Non-Stationary Transformer
│   ├── patchtst.py             # PatchTST with RevIN + channel independence
│   └── timesnet.py             # TimesNet (temporal 2D-variation)
│
├── utils/
│   ├── dataset.py              # BTCDataset — sliding-window DataLoader
│   └── metrics.py              # MAE, IC, net_IC, Sharpe, Sortino, …
│
├── configs/
│   ├── patchtst_small.yaml     # small PatchTST, h1 target
│   ├── patchtst_medium.yaml
│   ├── patchtst_large.yaml
│   ├── patchtst_longctx.yaml   # longer context window variant
│   ├── patchtst_patch32.yaml   # larger patch size variant
│   ├── nstransformer_small.yaml
│   ├── nstransformer_medium.yaml
│   ├── nstransformer_large.yaml
│   ├── nstransformer_longctx.yaml
│   ├── timesnet_small.yaml
│   ├── timesnet_medium.yaml
│   ├── timesnet_large.yaml
│   ├── timesnet_longctx.yaml
│   ├── *_h10_cls.yaml          # classification task variants
│   └── sweep/                  # TimesNet hyperparameter sweep configs
│
└── results/
    ├── _global_best/           # best checkpoint per arch × task type
    │   ├── nstransformer_reg_best.pt
    │   ├── patchtst_reg_best.pt
    │   └── timesnet_reg_best.pt
    ├── <experiment_name>/      # per-run outputs (checkpoint, results.json)
    └── backtest/               # backtest JSON reports + dashboard PNGs
```

---

## Features (56 total)

All features are computed at 100 ms resolution and stored as float32.

| # | Name | Description |
|---|---|---|
| F01 | `log_return` | 1-tick mid-price log return |
| F02 | `spread_bps` | Bid-ask spread in basis points |
| F03 | `micro_dev_bps` | Micro-price deviation from mid (bps) |
| F04 | `rvol_20` | Rolling return volatility (20-tick std) |
| F05 | `spread_z` | Spread z-score (100-tick rolling) |
| F06–F10 | `d_ask_1..5` | Ask level distances L1–L5 from mid (bps) |
| F11–F15 | `d_bid_1..5` | Bid level distances L1–L5 from mid (bps) |
| F16–F25 | `v_ask_1..10` | Log ask volume at each of 10 book levels |
| F26–F35 | `v_bid_1..10` | Log bid volume at each of 10 book levels |
| F36 | `bid_slope` | Linear fit slope of bid log-volumes across levels |
| F37 | `ask_slope` | Linear fit slope of ask log-volumes across levels |
| F38 | `obi_l1` | Order book imbalance at L1: (Vb−Va)/(Vb+Va) |
| F39 | `obi_near` | OBI over L1–L5 |
| F40 | `obi_deep` | OBI over L6–L10 |
| F41 | `wobi` | Weighted OBI (1/level weights) |
| F42 | `dobi` | Delta OBI (first difference of F38) |
| F43 | `ofi_bid` | Bid order flow imbalance (normalized) |
| F44 | `ofi_ask` | Ask order flow imbalance (normalized) |
| F45 | `cofi_10` | Cumulative net OFI over 10 ticks |
| F46 | `tfi` | Trade flow imbalance: (buy vol − sell vol) / total |
| F47 | `log_vol` | Log total trade volume in window |
| F48 | `log_ticks` | Log trade tick count |
| F49 | `vwap_dev_bps` | VWAP deviation from mid (bps) |
| F50 | `micro_vol_bps` | High–low range within window (bps) |
| F51 | `aggr_ratio` | Aggressor ratio: buy count / total count |
| F52 | `obi_z` | OBI z-score (50-tick rolling) |
| F53 | `mom16` | 16-tick log momentum |
| F54 | `mom32` | 32-tick log momentum |
| F55 | `obi_tfi_div` | Rolling OBI / TFI sign disagreement (20-tick) |
| F56 | `book_asym_slope` | Asymmetry slope: F37 − F36 |

**Targets** (three simultaneous log-return horizons):

| Name | Horizon | Description |
|---|---|---|
| `y_1` | 100 ms | 1-tick ahead log mid-price return |
| `y_5` | 500 ms | 5-tick ahead |
| `y_10` | 1 s | 10-tick ahead |

---

## Results

### Regression — Best Models per Architecture

| Architecture | Experiment | Val MAE | Val IC | Val DA |
|---|---|---|---|---|
| **NS-Transformer** | `nst_small_h1_bs64` | 4.91e-6 | — | — |
| **PatchTST** | `ptst_small_h1_bs64_dp05` | 5.35e-6 | — | — |
| **TimesNet** | `timesnet_med_h10_lr5e4` | 4.36e-5 | — | — |

> MAE is on the 1-step (100 ms) regression target unless otherwise noted.
> Lower MAE = better. IC and DA values are in `results/_global_best/*.json`.

### Key findings

- **NS-Transformer** achieves the lowest Val MAE on the h1 target, benefiting
  from its de-stationary attention which explicitly models changing mean/variance.
- **PatchTST** is competitive and more stable across hyperparameter settings due
  to channel independence isolating per-feature patterns.
- **TimesNet** shows stronger signal on the h10 (1 s) target, where periodic
  structure detected by FFT is more pronounced.
- Dropout (p = 0.05–0.1) and moderate batch sizes (64–128) improve generalization
  across all architectures.
- **net_IC** (spread-adjusted IC) is the key real-world viability metric:
  values > 0.1 indicate the signal survives round-trip spread costs.

---

## Setup

```bash
pip install torch torchvision numpy pandas pyarrow scikit-learn pyyaml matplotlib
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are recommended.

---

## How to Run

### Step 1 — Feature Engineering

Requires raw data in `../data-raw/orderbook-data/` and `../data-raw/tradebook-data/`.
Expected filenames: `BTC-USDT_batch_<N>.parquet`.

```bash
python 01_feature_engineering.py [--workers 16] [--output-dir ../data-processed]
```

Outputs: `../data-processed/{train,val,test}.parquet` + `feature_meta.json`.

### Step 2 — Training

```bash
# Single GPU / CPU
python 02_train.py --config configs/patchtst_small.yaml

# Multi-GPU (4 GPUs via torchrun)
torchrun --nproc_per_node=4 02_train.py --config configs/nstransformer_small.yaml
```

Training outputs are written to `results/<experiment_name>/`:
- `checkpoint_best.pt` — best checkpoint by val MAE (regression) or val DA (classification)
- `checkpoint_last.pt` — latest checkpoint (used for automatic resume)
- `results.json` — final val + test metrics + full training log
- `progress.json` — updated after every epoch (for live monitoring)

The best checkpoint across all runs of a given architecture is automatically
copied to `results/_global_best/<arch>_reg_best.pt`.

### Step 3 — Backtest

```bash
# Single model, horizon 1 (100 ms), fast approximate run (stride=64)
python backtest.py \
    --checkpoint results/_global_best/nstransformer_reg_best.pt \
    --data-dir ../data-processed \
    --horizon 1 --stride 64

# Full coverage inference (stride=1), GPU recommended
python backtest.py \
    --checkpoint results/_global_best/patchtst_reg_best.pt \
    --data-dir ../data-processed \
    --horizon 10 --stride 1 --device cuda:0

# Run all horizons for all three regression models + comparison plots
python backtest.py --all-three --data-dir ../data-processed --stride 64
```

Outputs: `results/backtest/<experiment>_h<N>_backtest.{json,png}`.

### Step 4 — Plot Results Dashboard

```bash
python plot_results.py
# Output: results/results_summary.png
```

---

## Config Reference

All config fields and their defaults are in `DEFAULT_CONFIG` inside `02_train.py`.
Key fields:

| Field | Default | Description |
|---|---|---|
| `model` | `patchtst` | `patchtst` / `nstransformer` / `timesnet` |
| `task_type` | `regression` | `regression` or `classification` |
| `n_classes` | `1` | 1 for regression; 3 for Down/Neutral/Up classification |
| `target_idx` | `0` | 0=y_1 (100ms), 1=y_5 (500ms), 2=y_10 (1s) |
| `seq_len` | `512` | Lookback window in ticks (512 = 51.2 s) |
| `d_model` | `256` | Transformer hidden dimension |
| `n_heads` | `8` | Attention heads |
| `n_layers` | `6` | Number of Transformer blocks |
| `batch_size` | `256` | |
| `lr` | `1e-4` | Max LR for OneCycleLR |
| `patience` | `15` | Early stopping patience (epochs) |
| `stride` | `1` | Dataset window stride (1 = every tick) |
| `wall_time_hours` | `23.5` | Graceful stop before this wall time |

---

## Metrics

| Metric | Applies to | Meaning |
|---|---|---|
| **MAE** | Regression | Mean absolute prediction error |
| **DA** | Both | Directional accuracy on non-zero-return ticks |
| **IC** | Regression | Pearson correlation of predictions vs targets |
| **net_IC** | Regression | IC on spread-adjusted returns (real-world viability) |
| **Sharpe** | Backtest | Per-trade Sharpe ratio at optimal threshold |
| **Hit rate** | Backtest | Fraction of trades with positive net return |
| **Profit factor** | Backtest | Gross wins / gross losses |
| **Max drawdown** | Backtest | Peak-to-trough cumulative P&L loss |

**Classification task** labels:
- `0` = Down   (return < −spread)
- `1` = Neutral (|return| ≤ spread)
- `2` = Up      (return > spread)

---

## Data Splits

| Split | Time range | Purpose |
|---|---|---|
| train | before 2024-12-01 | model training |
| val | 2024-12-01 → 2025-01-01 | hyperparameter selection + early stopping |
| test | after 2025-01-01 | final unbiased evaluation + backtest |
