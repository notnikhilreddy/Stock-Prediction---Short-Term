#!/usr/bin/env python3
"""
Step 1 — Feature Engineering Pipeline
======================================
Reads raw BTC-USDT orderbook + tradebook parquet files and produces a
56-feature matrix at 100 ms resolution, saved as three chronological splits:

    data-processed/
        train.parquet   (up to 2024-12-01)
        val.parquet     (2024-12-01 to 2025-01-01)
        test.parquet    (2025-01-01 onward)
        feature_meta.json

Features (56 total, all float32):
  F01  — mid-price log return (1 tick)
  F02  — bid-ask spread in bps
  F03  — micro-price deviation from mid in bps
  F04  — rolling return volatility (20-tick std)
  F05  — spread regime z-score (100-tick)
  F06-F10 — ask level distances L1-L5 (bps from mid)
  F11-F15 — bid level distances L1-L5 (bps from mid)
  F16-F25 — log ask volume at each of 10 book levels
  F26-F35 — log bid volume at each of 10 book levels
  F36  — bid log-volume slope (linear fit across levels)
  F37  — ask log-volume slope
  F38  — order book imbalance at L1 (OBI L1)
  F39  — OBI near (L1-L5 aggregate)
  F40  — OBI deep (L6-L10 aggregate)
  F41  — weighted OBI (1/level weights)
  F42  — delta OBI (first difference of OBI L1)
  F43  — order flow imbalance bid (OFI bid, normalized)
  F44  — order flow imbalance ask (OFI ask, normalized)
  F45  — cumulative net OFI over 10 ticks
  F46  — trade flow imbalance (buy volume / total volume)
  F47  — log trade volume
  F48  — log trade tick count
  F49  — VWAP deviation from mid in bps
  F50  — micro price volatility (high-low range in bps)
  F51  — aggressor ratio (buy count / total count)
  F52  — OBI z-score (50-tick rolling)
  F53  — 16-tick log momentum
  F54  — 32-tick log momentum
  F55  — OBI / TFI divergence (rolling 20-tick fraction)
  F56  — book asymmetry slope (ask slope − bid slope)

Targets (not in the 56 features; used during training):
  y_1  — 1-tick log return (100 ms horizon)
  y_5  — 5-tick log return (500 ms horizon)
  y_10 — 10-tick log return (1 s horizon)

Usage:
    python 01_feature_engineering.py [--workers N] [--output-dir PATH]

Memory notes:
    The tradebook is parsed fully into RAM before aggregation.
    The orderbook is processed in chunks (--ob-chunk-size files per chunk).
    float32 throughout to keep peak RAM within ~64 GB for the full dataset.
"""

import os
import sys
import json
import glob
import time
import logging
import argparse
import gc
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Resolve absolute paths relative to the repo root (one level up from this script)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_OB_DIR = os.path.join(BASE_DIR, "data-raw", "orderbook-data")
RAW_TR_DIR = os.path.join(BASE_DIR, "data-raw", "tradebook-data")
OUT_DIR    = os.path.join(BASE_DIR, "data-processed")

EPS         = 1e-12   # prevent division-by-zero in all ratio/log computations
LEVELS      = 10      # number of book levels in raw data
DIST_LEVELS = 5       # levels used for distance features F06-F15


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_orderbook_batch(filepath):
    """Parse a single orderbook parquet → flat float32 DataFrame with LEVELS bid/ask columns."""
    log.info(f"Parsing OB: {os.path.basename(filepath)}")
    df = pd.read_parquet(filepath, columns=["bids", "asks", "timestamp"])
    n  = len(df)

    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    bid_prices  = np.zeros((n, LEVELS), dtype=np.float32)
    bid_amounts = np.zeros((n, LEVELS), dtype=np.float32)
    ask_prices  = np.zeros((n, LEVELS), dtype=np.float32)
    ask_amounts = np.zeros((n, LEVELS), dtype=np.float32)

    bids_col = df["bids"].values
    asks_col = df["asks"].values

    for i in range(n):
        bids = bids_col[i]
        asks = asks_col[i]
        for j in range(min(LEVELS, len(bids))):
            bid_prices[i, j]  = bids[j]["price"]
            bid_amounts[i, j] = bids[j]["amount"]
        for j in range(min(LEVELS, len(asks))):
            ask_prices[i, j]  = asks[j]["price"]
            ask_amounts[i, j] = asks[j]["amount"]

    cols = {"timestamp": ts.values}
    for j in range(LEVELS):
        cols[f"bp{j+1}"] = bid_prices[:, j]
        cols[f"ba{j+1}"] = bid_amounts[:, j]
        cols[f"ap{j+1}"] = ask_prices[:, j]
        cols[f"aa{j+1}"] = ask_amounts[:, j]

    return pd.DataFrame(cols)


def parse_tradebook_batch(filepath):
    """Parse a single tradebook parquet → price/size/side/timestamp/count columns."""
    log.info(f"Parsing TR: {os.path.basename(filepath)}")
    df = pd.read_parquet(filepath, columns=["price", "size", "side", "timestamp", "count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["price"]     = df["price"].astype(np.float32)
    df["size"]      = df["size"].astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Parallel chunk helpers
# ---------------------------------------------------------------------------

def chunk_list(lst, n_chunks):
    """Split lst into n_chunks roughly-equal sub-lists."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def process_ob_chunk(file_list):
    """Parse and concat a list of OB parquet files (worker function for Pool.map)."""
    frames = []
    for fp in file_list:
        try:
            frames.append(parse_orderbook_batch(fp))
        except Exception as e:
            log.error(f"OB parse error {fp}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def process_tr_chunk(file_list):
    """Parse and concat a list of tradebook parquet files (worker function for Pool.map)."""
    frames = []
    for fp in file_list:
        try:
            frames.append(parse_tradebook_batch(fp))
        except Exception as e:
            log.error(f"TR parse error {fp}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Trade aggregation
# ---------------------------------------------------------------------------

def aggregate_trades_100ms(trades_df):
    """
    Aggregate raw tick-by-tick trades into 100 ms windows.

    For each 100 ms bucket, computes:
        v_buy/v_sell    — buy/sell volume
        count_buy/sell  — buy/sell trade count
        total_count     — total trades
        pv_sum          — price × volume sum (for VWAP)
        total_size      — total volume
        p_high/p_low    — price range within bucket
    """
    log.info("Aggregating trades to 100ms windows...")
    trades_df         = trades_df.sort_values("timestamp")
    trades_df["window"] = trades_df["timestamp"].dt.floor("100ms")
    if trades_df["window"].dt.tz is None:
        trades_df["window"] = trades_df["window"].dt.tz_localize("UTC")

    buy              = (trades_df["side"] == "buy")
    trades_df["bv"]  = trades_df["size"] * buy
    trades_df["sv"]  = trades_df["size"] * (~buy)
    trades_df["bc"]  = buy.astype(np.int32)
    trades_df["sc"]  = (~buy).astype(np.int32)
    trades_df["pv"]  = trades_df["price"] * trades_df["size"]

    agg = trades_df.groupby("window", sort=True).agg(
        v_buy       = ("bv",   "sum"),
        v_sell      = ("sv",   "sum"),
        count_buy   = ("bc",   "sum"),
        count_sell  = ("sc",   "sum"),
        total_count = ("size", "count"),
        pv_sum      = ("pv",   "sum"),
        total_size  = ("size", "sum"),
        p_high      = ("price","max"),
        p_low       = ("price","min"),
    ).reset_index().rename(columns={"window": "timestamp"})
    return agg


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features_chunked(ob_chunk, trade_agg, prev_tail=None):
    """
    Compute all 56 features + 3 targets for one chunk of orderbook data.

    Args:
        ob_chunk:   DataFrame with flattened OB columns + timestamp (100 ms resolution)
        trade_agg:  pre-aggregated trade stats for the same time range (may be None)
        prev_tail:  last N rows from the previous chunk used for rolling window warm-up

    Returns:
        DataFrame with F01–F56 features + y_1, y_5, y_10 targets.
        Target rows that fall at the end of the chunk (where look-ahead is unavailable)
        are set to NaN and dropped by the caller.
    """
    ob = ob_chunk if not isinstance(ob_chunk, str) else pd.read_parquet(ob_chunk)

    # Ensure UTC timezone on both sides before merge
    if ob["timestamp"].dt.tz is None:
        ob["timestamp"] = ob["timestamp"].dt.tz_localize("UTC")

    if trade_agg is not None and len(trade_agg) > 0:
        if trade_agg["timestamp"].dt.tz is None:
            trade_agg = trade_agg.copy()
            trade_agg["timestamp"] = trade_agg["timestamp"].dt.tz_localize("UTC")
        ob = ob.merge(trade_agg, on="timestamp", how="left")

    # Fill missing trade columns (ticks with no trades in that 100 ms window)
    for c in ["v_buy", "v_sell", "count_buy", "count_sell",
              "total_count", "pv_sum", "total_size", "p_high", "p_low"]:
        if c not in ob.columns:
            ob[c] = np.float32(0)
    trade_cols = ["v_buy", "v_sell", "count_buy", "count_sell",
                  "total_count", "pv_sum", "total_size"]
    ob[trade_cols] = ob[trade_cols].fillna(0)

    # Prepend previous tail for rolling window warm-up across chunk boundaries
    overlap = 0
    if prev_tail is not None and len(prev_tail) > 0:
        overlap = len(prev_tail)
        ob = pd.concat([prev_tail, ob], ignore_index=True)

    n   = len(ob)
    mid = ((ob["ap1"].values + ob["bp1"].values) / 2).astype(np.float64)
    v_bid_1 = ob["ba1"].values.astype(np.float64)
    v_ask_1 = ob["aa1"].values.astype(np.float64)

    feat = {}
    feat["timestamp"] = ob["timestamp"].values

    # ── Price / spread features ──────────────────────────────────────────────

    R = np.zeros(n, dtype=np.float32)
    R[1:] = np.log(mid[1:] / (mid[:-1] + EPS)).astype(np.float32)
    feat["F01_log_return"] = R

    feat["F02_spread_bps"] = (
        (ob["ap1"].values - ob["bp1"].values) / (mid + EPS) * 10000
    ).astype(np.float32)

    # Micro-price: L1 mid weighted by opposing side's volume
    p_micro = (ob["bp1"].values * v_ask_1 + ob["ap1"].values * v_bid_1) / (v_bid_1 + v_ask_1 + EPS)
    feat["F03_micro_dev_bps"] = ((p_micro - mid) / (mid + EPS) * 10000).astype(np.float32)

    R_s = pd.Series(R)
    feat["F04_rvol_20"] = R_s.rolling(20, min_periods=1).std().values.astype(np.float32)

    sp    = pd.Series(feat["F02_spread_bps"])
    mu_s  = sp.rolling(100, min_periods=1).mean().values
    sig_s = sp.rolling(100, min_periods=1).std().values
    feat["F05_spread_z"] = ((feat["F02_spread_bps"] - mu_s) / (sig_s + EPS)).astype(np.float32)

    # ── Book level distances (bps from mid) ──────────────────────────────────

    for i in range(1, DIST_LEVELS + 1):
        feat[f"F{5+i:02d}_d_ask_{i}"] = (
            (ob[f"ap{i}"].values - mid) / (mid + EPS) * 10000
        ).astype(np.float32)

    for i in range(1, DIST_LEVELS + 1):
        feat[f"F{10+i:02d}_d_bid_{i}"] = (
            (mid - ob[f"bp{i}"].values) / (mid + EPS) * 10000
        ).astype(np.float32)

    # ── Log volumes at each book level ───────────────────────────────────────

    for i in range(1, LEVELS + 1):
        feat[f"F{15+i:02d}_v_ask_{i}"] = np.log(ob[f"aa{i}"].values + 1).astype(np.float32)

    for i in range(1, LEVELS + 1):
        feat[f"F{25+i:02d}_v_bid_{i}"] = np.log(ob[f"ba{i}"].values + 1).astype(np.float32)

    # ── Book shape (volume slopes) ────────────────────────────────────────────

    x     = np.arange(1, LEVELS + 1, dtype=np.float64)
    x_m   = x.mean()
    var_x = np.var(x)

    bid_lv  = np.column_stack([np.log(ob[f"ba{i}"].values + 1) for i in range(1, LEVELS + 1)])
    bid_ym  = bid_lv.mean(axis=1)
    bid_cov = ((bid_lv - bid_ym[:, None]) * (x - x_m)).mean(axis=1)
    feat["F36_bid_slope"] = (bid_cov / (var_x + EPS)).astype(np.float32)

    ask_lv  = np.column_stack([np.log(ob[f"aa{i}"].values + 1) for i in range(1, LEVELS + 1)])
    ask_ym  = ask_lv.mean(axis=1)
    ask_cov = ((ask_lv - ask_ym[:, None]) * (x - x_m)).mean(axis=1)
    feat["F37_ask_slope"] = (ask_cov / (var_x + EPS)).astype(np.float32)

    # ── Order book imbalance (OBI) ────────────────────────────────────────────

    bid_vols = np.column_stack([ob[f"ba{i}"].values for i in range(1, LEVELS + 1)])
    ask_vols = np.column_stack([ob[f"aa{i}"].values for i in range(1, LEVELS + 1)])

    feat["F38_obi_l1"] = (
        (v_bid_1 - v_ask_1) / (v_bid_1 + v_ask_1 + EPS)
    ).astype(np.float32)

    bn = bid_vols[:, :5].sum(axis=1)
    an = ask_vols[:, :5].sum(axis=1)
    feat["F39_obi_near"] = ((bn - an) / (bn + an + EPS)).astype(np.float32)

    bd = bid_vols[:, 5:10].sum(axis=1)
    ad = ask_vols[:, 5:10].sum(axis=1)
    feat["F40_obi_deep"] = ((bd - ad) / (bd + ad + EPS)).astype(np.float32)

    w  = 1.0 / np.arange(1, LEVELS + 1, dtype=np.float64)
    wb = (bid_vols * w).sum(axis=1)
    wa = (ask_vols * w).sum(axis=1)
    feat["F41_wobi"] = ((wb - wa) / (wb + wa + EPS)).astype(np.float32)

    obi_l1        = feat["F38_obi_l1"]
    dobi          = np.zeros(n, dtype=np.float32)
    dobi[1:]      = obi_l1[1:] - obi_l1[:-1]
    feat["F42_dobi"] = dobi

    # ── Order flow imbalance (OFI) ────────────────────────────────────────────

    bp1     = ob["bp1"].values.astype(np.float64)
    ap1     = ob["ap1"].values.astype(np.float64)
    top_vol = v_bid_1 + v_ask_1 + EPS

    bp1_prev  = np.roll(bp1, 1)
    ap1_prev  = np.roll(ap1, 1)
    vb1_prev  = np.roll(v_bid_1, 1)
    va1_prev  = np.roll(v_ask_1, 1)

    # Bid OFI: +v_bid if price improved, -v_bid_prev if price deteriorated, else diff
    raw_ofi_bid = np.where(bp1 > bp1_prev,  v_bid_1,
                  np.where(bp1 < bp1_prev, -vb1_prev, v_bid_1 - vb1_prev))
    ofi_bid     = (raw_ofi_bid / top_vol).astype(np.float32)
    ofi_bid[0]  = 0
    feat["F43_ofi_bid"] = ofi_bid

    raw_ofi_ask = np.where(ap1 < ap1_prev,  v_ask_1,
                  np.where(ap1 > ap1_prev, -va1_prev, v_ask_1 - va1_prev))
    ofi_ask     = (raw_ofi_ask / top_vol).astype(np.float32)
    ofi_ask[0]  = 0
    feat["F44_ofi_ask"] = ofi_ask

    net_ofi         = ofi_bid - ofi_ask
    feat["F45_cofi_10"] = pd.Series(net_ofi).rolling(10, min_periods=1).sum().values.astype(np.float32)

    # ── Trade features ────────────────────────────────────────────────────────

    vb    = ob["v_buy"].values.astype(np.float64)
    vs    = ob["v_sell"].values.astype(np.float64)
    cb    = ob["count_buy"].values.astype(np.float64)
    cs    = ob["count_sell"].values.astype(np.float64)
    ts_v  = ob["total_size"].values.astype(np.float64)
    pv    = ob["pv_sum"].values.astype(np.float64)
    tc    = ob["total_count"].values.astype(np.float64)

    ph_s = pd.Series(ob["p_high"].values.copy().astype(np.float64)).ffill().fillna(mid[0]).values
    pl_s = pd.Series(ob["p_low"].values.copy().astype(np.float64)).ffill().fillna(mid[0]).values

    feat["F46_tfi"]          = ((vb - vs) / (vb + vs + EPS)).astype(np.float32)
    feat["F47_log_vol"]      = np.log(vb + vs + 1).astype(np.float32)
    feat["F48_log_ticks"]    = np.log(tc + 1).astype(np.float32)
    vwap = np.where(ts_v > 0, pv / ts_v, mid)
    feat["F49_vwap_dev_bps"] = ((vwap - mid) / (mid + EPS) * 10000).astype(np.float32)
    feat["F50_micro_vol_bps"]= np.where(tc > 1, (ph_s - pl_s) / (mid + EPS) * 10000, 0.0).astype(np.float32)
    feat["F51_aggr_ratio"]   = (cb / (cb + cs + EPS)).astype(np.float32)

    # ── Cross-feature / composite ─────────────────────────────────────────────

    obi_s = pd.Series(obi_l1.astype(np.float64))
    mu_o  = obi_s.rolling(50, min_periods=1).mean().values
    sig_o = obi_s.rolling(50, min_periods=1).std().values
    feat["F52_obi_z"] = ((obi_l1 - mu_o) / (sig_o + EPS)).astype(np.float32)

    mom16        = np.zeros(n, dtype=np.float32)
    mom16[16:]   = np.log(mid[16:] / (mid[:-16] + EPS)).astype(np.float32)
    feat["F53_mom16"] = mom16

    mom32        = np.zeros(n, dtype=np.float32)
    mom32[32:]   = np.log(mid[32:] / (mid[:-32] + EPS)).astype(np.float32)
    feat["F54_mom32"] = mom32

    # OBI/TFI divergence: fraction of last 20 ticks where book and trade signals disagree
    sign_dis = (np.sign(obi_l1) != np.sign(feat["F46_tfi"])).astype(np.float32)
    feat["F55_obi_tfi_div"] = pd.Series(sign_dis).rolling(20, min_periods=1).mean().values.astype(np.float32)

    feat["F56_book_asym_slope"] = (feat["F37_ask_slope"] - feat["F36_bid_slope"]).astype(np.float32)

    # ── Targets (future log returns) ─────────────────────────────────────────

    # y_1: next tick return (1-step ahead)
    feat["y_1"] = np.roll(R, -1).astype(np.float32)

    y5       = np.zeros(n, dtype=np.float32)
    y5[:-5]  = np.log(mid[5:]  / (mid[:-5]  + EPS)).astype(np.float32)
    feat["y_5"] = y5

    y10       = np.zeros(n, dtype=np.float32)
    y10[:-10] = np.log(mid[10:] / (mid[:-10] + EPS)).astype(np.float32)
    feat["y_10"] = y10

    result = pd.DataFrame(feat)

    # Strip warm-up rows from prev_tail
    if overlap > 0:
        result = result.iloc[overlap:].reset_index(drop=True)

    # Mark look-ahead targets as NaN at chunk tail (will be dropped globally)
    rlen = len(result)
    result.iloc[rlen - 1:rlen, result.columns.get_loc("y_1")]  = np.nan
    if rlen >= 5:
        result.iloc[rlen - 5:rlen,  result.columns.get_loc("y_5")]  = np.nan
    if rlen >= 10:
        result.iloc[rlen - 10:rlen, result.columns.get_loc("y_10")] = np.nan

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BTC-USDT Feature Engineering — builds train/val/test parquets"
    )
    parser.add_argument("--workers",       type=int, default=min(32, cpu_count()),
                        help="Parallel workers for parsing (default: min(32, CPU count))")
    parser.add_argument("--output-dir",    type=str, default=OUT_DIR,
                        help="Output directory for processed parquets and metadata")
    parser.add_argument("--ob-chunk-size", type=int, default=100,
                        help="Number of OB parquet files per processing mega-chunk")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Discover all parquet files, sorted by batch index
    ob_files = sorted(
        glob.glob(os.path.join(RAW_OB_DIR, "BTC-USDT_batch_*.parquet")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
    )
    tr_files = sorted(
        glob.glob(os.path.join(RAW_TR_DIR, "BTC-USDT_batch_*.parquet")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
    )
    log.info(f"Found {len(ob_files)} OB files, {len(tr_files)} TR files")

    # Parse entire tradebook in parallel (relatively small; fits in RAM)
    log.info(f"Parsing tradebook with {args.workers} workers...")
    tr_chunks  = chunk_list(tr_files, args.workers)
    with Pool(args.workers) as pool:
        tr_results = pool.map(process_tr_chunk, tr_chunks)
    trades_df  = pd.concat([r for r in tr_results if len(r) > 0], ignore_index=True)
    del tr_results
    gc.collect()
    log.info(f"Tradebook total rows: {len(trades_df)}")

    # Aggregate all trades to 100 ms windows (needed to align with OB timestamps)
    trade_agg = aggregate_trades_100ms(trades_df)
    del trades_df
    gc.collect()
    log.info(f"Trade aggregates: {len(trade_agg)} 100ms windows")

    # Process OB in mega-chunks to control peak memory
    mega_chunks = [ob_files[i:i + args.ob_chunk_size]
                   for i in range(0, len(ob_files), args.ob_chunk_size)]
    log.info(f"Processing {len(mega_chunks)} OB mega-chunks (~{args.ob_chunk_size} files each)")

    all_features = []
    for ci, chunk_files in enumerate(mega_chunks):
        log.info(f"=== Mega-chunk {ci+1}/{len(mega_chunks)} ({len(chunk_files)} files) ===")

        # Parse this chunk of OB files in parallel sub-chunks
        sub_chunks = chunk_list(chunk_files, min(args.workers, len(chunk_files)))
        with Pool(min(args.workers, len(chunk_files))) as pool:
            ob_results = pool.map(process_ob_chunk, sub_chunks)

        ob_chunk = pd.concat([r for r in ob_results if len(r) > 0], ignore_index=True)
        del ob_results
        gc.collect()

        # Sort, floor to 100 ms, deduplicate (keep most recent snapshot per window)
        ob_chunk.sort_values("timestamp", inplace=True)
        ob_chunk["timestamp"] = ob_chunk["timestamp"].dt.floor("100ms")
        if ob_chunk["timestamp"].dt.tz is None:
            ob_chunk["timestamp"] = ob_chunk["timestamp"].dt.tz_localize("UTC")
        ob_chunk.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        ob_chunk.reset_index(drop=True, inplace=True)

        log.info(f"  OB chunk: {len(ob_chunk)} rows, "
                 f"{ob_chunk['timestamp'].iloc[0]} → {ob_chunk['timestamp'].iloc[-1]}")

        # Restrict trade aggregates to this chunk's time range (with small padding)
        if trade_agg["timestamp"].dt.tz is None:
            trade_agg["timestamp"] = trade_agg["timestamp"].dt.tz_localize("UTC")
        t_start  = ob_chunk["timestamp"].iloc[0]
        t_end    = ob_chunk["timestamp"].iloc[-1]
        padding  = pd.Timedelta("10s")
        ta_chunk = trade_agg[
            (trade_agg["timestamp"] >= t_start - padding) &
            (trade_agg["timestamp"] <= t_end   + padding)
        ].copy()

        feat_chunk = compute_features_chunked(ob_chunk, ta_chunk)
        del ob_chunk, ta_chunk
        gc.collect()

        log.info(f"  Feature chunk shape: {feat_chunk.shape}")
        all_features.append(feat_chunk)
        del feat_chunk
        gc.collect()

    # Concatenate everything
    log.info("Concatenating all feature chunks...")
    feat_df = pd.concat(all_features, ignore_index=True)
    del all_features
    gc.collect()
    log.info(f"Total feature matrix: {feat_df.shape}")

    # Drop warm-up rows and NaN targets
    feat_df = feat_df.iloc[100:].reset_index(drop=True)
    feat_df.dropna(subset=["y_1", "y_5", "y_10"], inplace=True)
    feat_df.reset_index(drop=True, inplace=True)
    log.info(f"After warmup + NaN drop: {feat_df.shape}")

    # Fill any remaining NaN in features with 0
    feature_cols = [c for c in feat_df.columns if c.startswith("F")]
    for c in feature_cols:
        if feat_df[c].isna().any():
            feat_df[c] = feat_df[c].fillna(0.0)

    # Write per-feature statistics (used for debugging and normalization reference)
    meta = {
        c: {
            "mean": float(feat_df[c].mean()),
            "std":  float(feat_df[c].std()),
            "min":  float(feat_df[c].min()),
            "max":  float(feat_df[c].max()),
        }
        for c in feature_cols
    }
    with open(os.path.join(args.output_dir, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved feature_meta.json")

    # Chronological split: train / val / test
    ts = feat_df["timestamp"].values
    ts_int    = ts.astype(np.int64)
    t_dec_int = pd.Timestamp("2024-12-01", tz="UTC").value
    t_jan_int = pd.Timestamp("2025-01-01", tz="UTC").value

    idx_dec = int(np.searchsorted(ts_int, t_dec_int))
    idx_jan = int(np.searchsorted(ts_int, t_jan_int))
    log.info(f"Split: train 0:{idx_dec} | val {idx_dec}:{idx_jan} | test {idx_jan}:{len(feat_df)}")

    save_cols = [c for c in feat_df.columns if c != "timestamp"]
    for name, start, end in [("train", 0, idx_dec), ("val", idx_dec, idx_jan), ("test", idx_jan, len(feat_df))]:
        path  = os.path.join(args.output_dir, f"{name}.parquet")
        chunk = feat_df.iloc[start:end][save_cols]
        chunk.to_parquet(path, index=False, engine="pyarrow")
        log.info(f"  Saved {name}: {end - start:,} rows → {path}")
        del chunk
        gc.collect()

    log.info(f"Pipeline complete in {time.time() - t0:.1f}s | total rows: {len(feat_df):,}")


if __name__ == "__main__":
    main()
