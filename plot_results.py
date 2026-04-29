#!/usr/bin/env python3
"""
Step 4 — Results Visualization
================================
Scans all experiment subdirectories under results/, loads their results.json
or progress.json files, and produces a comprehensive 11-panel summary figure
saved to results/results_summary.png.

Panels:
  1. Val MAE bar chart (top 25 experiments, lower is better)
  2. IC horizontal bar chart (top 18, higher is better)
  3. NST-Small learning curves: Val MAE
  4. NST-Small learning curves: IC
  5. PatchTST-Small learning curves vs. best NST reference
  6. MAE vs DA scatter by horizon (h1 / h5 / h10)
  7. Hyperparameter sweep bar chart (NST Med h1)
  8. Architecture MAE range plot (min/median/max per arch)
  9. Net IC learning curves (spread-adjusted, real-world viability)
  10. Top-15 results table with color-coded metrics
  11. LR / Dropout sweep IC vs MAE scatter

Usage:
    python plot_results.py
    # Output: results/results_summary.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "results")
SKIP = {
    "_global_best", "nstransformer_deep", "nstransformer_large",
    "nstransformer_longctx", "nstransformer_medium", "nstransformer_small",
    "patchtst_large", "patchtst_longctx", "patchtst_medium",
    "patchtst_patch32", "patchtst_small",
}

# ── load all experiments ──────────────────────────────────────────────────────
def load_all():
    data = {}
    for name in sorted(os.listdir(BASE)):
        if name in SKIP:
            continue
        d = os.path.join(BASE, name)
        for fname in ("results.json", "progress.json"):
            fp = os.path.join(d, fname)
            if os.path.exists(fp):
                data[name] = json.load(open(fp))
                break
    return data

def get_history(p):
    """Return list of (epoch, mae, da, ic, net_ic) from progress.json or results.json."""
    log = p.get("train_log", [])
    return [
        (
            e["epoch"],
            e["val_mae"],
            e["val_da"],
            e["val_ic"],
            e.get("val_net_ic", float("nan")),
        )
        for e in log
        if "val_mae" in e
    ]

def last_metrics(p):
    hist = get_history(p)
    if not hist:
        return None
    ep, mae, da, ic, net_ic = hist[-1]
    best_mae = p.get("best_val_mae", mae)
    return {"epoch": ep, "mae": best_mae, "last_mae": mae,
            "da": da, "ic": ic, "net_ic": net_ic}

# ── colour helpers ────────────────────────────────────────────────────────────
PALETTE = {
    "nst_small":  "#00b4d8",
    "nst_med":    "#0077b6",
    "nst_large":  "#023e8a",
    "ptst_small": "#e63946",
    "ptst_med":   "#c1121f",
    "ptst_large": "#6d0b0b",
    "ptst_p32":   "#ff6b35",
}

def color_for(name):
    for key, col in PALETTE.items():
        if name.startswith(key):
            return col
    return "#888888"

def arch_label(name):
    if name.startswith("nst_small"):  return "NST-Small"
    if name.startswith("nst_med"):    return "NST-Med"
    if name.startswith("nst_large"):  return "NST-Large"
    if name.startswith("ptst_small"): return "PTST-Small"
    if name.startswith("ptst_med"):   return "PTST-Med"
    if name.startswith("ptst_p32"):   return "PTST-p32"
    if name.startswith("ptst_large"): return "PTST-Large"
    return "Other"

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    all_data = load_all()
    out_path = os.path.join(BASE, "results_summary.png")

    # build summary rows
    rows = []
    for name, p in all_data.items():
        m = last_metrics(p)
        if m is None:
            continue
        rows.append({
            "name": name,
            "arch": arch_label(name),
            "color": color_for(name),
            "epoch": m["epoch"],
            "mae": m["mae"],
            "last_mae": m["last_mae"],
            "da": m["da"],
            "ic": m["ic"],
            "net_ic": m.get("net_ic", float("nan")),
            "hist": get_history(p),
        })

    rows.sort(key=lambda r: r["mae"])

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 34), facecolor="#0d1117")
    gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.95, bottom=0.03, left=0.06, right=0.97)

    txt_kw = dict(color="#e6edf3", fontfamily="monospace")
    title_kw = dict(color="#58a6ff", fontsize=11, fontweight="bold", pad=8)

    fig.text(0.5, 0.975, "BTC-USDT HFT — Experiment Results Dashboard",
             ha="center", va="top", fontsize=18, fontweight="bold",
             color="#58a6ff", fontfamily="monospace")
    best_row = rows[0] if rows else {}
    best_label = (f"Best: {best_row.get('name','?')}  "
                  f"MAE={best_row.get('mae',0):.2e}  "
                  f"IC={best_row.get('ic',0):.3f}  "
                  f"net_IC={best_row.get('net_ic', float('nan')):.3f}")
    fig.text(0.5, 0.963, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M EDT')}   |   "
             f"{len(rows)} experiments with data   |   {best_label}",
             ha="center", va="top", fontsize=9.5, color="#8b949e", fontfamily="monospace")

    ax_bg = lambda ax: ax.set_facecolor("#161b22")

    # ── Panel 1: Val MAE ranking (bar chart, top 25) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax_bg(ax1)
    top = rows[:25]
    names = [r["name"] for r in top]
    maes  = [r["mae"]  for r in top]
    cols  = [r["color"] for r in top]
    xs = np.arange(len(names))
    bars = ax1.bar(xs, maes, color=cols, edgecolor="#30363d", linewidth=0.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5, **txt_kw)
    ax1.set_ylabel("Val MAE (best)", **txt_kw, fontsize=9)
    ax1.set_title("Val MAE — Top 25 Experiments (lower is better)", **title_kw)
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")
    # annotate top 5
    for i, (bar, r) in enumerate(zip(bars[:5], top[:5])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                 f"{r['mae']:.2e}", ha="center", va="bottom",
                 fontsize=6, color="#e6edf3", fontfamily="monospace")
    ax1.set_facecolor("#161b22")
    ax1.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax1.set_axisbelow(True)

    # ── Panel 2: IC ranking (top 20) ──────────────────────────────────────────
    rows_by_ic = sorted(rows, key=lambda r: r["ic"], reverse=True)
    ax2 = fig.add_subplot(gs[0, 2])
    ax_bg(ax2)
    top_ic = rows_by_ic[:18]
    names_ic = [r["name"] for r in top_ic]
    ics = [r["ic"] for r in top_ic]
    cols_ic = [r["color"] for r in top_ic]
    ax2.barh(range(len(names_ic)), ics, color=cols_ic, edgecolor="#30363d", linewidth=0.5)
    ax2.set_yticks(range(len(names_ic)))
    ax2.set_yticklabels(names_ic, fontsize=6.5, **txt_kw)
    ax2.set_xlabel("Val IC (higher is better)", **txt_kw, fontsize=9)
    ax2.set_title("IC Ranking — Top 18", **title_kw)
    ax2.axvline(0, color="#30363d", linewidth=1)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    ax2.xaxis.grid(True, color="#21262d", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()

    # ── Panel 3: Learning curves — NST Small variants ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax_bg(ax3)
    nst_small_names = [r for r in rows if r["name"].startswith("nst_small")]
    for r in nst_small_names:
        if not r["hist"]: continue
        eps = [h[0] for h in r["hist"]]
        maes = [h[1] for h in r["hist"]]
        label = r["name"].replace("nst_small_", "")
        ax3.semilogy(eps, maes, marker="o", markersize=3, linewidth=1.5,
                     label=label)
    ax3.set_xlabel("Epoch", **txt_kw, fontsize=9)
    ax3.set_ylabel("Val MAE (log)", **txt_kw, fontsize=9)
    ax3.set_title("NST-Small: Val MAE Learning Curves", **title_kw)
    ax3.legend(fontsize=6, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper right")
    ax3.tick_params(colors="#8b949e")
    ax3.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax3.xaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 4: IC learning curves — NST Small ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax_bg(ax4)
    for r in nst_small_names:
        if not r["hist"]: continue
        eps = [h[0] for h in r["hist"]]
        ics = [h[3] for h in r["hist"]]
        label = r["name"].replace("nst_small_", "")
        ax4.plot(eps, ics, marker="o", markersize=3, linewidth=1.5, label=label)
    ax4.axhline(0, color="#30363d", linewidth=1, linestyle="--")
    ax4.set_xlabel("Epoch", **txt_kw, fontsize=9)
    ax4.set_ylabel("Val IC", **txt_kw, fontsize=9)
    ax4.set_title("NST-Small: IC Learning Curves", **title_kw)
    ax4.legend(fontsize=6, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="lower right")
    ax4.tick_params(colors="#8b949e")
    ax4.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax4.xaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 5: PatchTST small learning curves ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax_bg(ax5)
    ptst_small_names = [r for r in rows if r["name"].startswith("ptst_small") or r["name"].startswith("ptst_p32")]
    for r in ptst_small_names:
        if not r["hist"]: continue
        eps = [h[0] for h in r["hist"]]
        maes = [h[1] for h in r["hist"]]
        label = r["name"].replace("ptst_", "")
        ax5.semilogy(eps, maes, marker="s", markersize=3, linewidth=1.5, label=label)
    # overlay best NST for comparison
    best_nst = next((r for r in rows if r["name"] == "nst_small_h1_bs64"), None)
    if best_nst and best_nst["hist"]:
        eps = [h[0] for h in best_nst["hist"]]
        maes = [h[1] for h in best_nst["hist"]]
        ax5.semilogy(eps, maes, "--", color="#00ff88", linewidth=2,
                     label="nst_bs64 (ref)", zorder=10)
    ax5.set_xlabel("Epoch", **txt_kw, fontsize=9)
    ax5.set_ylabel("Val MAE (log)", **txt_kw, fontsize=9)
    ax5.set_title("PatchTST Small/p32: MAE + NST ref", **title_kw)
    ax5.legend(fontsize=6, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3")
    ax5.tick_params(colors="#8b949e")
    ax5.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax5.xaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 6: Horizon comparison (h1/h5/h10) scatter ───────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax_bg(ax6)
    horizon_groups = {"h1": [], "h5": [], "h10": []}
    for r in rows:
        n = r["name"]
        for h in ("h1", "h5", "h10"):
            if n.endswith(h) or f"_{h}_" in n or n.endswith(f"_{h}"):
                horizon_groups[h].append(r)
                break
    hcolors = {"h1": "#ff6b6b", "h5": "#ffd166", "h10": "#06d6a0"}
    for h, grp in horizon_groups.items():
        if not grp: continue
        maes = [r["mae"] for r in grp]
        das  = [r["da"]  for r in grp]
        ax6.scatter(das, maes, c=hcolors[h], label=h, s=60, alpha=0.85,
                    edgecolors="#30363d", linewidths=0.5)
        # label each point
        for r in grp:
            short = r["name"].replace("nst_","N").replace("ptst_","P").replace("_small","s").replace("_med","m").replace("_large","L")
            ax6.annotate(short, (r["da"], r["mae"]), fontsize=4.5,
                         color="#8b949e", textcoords="offset points", xytext=(3,2))
    ax6.set_yscale("log")
    ax6.set_xlabel("Directional Accuracy", **txt_kw, fontsize=9)
    ax6.set_ylabel("Val MAE (log)", **txt_kw, fontsize=9)
    ax6.set_title("MAE vs DA by Horizon (h1/h5/h10)", **title_kw)
    ax6.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax6.tick_params(colors="#8b949e")
    ax6.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax6.xaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax6.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 7: Hyperparameter sweep — NST medium h1 ─────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    ax_bg(ax7)
    sweep_names = [r for r in rows if "nst_med_h1" in r["name"]]
    sweep_names_s = sorted(sweep_names, key=lambda r: r["mae"])
    snames = [r["name"].replace("nst_med_h1","").lstrip("_") or "baseline" for r in sweep_names_s]
    smaes  = [r["mae"] for r in sweep_names_s]
    sics   = [r["ic"]  for r in sweep_names_s]
    x = np.arange(len(snames))
    ax7b = ax7.twinx()
    ax7.bar(x - 0.2, smaes, width=0.35, color="#0077b6", alpha=0.85,
            edgecolor="#30363d", linewidth=0.5, label="Val MAE")
    ax7b.bar(x + 0.2, sics, width=0.35, color="#90e0ef", alpha=0.85,
             edgecolor="#30363d", linewidth=0.5, label="IC")
    ax7.set_xticks(x)
    ax7.set_xticklabels(snames, rotation=45, ha="right", fontsize=6.5, **txt_kw)
    ax7.set_ylabel("Val MAE", color="#0077b6", fontsize=9)
    ax7b.set_ylabel("IC", color="#90e0ef", fontsize=9)
    ax7.set_title("NST-Med h1 Hyperparam Sweep", **title_kw)
    ax7.tick_params(colors="#8b949e")
    ax7b.tick_params(colors="#90e0ef")
    ax7.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax7.set_axisbelow(True)
    for spine in ax7.spines.values():
        spine.set_edgecolor("#30363d")
    for spine in ax7b.spines.values():
        spine.set_edgecolor("#30363d")
    ax7.set_facecolor("#161b22")

    # ── Panel 8: Architecture comparison — same-horizon MAE ──────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax_bg(ax8)
    arch_groups = {
        "NST\nSmall": [r for r in rows if r["name"].startswith("nst_small")],
        "NST\nMed":   [r for r in rows if r["name"].startswith("nst_med")],
        "NST\nLarge": [r for r in rows if r["name"].startswith("nst_large")],
        "PTST\nSmall":[r for r in rows if r["name"].startswith("ptst_small")],
        "PTST\nMed":  [r for r in rows if r["name"].startswith("ptst_med")],
        "PTST\np32":  [r for r in rows if r["name"].startswith("ptst_p32")],
    }
    ag_labels, ag_min, ag_med, ag_max = [], [], [], []
    for lbl, grp in arch_groups.items():
        if not grp: continue
        maes_g = [r["mae"] for r in grp]
        ag_labels.append(lbl)
        ag_min.append(min(maes_g))
        ag_med.append(np.median(maes_g))
        ag_max.append(max(maes_g))
    x = np.arange(len(ag_labels))
    arch_cols = ["#00b4d8","#0077b6","#023e8a","#e63946","#c1121f","#ff6b35"]
    for i, (mn, md, mx, col) in enumerate(zip(ag_min, ag_med, ag_max, arch_cols)):
        ax8.plot([i,i], [mn, mx], color=col, linewidth=2, alpha=0.5)
        ax8.scatter([i], [mn], color=col, s=80, zorder=5, marker="^")
        ax8.scatter([i], [md], color=col, s=60, zorder=5, marker="o", alpha=0.7)
        ax8.scatter([i], [mx], color=col, s=80, zorder=5, marker="v", alpha=0.5)
    ax8.set_yscale("log")
    ax8.set_xticks(x)
    ax8.set_xticklabels(ag_labels, fontsize=8, **txt_kw)
    ax8.set_ylabel("Val MAE (log) — min/med/max", **txt_kw, fontsize=8)
    ax8.set_title("Architecture MAE Ranges", **title_kw)
    ax8.tick_params(colors="#8b949e")
    ax8.yaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax8.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 9: Net IC learning curves — NST Small variants ─────────────────
    ax9 = fig.add_subplot(gs[3, :])
    ax9.set_facecolor("#161b22")
    ax9.set_title("NST-Small: Net IC Learning Curves (spread-adjusted, real-world signal)",
                  **title_kw)
    has_net_ic = False
    for r in nst_small_names:
        hist = r.get("hist", [])
        net_ics = [(h[0], h[4]) for h in hist if not np.isnan(h[4])]
        if not net_ics:
            continue
        has_net_ic = True
        eps_ni, nic_vals = zip(*net_ics)
        label = r["name"].replace("nst_small_", "")
        ax9.plot(eps_ni, nic_vals, marker="D", markersize=3, linewidth=1.5, label=label)
    ax9.axhline(0,   color="#30363d", linewidth=1,   linestyle="--", label="net_IC=0 (no edge)")
    ax9.axhline(0.1, color="#3fb950", linewidth=0.8, linestyle=":", label="net_IC=0.1 (viable)")
    if not has_net_ic:
        ax9.text(0.5, 0.5, "net_IC not yet logged\n(jobs using old 02_train.py)",
                 transform=ax9.transAxes, ha="center", va="center",
                 fontsize=11, color="#8b949e", fontfamily="monospace")
    ax9.set_xlabel("Epoch", **txt_kw, fontsize=9)
    ax9.set_ylabel("Val net_IC (spread-adjusted)", **txt_kw, fontsize=9)
    ax9.legend(fontsize=6, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="lower right", ncol=3)
    ax9.tick_params(colors="#8b949e")
    ax9.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax9.xaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax9.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 10: Top 15 table ────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[4, :2])
    ax10.set_facecolor("#0d1117")
    ax10.axis("off")
    ax10.set_title("Top 15 Experiments — Current Best Metrics", **title_kw)

    top15 = rows[:15]
    col_labels = ["Rank", "Experiment", "Arch", "Epochs", "Val MAE", "DA", "IC", "net_IC"]
    col_widths = [0.04, 0.26, 0.09, 0.06, 0.12, 0.07, 0.09, 0.09]
    col_x = [sum(col_widths[:i]) + 0.02 for i in range(len(col_widths))]
    row_h = 0.057
    y0 = 0.92
    for j, (lbl, cx) in enumerate(zip(col_labels, col_x)):
        ax10.text(cx, y0, lbl, transform=ax10.transAxes,
                  fontsize=8, fontweight="bold", color="#58a6ff",
                  fontfamily="monospace", va="top")
    for i, r in enumerate(top15):
        y = y0 - (i + 1) * row_h
        bg_col = "#161b22" if i % 2 == 0 else "#0d1117"
        ax10.add_patch(mpatches.FancyBboxPatch(
            (0.01, y - 0.005), 0.97, row_h * 0.92,
            transform=ax10.transAxes, boxstyle="round,pad=0.002",
            facecolor=bg_col, edgecolor="none", zorder=0))
        net_ic_str = (f"{r['net_ic']:.4f}" if not np.isnan(r.get("net_ic", float("nan")))
                      else "—")
        vals = [
            f"#{i+1}",
            r["name"],
            r["arch"],
            str(r["epoch"]),
            f"{r['mae']:.4e}",
            f"{r['da']:.4f}",
            f"{r['ic']:.4f}",
            net_ic_str,
        ]
        for j, (val, cx) in enumerate(zip(vals, col_x)):
            cell_col = r["color"] if j == 1 else "#e6edf3"
            if j == 6 and r["ic"] > 0.5:
                cell_col = "#39d353"
            elif j == 6 and r["ic"] < 0:
                cell_col = "#f85149"
            if j == 7 and not np.isnan(r.get("net_ic", float("nan"))):
                if r["net_ic"] > 0.1:
                    cell_col = "#39d353"
                elif r["net_ic"] < 0:
                    cell_col = "#f85149"
            ax10.text(cx, y + row_h * 0.3, val, transform=ax10.transAxes,
                      fontsize=7, color=cell_col, fontfamily="monospace", va="center")

    # ── Panel 11: NST Med LR / Dropout scatter ───────────────────────────────
    ax11 = fig.add_subplot(gs[4, 2])
    ax_bg(ax11)
    lr_runs  = [r for r in rows if r["name"].startswith("nst_med_h1_lr")]
    dp_runs  = [r for r in rows if r["name"].startswith("nst_med_h1_dp")]
    wd_runs  = [r for r in rows if r["name"].startswith("nst_med_h1_wd")]
    base_run = [r for r in rows if r["name"] == "nst_med_h1"]
    for grp, lbl, col, mkr in [
        (lr_runs,  "LR sweep",      "#ffd166", "o"),
        (dp_runs,  "Dropout sweep", "#ef476f", "s"),
        (wd_runs,  "WD sweep",      "#06d6a0", "^"),
        (base_run, "Baseline",      "#ffffff",  "D"),
    ]:
        if not grp: continue
        xs = [r["ic"]  for r in grp]
        ys = [r["mae"] for r in grp]
        ax11.scatter(xs, ys, c=col, label=lbl, s=70, alpha=0.9,
                     edgecolors="#30363d", linewidths=0.5, marker=mkr, zorder=5)
        for r in grp:
            short = r["name"].replace("nst_med_h1_", "")
            ax11.annotate(short, (r["ic"], r["mae"]), fontsize=5.5,
                          color="#8b949e", textcoords="offset points", xytext=(3, 3))
    ax11.set_yscale("log")
    ax11.set_xlabel("IC (higher = better)", **txt_kw, fontsize=9)
    ax11.set_ylabel("Val MAE (log)", **txt_kw, fontsize=9)
    ax11.set_title("NST-Med h1: Hyperparams IC vs MAE", **title_kw)
    ax11.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d",
                labelcolor="#e6edf3")
    ax11.tick_params(colors="#8b949e")
    ax11.yaxis.grid(True, color="#21262d", linewidth=0.6)
    ax11.xaxis.grid(True, color="#21262d", linewidth=0.6)
    for spine in ax11.spines.values():
        spine.set_edgecolor("#30363d")

    # ── arch legend ───────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in [("NST-Small","#00b4d8"),("NST-Med","#0077b6"),
                     ("NST-Large","#023e8a"),("PTST-Small","#e63946"),
                     ("PTST-Med","#c1121f"),("PTST-p32","#ff6b35")]
    ]
    fig.legend(handles=legend_patches, loc="upper right", ncol=3,
               fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", bbox_to_anchor=(0.97, 0.957))

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    return out_path

if __name__ == "__main__":
    main()
