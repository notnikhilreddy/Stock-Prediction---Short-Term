"""
Evaluation metrics for BTC price prediction and real-world trading simulation.

Two evaluation paths:
  evaluate_all()     — regression: MAE, MSE, DA, IC, net_IC
  evaluate_all_cls() — classification: accuracy, directional accuracy, ordinal IC
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mae(pred, target):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


def mse(pred, target):
    """Mean Squared Error."""
    return torch.mean((pred - target) ** 2).item()


def rmse(pred, target):
    """Root Mean Squared Error."""
    return np.sqrt(mse(pred, target))


def directional_accuracy(pred, target):
    """
    Fraction of non-zero-return samples where the predicted sign matches actual sign.

    Zero-return ticks (|target| < 1e-10) are excluded because sign(0) = 0 can
    never match a non-zero prediction, producing artificially low DA at h1 where
    roughly half of 100ms ticks have no mid-price movement.
    """
    mask = torch.abs(target) > 1e-10
    if mask.sum() == 0:
        return float("nan")
    correct = (torch.sign(pred[mask]) == torch.sign(target[mask])).float()
    return correct.mean().item()


def ic(pred, target):
    """Information Coefficient (Pearson correlation between predictions and targets)."""
    p = pred - pred.mean()
    t = target - target.mean()
    denom = torch.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-12
    return (p * t).sum().item() / denom.item()


def net_ic(pred, target, spread_frac):
    """
    IC on spread-adjusted returns.

    spread_frac: full round-trip spread as a fraction (F02_spread_bps / 10000).
    net_target = target - spread_frac approximates the return achievable after
    paying the bid-ask spread on entry (using entry spread as a proxy for both
    entry and exit costs in the per-epoch training context).

    A positive net_IC means the model's signal survives spread costs — the
    primary real-world viability indicator tracked during training.
    """
    net_target = target - spread_frac
    return ic(pred, net_target)


def evaluate_all(pred, target, spread_frac=None):
    """Compute all regression metrics and return as dict."""
    result = {
        "mae":  mae(pred, target),
        "mse":  mse(pred, target),
        "rmse": rmse(pred, target),
        "da":   directional_accuracy(pred, target),
        "ic":   ic(pred, target),
    }
    if spread_frac is not None:
        result["net_ic"] = float(net_ic(pred, target, spread_frac))
    return result


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def class_accuracy(logits, labels):
    """
    Overall fraction of samples where argmax(logits) == label.
    logits: (N, n_classes), labels: (N,) long
    """
    pred_cls = logits.argmax(dim=-1)
    return (pred_cls == labels).float().mean().item()


def directional_accuracy_cls(logits, labels):
    """
    Directional accuracy on non-neutral samples only (label != 1).
    Measures how well the model distinguishes Down (0) vs Up (2) ignoring
    the Neutral class — directly analogous to trade-direction accuracy.
    """
    mask = labels != 1
    if mask.sum() == 0:
        return float("nan")
    pred_cls = logits.argmax(dim=-1)
    correct = (pred_cls[mask] == labels[mask]).float()
    return correct.mean().item()


def ordinal_ic(logits, labels):
    """
    Pearson IC between the net-up probability margin P(Up) - P(Down) and
    the ordinal label mapped to {-1, 0, +1}.

    This is the classification equivalent of regression IC — it measures
    whether the model's directional confidence aligns with actual moves.
    """
    probs = F.softmax(logits, dim=-1)               # (N, 3)
    signal = probs[:, 2] - probs[:, 0]              # P(Up) - P(Down), (N,)
    ordinal = (labels.float() - 1.0)               # {0,1,2} -> {-1,0,+1}
    return ic(signal, ordinal)


def evaluate_all_cls(logits, labels):
    """
    Compute all classification metrics and return as dict.
    logits: (N, n_classes) float tensor
    labels: (N,) long tensor
    """
    return {
        "accuracy":   class_accuracy(logits, labels),
        "da":         directional_accuracy_cls(logits, labels),
        "ordinal_ic": ordinal_ic(logits, labels),
        "ce_loss":    F.cross_entropy(logits, labels).item(),
    }


# ---------------------------------------------------------------------------
# Trading / backtest metrics (operate on numpy arrays of per-trade returns)
# ---------------------------------------------------------------------------

def sharpe_ratio(returns, annualize_factor=1.0):
    """
    Annualized Sharpe ratio of a series of per-trade returns.

    annualize_factor: multiply std-normalized mean by this to annualize.
    For 100ms trades at 6.5h/day: sqrt(6.5*3600/0.1 * 252) ≈ 24300.
    Leave as 1.0 to get the per-trade Sharpe.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-15:
        return 0.0
    return (returns.mean() / std) * annualize_factor


def sortino_ratio(returns, annualize_factor=1.0):
    """Sortino ratio: like Sharpe but only penalizes downside volatility."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std()
    if downside_std < 1e-15:
        return 0.0
    return (returns.mean() / downside_std) * annualize_factor


def hit_rate(returns):
    """Fraction of trades with positive net return."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def max_drawdown(returns):
    """
    Maximum peak-to-trough drawdown of the cumulative return series.
    Returns a positive number representing the magnitude of the worst drawdown.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return 0.0
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    return float(drawdowns.max())


def profit_factor(returns):
    """
    Sum of winning returns divided by absolute sum of losing returns.
    > 1.0 means the strategy makes more on winners than it loses on losers.
    """
    returns = np.asarray(returns, dtype=np.float64)
    wins   = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    if losses < 1e-15:
        return float("inf") if wins > 0 else 1.0
    return float(wins / losses)


def trading_metrics(returns):
    """Compute all trading metrics for a given array of per-trade net returns."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return {
            "n_trades":     0,
            "hit_rate":     float("nan"),
            "mean_return":  float("nan"),
            "total_return": float("nan"),
            "sharpe":       float("nan"),
            "sortino":      float("nan"),
            "max_drawdown": float("nan"),
            "profit_factor":float("nan"),
        }
    return {
        "n_trades":     int(len(returns)),
        "hit_rate":     hit_rate(returns),
        "mean_return":  float(returns.mean()),
        "total_return": float(returns.sum()),
        "sharpe":       sharpe_ratio(returns),
        "sortino":      sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "profit_factor":profit_factor(returns),
    }
