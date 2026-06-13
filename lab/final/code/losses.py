from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_zscore(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / denom
    var = (((x - mean) ** 2) * mask).sum(dim=1, keepdim=True) / denom
    return (x - mean) / torch.sqrt(var + eps)


def ic_loss(scores: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative cross-sectional Pearson IC, averaged by date/batch."""
    s = masked_zscore(scores, mask, eps)
    r = masked_zscore(returns, mask, eps)
    valid_n = mask.sum(dim=1).clamp_min(1.0)
    ic = (s * r * mask).sum(dim=1) / valid_n
    return -ic.mean()


def mse_rank_loss(scores: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss: regress z-scored returns, not raw noisy returns."""
    s = masked_zscore(scores, mask)
    r = masked_zscore(returns, mask)
    return (((s - r) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def scores_to_weights(
    scores: torch.Tensor,
    mask: torch.Tensor | None = None,
    top_n: int | None = 30,
    temperature: float = 0.05,
    equal_weight: bool = False,
) -> torch.Tensor:
    """Long-only, fully-invested portfolio weights from scores."""
    if mask is None:
        mask = torch.ones_like(scores)
    mask = mask.float()
    s = scores.masked_fill(mask <= 0, -1e9)

    if top_n is not None and top_n < scores.size(1):
        idx = torch.topk(s, k=top_n, dim=1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
        keep = keep & (mask > 0)
    else:
        keep = mask > 0

    if equal_weight:
        w = keep.float()
        return w / w.sum(dim=1, keepdim=True).clamp_min(1.0)

    logits = (scores / temperature).masked_fill(~keep, -1e9)
    return F.softmax(logits, dim=1)


def portfolio_sharpe_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    top_n: int | None = 30,
    temperature: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable approximation of portfolio objective."""
    w = scores_to_weights(scores, mask, top_n=top_n, temperature=temperature, equal_weight=False)
    port_ret = (w * returns * mask).sum(dim=1)
    # std() can have NaN gradients if variance is exactly 0. Compute var manually.
    var = port_ret.var(unbiased=False)
    sharpe = port_ret.mean() / (torch.sqrt(var + eps))
    return -sharpe


def combined_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    alpha_mse: float = 0.1,
    alpha_sharpe: float = 0.05,
    top_n: int = 30,
) -> torch.Tensor:
    return (
        ic_loss(scores, returns, mask)
        + alpha_mse * mse_rank_loss(scores, returns, mask)
        + alpha_sharpe * portfolio_sharpe_loss(scores, returns, mask, top_n=top_n)
    )
