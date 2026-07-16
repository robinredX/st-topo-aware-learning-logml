"""Deep Graph Infomax machinery: readout, discriminator and the contrastive loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def avg_readout(h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Graph/complex summary: sigmoid of the (optionally masked) mean patch embedding."""
    if mask is not None:
        m = mask.float().unsqueeze(-1)
        s = (h * m).sum(0) / m.sum().clamp_min(1.0)
    else:
        s = h.mean(0)
    return torch.sigmoid(s)


class BilinearDiscriminator(nn.Module):
    """DGI discriminator: bilinear score ``h_i^T W s`` between a patch and the summary."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """Return logits ``[N]`` scoring each patch embedding against ``summary`` ``[dim]``."""
        ws = self.weight @ summary
        return h @ ws


class InfomaxHead(nn.Module):
    """Readout + discriminator + BCE loss for one embedding space (one rank).

    Parameters
    ----------
    dim : int
        Embedding dimension produced by the encoder for this rank.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.discriminator = BilinearDiscriminator(dim)

    def forward(
        self,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        summary: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """Compute the DGI BCE loss and return ``(loss, pos_logits, neg_logits)``.

        The summary is read out from the *positive* embeddings (detached-free, as in DGI).
        Positive patches are labelled 1, corrupted patches 0.
        """
        if pos_emb.shape[0] == 0:
            zero = pos_emb.new_zeros(())
            empty = pos_emb.new_zeros((0,))
            return zero, empty, empty
        if summary is None:
            summary = avg_readout(pos_emb, mask=mask)
        pos_logits = self.discriminator(pos_emb, summary)
        neg_logits = self.discriminator(neg_emb, summary)
        loss = dgi_bce_loss(pos_logits, neg_logits)
        return loss, pos_logits, neg_logits


def dgi_bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with real patches -> 1 and corrupted patches -> 0."""
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits)
    )
    return pos_loss + neg_loss


@torch.no_grad()
def discriminator_metrics(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> dict:
    """Contrastive-quality metrics from the discriminator's logits.

    Returns accuracy (threshold 0), and the AUROC of separating real from corrupted patches
    -- the natural "is the model learning?" signal for a DGI run.
    """
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return {"dgi_acc": float("nan"), "dgi_auroc": float("nan")}
    pos = pos_logits.detach().float()
    neg = neg_logits.detach().float()
    acc = 0.5 * ((pos > 0).float().mean() + (neg <= 0).float().mean())
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float, device=scores.device)
    n_pos = pos.numel()
    n_neg = neg.numel()
    auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"dgi_acc": float(acc), "dgi_auroc": float(auroc)}
