"""Whole-complex binary classifier built on top of the HOGAT backbone."""

import torch
import torch.nn as nn

from hogat import HOGAT

__all__ = ["HOGATGraphClassifier"]


"""Deep Graph Infomax over a cellular complex with a fused, whole-complex summary."""
import torch
import torch.nn as nn
from hogat import HOGAT

EPS = 1e-15


class HOGATInfomax(nn.Module):
    def __init__(self, in_channels_0, in_channels_1, in_channels_2,
                 hid_channels, n_layers, **kwargs):
        super().__init__()
        self.backbone = HOGAT(
            in_channels_0=in_channels_0,
            in_channels_1=in_channels_1,
            in_channels_2=in_channels_2,
            hid_channels=hid_channels,
            n_layers=n_layers,
            **kwargs,
        )
        self.summary_proj = nn.Linear(3 * hid_channels, hid_channels)
        self.weight = nn.Parameter(torch.empty(hid_channels, hid_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.summary_proj.reset_parameters()
        if hasattr(self.backbone, "reset_parameters"):
            self.backbone.reset_parameters()

    @staticmethod
    def corrupt_features(x_0, x_1, x_2):
        """Independently shuffle rows within each rank; structure is untouched."""
        idx_0 = torch.randperm(x_0.size(0), device=x_0.device)
        idx_1 = torch.randperm(x_1.size(0), device=x_1.device)
        idx_2 = torch.randperm(x_2.size(0), device=x_2.device)
        return x_0[idx_0], x_1[idx_1], x_2[idx_2]

    def summary_fn(self, x_0, x_1, x_2):
        pooled = torch.cat([x_0.mean(dim=0), x_1.mean(dim=0), x_2.mean(dim=0)], dim=-1)
        return torch.sigmoid(self.summary_proj(pooled))

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    '''def forward(self, x_0, x_1, x_2, adjacency_0_up, incidence_1, incidence_1_t,
                adjacency_1_down, adjacency_1_up, incidence_2, incidence_2_t,
                adjacency_2_down):
        structure = (adjacency_0_up, incidence_1, incidence_1_t, adjacency_1_down,
                     adjacency_1_up, incidence_2, incidence_2_t, adjacency_2_down)

        pos_0, pos_1, pos_2 = self.backbone(x_0, x_1, x_2, *structure)

        x_0_c, x_1_c, x_2_c = self.corrupt_features(x_0, x_1, x_2)
        neg_0, neg_1, neg_2 = self.backbone(x_0_c, x_1_c, x_2_c, *structure)

        summary = self.summary_fn(pos_0, pos_1, pos_2)
        return (pos_0, pos_1, pos_2), (neg_0, neg_1, neg_2), summary'''
    
    def forward(self, x_0, x_1, x_2, adjacency_0_up, incidence_1, incidence_1_t,
                adjacency_1_down, adjacency_1_up, incidence_2, incidence_2_t,
                adjacency_2_down, return_attention: bool = False):
        structure = (adjacency_0_up, incidence_1, incidence_1_t, adjacency_1_down,
                    adjacency_1_up, incidence_2, incidence_2_t, adjacency_2_down)

        if return_attention:
            pos_0, pos_1, pos_2, all_attention = self.backbone(
                x_0, x_1, x_2, *structure, return_attention=True
            )
        else:
            pos_0, pos_1, pos_2 = self.backbone(x_0, x_1, x_2, *structure)

        x_0_c, x_1_c, x_2_c = self.corrupt_features(x_0, x_1, x_2)
        neg_0, neg_1, neg_2 = self.backbone(x_0_c, x_1_c, x_2_c, *structure)

        summary = self.summary_fn(pos_0, pos_1, pos_2)

        if return_attention:
            return (pos_0, pos_1, pos_2), (neg_0, neg_1, neg_2), summary, all_attention
        return (pos_0, pos_1, pos_2), (neg_0, neg_1, neg_2), summary

    def loss(self, pos, neg, summary):
        pos_0, pos_1, pos_2 = pos
        neg_0, neg_1, neg_2 = neg
        pos_loss = neg_loss = 0.0
        for pz, nz in ((pos_0, neg_0), (pos_1, neg_1), (pos_2, neg_2)):
            pos_loss = pos_loss - torch.log(self.discriminate(pz, summary) + 1e-15).mean()
            neg_loss = neg_loss - torch.log(1 - self.discriminate(nz, summary) + 1e-15).mean()
        return pos_loss + neg_loss