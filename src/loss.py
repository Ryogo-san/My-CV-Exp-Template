import numpy as np
import torch
import torch.nn as nn


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()

        loss = (pred - gt) ** 2
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)

        return loss


class WeightedHeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()

        weight = 0.6
        loss = pred - gt
        loss = np.where(loss >= 0, weight * loss, torch.abs(loss))
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)

        return loss
