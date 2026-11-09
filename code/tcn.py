"""Small optional TCN architecture. No trained checkpoint or quality claim."""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(0.3))
    def forward(self, x):
        return x + self.block(x)

class ImportanceTCN(nn.Module):
    def __init__(self, dimension=512):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.3))
        self.temporal = nn.Sequential(*(ResidualBlock(128, d) for d in (1, 2, 4)))
        self.output = nn.Linear(128, 1)
    def forward(self, x):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("Expected nonempty [batch, time, dimension].")
        encoded = self.input(x).transpose(1, 2)
        return torch.sigmoid(self.output(self.temporal(encoded).transpose(1, 2))).squeeze(-1)

def ranking_mse_loss(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] == 0:
        raise ValueError("Predictions and targets require matching nonempty [batch,time].")
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Targets must be finite and within [0,1].")
    loss = torch.mean((predicted - target) ** 2)
    # Deterministic within-video adjacent pairs. No cross-video ranking pairs.
    delta = target[:, 1:] - target[:, :-1]
    mask = delta != 0
    if mask.any():
        ranking = torch.relu(0.05 - torch.sign(delta) * (predicted[:, 1:] - predicted[:, :-1]))
        loss = loss + ranking[mask].mean()
    return loss
