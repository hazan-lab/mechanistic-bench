"""Simple recurrent mixers: Elman RNN, LSTM, and an MLP baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class ElmanRNN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cell = nn.RNN(d_model, d_model, num_layers=1, batch_first=True, nonlinearity="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.cell(x)
        return out


class LSTMMixer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cell = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.cell(x)
        return out


class MLPMixer(nn.Module):
    """Token-wise MLP — no cross-token mixing. Useful as a low-capability baseline."""

    def __init__(self, d_model: int, hidden_mult: float = 4.0):
        super().__init__()
        h = int(d_model * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
