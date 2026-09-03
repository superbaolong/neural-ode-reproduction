from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint


class NFENet(nn.Module):
    """Small vector field that counts function evaluations."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.Tanh(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.Tanh(),
        )
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.net(x)

    def reset_nfe(self) -> None:
        self.nfe = 0


class ODEBlock(nn.Module):
    def __init__(self, channels: int, adjoint: bool = False, rtol: float = 1e-3, atol: float = 1e-5):
        super().__init__()
        self.func = NFENet(channels)
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.func.reset_nfe()
        solver = odeint_adjoint if self.adjoint else odeint
        times = x.new_tensor([0.0, 1.0])
        out = solver(self.func, x, times, rtol=self.rtol, atol=self.atol, method="dopri5")
        return out[-1]

    @property
    def nfe(self) -> int:
        return self.func.nfe


class ODENet(nn.Module):
    """Compact ODENet suitable for MNIST and low-memory hardware."""

    def __init__(self, channels: int = 32, classes: int = 10, adjoint: bool = False, rtol: float = 1e-3, atol: float = 1e-5):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.ode = ODEBlock(channels, adjoint=adjoint, rtol=rtol, atol=atol)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ode(self.downsampling(x)))

    @property
    def nfe(self) -> int:
        return self.ode.nfe


class ResNetBaseline(nn.Module):
    """Residual baseline with a comparable parameter scale."""

    def __init__(self, channels: int = 32, classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU())
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels)
        )
        self.pool = nn.AvgPool2d(2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.relu(x + self.block(x))
        return self.head(self.pool(x))

    @property
    def nfe(self) -> int:
        return 0
