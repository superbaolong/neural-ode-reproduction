from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint


class SpiralFunc(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, t, z):
        return self.net(z)


def make_spiral(n_points: int, device: torch.device):
    t = torch.linspace(0, 1, n_points, device=device)
    radius = 0.15 + 1.8 * t
    angle = 1.5 * math.pi * t
    z = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
    return t, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/spiral")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    t, target = make_spiral(args.n_points, device)
    func = SpiralFunc().to(device)
    opt = torch.optim.Adam(func.parameters(), lr=args.lr)
    rows = []
    z0 = target[0:1]
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        pred = odeint(func, z0, t, rtol=1e-5, atol=1e-7, method="dopri5")[:, 0]
        loss = (pred - target).pow(2).mean()
        loss.backward(); opt.step()
        rows.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch:04d} loss={loss.item():.6f}")
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"]); writer.writeheader(); writer.writerows(rows)
    with (out / "config.json").open("w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2)
    plt.figure(figsize=(6, 5)); plt.plot(target[:, 0].cpu(), target[:, 1].cpu(), label="target"); plt.plot(pred[:, 0].detach().cpu(), pred[:, 1].detach().cpu(), "--", label="Neural ODE"); plt.legend(); plt.axis("equal"); plt.tight_layout(); plt.savefig(out / "spiral.png", dpi=160); plt.close()
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
