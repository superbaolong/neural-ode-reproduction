from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mnist_data import make_loaders
from .models import ODENet, ResNetBaseline


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, train: bool, max_batches: int | None):
    model.train(train); total_loss = total_correct = total = 0; start = time.perf_counter(); nfe = []
    criterion = nn.CrossEntropyLoss()
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches: break
        x, y = x.to(device), y.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = criterion(logits, y)
        if train: loss.backward(); optimizer.step()
        total_loss += loss.item() * len(y); total_correct += (logits.argmax(1) == y).sum().item(); total += len(y)
        nfe.append(getattr(model, "nfe", 0))
    return {"loss": total_loss / max(1, total), "accuracy": total_correct / max(1, total), "seconds": time.perf_counter() - start, "nfe_mean": float(np.mean(nfe)) if nfe else 0.0}


def build_model(name: str, args):
    if name == "odenet": return ODENet(channels=args.channels, adjoint=args.adjoint, rtol=args.rtol, atol=args.atol)
    if name == "resnet": return ResNetBaseline(channels=args.channels)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["odenet", "resnet", "both"], default="both")
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32); p.add_argument("--adjoint", action="store_true"); p.add_argument("--rtol", type=float, default=1e-3); p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--tolerances", nargs="+", type=float); p.add_argument("--seed", type=int, default=0); p.add_argument("--data-dir", default="data/mnist"); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int); p.add_argument("--max-test-batches", type=int); p.add_argument("--output-dir", default="outputs/mnist")
    args = p.parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.workers)
    names = ["odenet", "resnet"] if args.model == "both" else [args.model]
    tolerances = args.tolerances or [args.rtol]
    all_rows = []
    for name in names:
        for rtol in tolerances if name == "odenet" else [args.rtol]:
            local_args = argparse.Namespace(**vars(args)); local_args.rtol = rtol
            model = build_model(name, local_args).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            params = sum(p.numel() for p in model.parameters())
            print(f"\n{name} rtol={rtol:g} params={params:,} device={device} adjoint={args.adjoint}")
            for epoch in range(1, args.epochs + 1):
                tr = run_epoch(model, train_loader, optimizer, device, True, args.max_train_batches)
                with torch.no_grad(): te = run_epoch(model, test_loader, optimizer, device, False, args.max_test_batches)
                row = {"model": name, "rtol": rtol, "epoch": epoch, "params": params, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "test_loss": te["loss"], "test_accuracy": te["accuracy"], "train_seconds": tr["seconds"], "test_seconds": te["seconds"], "nfe_mean": tr["nfe_mean"], "adjoint": args.adjoint, "seed": args.seed}
                all_rows.append(row); print(f"epoch={epoch:02d} train_acc={tr['accuracy']:.4f} test_acc={te['accuracy']:.4f} nfe={tr['nfe_mean']:.1f} time={tr['seconds']:.1f}s")
    fields = list(all_rows[0]) if all_rows else []
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    with (out / "config.json").open("w") as f: json.dump(vars(args) | {"device": str(device)}, f, indent=2)
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__": main()
