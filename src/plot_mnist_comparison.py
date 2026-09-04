from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    grouped = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["model"]].append(row)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("mnist-comparison.png"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    grouped = load_rows(args.metrics)
    colors = {"odenet": "#3b82f6", "resnet": "#f97316"}
    labels = {"odenet": "ODENet", "resnet": "ResNet"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for model, rows in grouped.items():
        epochs = [int(row["epoch"]) for row in rows]
        test_accuracy = [100 * float(row["test_accuracy"]) for row in rows]
        test_loss = [float(row["test_loss"]) for row in rows]
        train_seconds = [float(row["train_seconds"]) for row in rows]
        nfe_mean = [float(row["nfe_mean"]) for row in rows]
        axes[0].plot(epochs, test_accuracy, marker="o", label=labels.get(model, model), color=colors.get(model))
        axes[1].plot(epochs, test_loss, marker="o", label=labels.get(model, model), color=colors.get(model))
        axes[2].plot(epochs, train_seconds, marker="o", label=labels.get(model, model), color=colors.get(model))
        axes[3].plot(epochs, nfe_mean, marker="o", label=labels.get(model, model), color=colors.get(model))

    max_epoch = max(int(row["epoch"]) for rows in grouped.values() for row in rows)
    tick_step = 1 if max_epoch <= 10 else 5
    ticks = list(range(tick_step, max_epoch + 1, tick_step))
    if 1 not in ticks:
        ticks.insert(0, 1)

    axes[0].set(title="MNIST test accuracy", xlabel="Epoch", ylabel="Accuracy (%)", xticks=ticks)
    axes[1].set(title="MNIST test loss", xlabel="Epoch", ylabel="Cross-entropy", xticks=ticks)
    axes[2].set(title="Training time per epoch", xlabel="Epoch", ylabel="Seconds", xticks=ticks)
    axes[3].set(title="Mean function evaluations", xlabel="Epoch", ylabel="NFE", xticks=ticks)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(args.title or f"MNIST comparison ({max_epoch} epochs)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()

