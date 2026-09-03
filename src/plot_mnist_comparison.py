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
    args = parser.parse_args()

    grouped = load_rows(args.metrics)
    colors = {"odenet": "#3b82f6", "resnet": "#f97316"}
    labels = {"odenet": "ODENet", "resnet": "ResNet"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for model, rows in grouped.items():
        epochs = [int(row["epoch"]) for row in rows]
        test_accuracy = [100 * float(row["test_accuracy"]) for row in rows]
        train_seconds = [float(row["train_seconds"]) for row in rows]
        axes[0].plot(epochs, test_accuracy, marker="o", label=labels.get(model, model), color=colors.get(model))
        axes[1].plot(epochs, train_seconds, marker="o", label=labels.get(model, model), color=colors.get(model))

    model_names = list(grouped)
    peak_memory = [max(float(row["peak_memory_mb"]) for row in grouped[name]) for name in model_names]
    axes[2].bar([labels.get(name, name) for name in model_names], peak_memory, color=[colors.get(name) for name in model_names])

    axes[0].set(title="MNIST test accuracy", xlabel="Epoch", ylabel="Accuracy (%)", xticks=range(1, 6))
    axes[1].set(title="Training time per epoch", xlabel="Epoch", ylabel="Seconds", xticks=range(1, 6))
    axes[2].set(title="Peak allocated GPU memory", ylabel="MiB")
    axes[0].legend()
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)

    fig.suptitle("Fair MNIST comparison (seed 0, 5 epochs, batch size 64)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()

