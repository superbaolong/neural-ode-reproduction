from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic == 2051:
            n, rows, cols = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        if magic == 2049:
            n, = struct.unpack(">I", f.read(4))
            return np.frombuffer(f.read(), dtype=np.uint8)[:n]
    raise ValueError(f"Unsupported IDX magic number in {path}")


class MNISTIdx(Dataset):
    def __init__(self, root: str | os.PathLike[str], train: bool):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        prefix = "train" if train else "test"
        image_key, label_key = f"{prefix}_images", f"{prefix}_labels"
        image_path, label_path = root / FILES[image_key], root / FILES[label_key]
        for key, path in ((image_key, image_path), (label_key, label_path)):
            if not path.exists():
                print(f"Downloading {FILES[key]} ...")
                urllib.request.urlretrieve(BASE_URL + FILES[key], path)
        images = _read_idx(image_path).astype(np.float32) / 255.0
        labels = _read_idx(label_path).astype(np.int64)
        self.images = torch.from_numpy(images[:, None])
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def make_loaders(data_dir: str, batch_size: int, workers: int = 0):
    train = MNISTIdx(data_dir, train=True)
    test = MNISTIdx(data_dir, train=False)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available()),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available()),
    )
