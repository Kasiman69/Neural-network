"""Step 1: load MNIST and prepare 784 brightness values per image."""

import gzip
import struct
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download_file(filename):
    """Download once; reuse the local file on later runs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}...", flush=True)
        temporary = path.with_suffix(".download")
        try:
            urlretrieve(BASE_URL + filename, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def load_images(filename):
    with gzip.open(download_file(filename), "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError("Expected MNIST images with 28 x 28 pixels.")
        pixels = np.frombuffer(file.read(), dtype=np.uint8)

    # One row per image, with 784 columns in a consistent pixel order.
    images = pixels.reshape(count, rows * columns)

    # Convert integer brightness (0..255) into decimal brightness (0..1).
    return images.astype(np.float32) / 255.0


def load_labels(filename):
    with gzip.open(download_file(filename), "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError("Expected an MNIST label file.")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if len(labels) != count or np.any(labels > 9):
        raise ValueError("Invalid MNIST labels.")
    return labels


def load_mnist():
    # Keep the test examples separate: they are for evaluating later.
    x_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    x_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_mnist()
    print("Training images:", x_train.shape)
    print("Test images:", x_test.shape)
    print("Brightness range:", x_train.min(), "to", x_train.max())
    print("First image's correct digit:", y_train[0])

    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for index, axis in enumerate(axes.flat):
        # Reshape only for display; the network will receive the flat vector.
        axis.imshow(x_train[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Label: {y_train[index]}")
        axis.axis("off")
    figure.suptitle("MNIST: our first ten training examples")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
