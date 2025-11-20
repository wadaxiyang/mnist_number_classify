import os
import struct
import numpy as np
import gzip
import urllib.request


def _download(urls, dest):
    if os.path.exists(dest):
        return
    for u in urls:
        try:
            urllib.request.urlretrieve(u, dest)
            return
        except Exception:
            pass
    raise RuntimeError("download failed")

def _read_images(path):
    with gzip.open(path, "rb") as f:
        m, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, r * c)

def _read_labels(path):
    with gzip.open(path, "rb") as f:
        m, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def load_mnist(root):
    os.makedirs(root, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    mirrors = [
        base,
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    files = {
        "train-images-idx3-ubyte.gz": _read_images,
        "train-labels-idx1-ubyte.gz": _read_labels,
        "t10k-images-idx3-ubyte.gz": _read_images,
        "t10k-labels-idx1-ubyte.gz": _read_labels,
    }
    for name in files.keys():
        dest = os.path.join(root, name)
        if not os.path.exists(dest):
            urls = [m + name for m in mirrors]
            _download(urls, dest)
    X_train = files["train-images-idx3-ubyte.gz"](os.path.join(root, "train-images-idx3-ubyte.gz"))
    y_train = files["train-labels-idx1-ubyte.gz"](os.path.join(root, "train-labels-idx1-ubyte.gz"))
    X_test = files["t10k-images-idx3-ubyte.gz"](os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    y_test = files["t10k-labels-idx1-ubyte.gz"](os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
    return X_train.astype(np.float64) / 255.0, y_train.astype(np.int64), X_test.astype(np.float64) / 255.0, y_test.astype(np.int64)
