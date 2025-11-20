import numpy as np
from myCNN.utils import get_im2col_indices


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, pad=2, seed=0):
        rng = np.random.default_rng(seed)
        kH = kernel_size
        kW = kernel_size
        self.W = rng.standard_normal((out_channels, in_channels, kH, kW)).astype(np.float32) * np.sqrt(2.0 / (in_channels * kH * kW))
        self.b = np.zeros((out_channels,), dtype=np.float32)
        self.stride = stride
        self.pad = pad
        self.cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        kH, kW = self.W.shape[2], self.W.shape[3]
        Xp = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))
        i, j, k, out_h, out_w = get_im2col_indices(X.shape, kH, kW, self.pad, self.stride)
        cols = Xp[:, k, i, j]
        Wc = self.W.reshape(self.W.shape[0], -1)
        out = np.tensordot(cols, Wc, axes=([1], [1]))
        out = out.transpose(0, 2, 1).reshape(N, self.W.shape[0], out_h, out_w)
        out += self.b.reshape(1, -1, 1, 1)
        self.cache = (X.shape, Xp.shape, i, j, k, cols, Wc, out_h, out_w)
        return out

    def backward(self, dout):
        X_shape, Xp_shape, i, j, k, cols, Wc, out_h, out_w = self.cache
        N, C, H, W = X_shape
        F = Wc.shape[0]
        kHkW = Wc.shape[1]
        dout_r = dout.reshape(N, F, out_h * out_w)
        db = dout.sum(axis=(0, 2, 3))
        dWc = np.tensordot(dout_r, cols, axes=([0, 2], [0, 2]))
        dcols = np.tensordot(dout_r, Wc, axes=([1], [0])).transpose(0, 2, 1)
        dXp = np.zeros(Xp_shape, dtype=np.float32)
        kk = np.repeat(k, out_h * out_w, axis=1)
        for n in range(N):
            np.add.at(dXp[n], (kk, i, j), dcols[n])
        dX = dXp[:, :, self.pad:self.pad + H, self.pad:self.pad + W]
        dW = dWc.reshape(F, C, int(np.sqrt(kHkW)), int(np.sqrt(kHkW)))
        return dX, dW.astype(np.float32), db.astype(np.float32)

