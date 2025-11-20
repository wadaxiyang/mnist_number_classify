import numpy as np
from myCNN.utils import get_im2col_indices


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kH = kernel_size
        self.kW = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        i, j, k, out_h, out_w = get_im2col_indices(X.shape, self.kH, self.kW, 0, self.stride)
        cols = X[:, k, i, j]
        cols = cols.reshape(N, C, self.kH * self.kW, out_h * out_w)
        idx = cols.argmax(axis=2)
        out = cols.max(axis=2)
        out = out.reshape(N, C, out_h, out_w)
        self.cache = (X.shape, i, j, k, out_h, out_w, idx)
        return out

    def backward(self, dout):
        X_shape, i, j, k, out_h, out_w, idx = self.cache
        N, C, H, W = X_shape
        Q = out_h * out_w
        dcols = np.zeros((N, C, self.kH * self.kW, Q), dtype=dout.dtype)
        df = dout.reshape(N, C, Q)
        for p in range(self.kH * self.kW):
            mask = (idx == p)
            dcols[:, :, p, :] = df * mask
        dcols = dcols.reshape(N, C * self.kH * self.kW, Q)
        dX = np.zeros((N, C, H, W), dtype=dout.dtype)
        kk = np.repeat(k, Q, axis=1)
        for n in range(N):
            np.add.at(dX[n], (kk, i, j), dcols[n])
        return dX

