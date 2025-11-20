import numpy as np

def softmax_cross_entropy(logits, y):
    m = logits.shape[0]
    a = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - a)
    s = e.sum(axis=1, keepdims=True)
    p = e / s
    loss = -np.log(p[np.arange(m), y] + 1e-12).mean()
    grad = p
    grad[np.arange(m), y] -= 1.0
    grad /= m
    return loss, grad

def get_im2col_indices(x_shape, kH, kW, pad, stride):
    N, C, H, W = x_shape
    H_p = H + 2 * pad
    W_p = W + 2 * pad
    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    j0 = np.tile(np.arange(kW), kH)
    j0 = np.tile(j0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kH * kW).reshape(-1, 1)
    return i.astype(np.int64), j.astype(np.int64), k.astype(np.int64), out_h, out_w

