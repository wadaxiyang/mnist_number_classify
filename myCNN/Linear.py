import numpy as np

class Linear:
    def __init__(self, in_features, out_features, seed=0):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((in_features, out_features)).astype(np.float32) * np.sqrt(2.0 / in_features))
        self.b = np.zeros((out_features,), dtype=np.float32)
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY):
        dW = self.X.T @ dY
        db = dY.sum(axis=0)
        dX = dY @ self.W.T
        return dX, dW.astype(np.float32), db.astype(np.float32)
