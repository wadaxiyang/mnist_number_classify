import numpy as np


class LinearSVM:
    def __init__(self, num_classes=10, reg=1e-4, lr=0.2, epochs=10, batch_size=256, seed=42):
        self.num_classes = num_classes
        self.reg = reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.W = None

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.W = np.zeros((d, self.num_classes))
        for e in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = idx[s : s + self.batch_size]
                Xb = X[b]
                yb = y[b]
                scores = Xb @ self.W
                correct = scores[np.arange(len(yb)), yb].reshape(-1, 1)
                margins = scores - correct + 1.0
                margins[np.arange(len(yb)), yb] = 0.0
                G = (margins > 0).astype(np.float64)
                row_sum = G.sum(axis=1)
                G[np.arange(len(yb)), yb] = -row_sum
                grad = Xb.T @ G
                grad /= len(yb)
                grad += self.reg * self.W
                self.W -= self.lr * grad

    def predict(self, X):
        return (X @ self.W).argmax(axis=1)

