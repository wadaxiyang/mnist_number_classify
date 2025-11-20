import numpy as np

def rbf_kernel(X, Y, gamma):
    Xn = (X ** 2).sum(axis=1).reshape(-1, 1)
    Yn = (Y ** 2).sum(axis=1).reshape(1, -1)
    d2 = Xn + Yn - 2.0 * (X @ Y.T)
    return np.exp(-gamma * d2)

class KernelSVMBinary:
    def __init__(self, C=1.0, gamma=0.05, tol=1e-3, max_passes=5, max_iter=100000):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.support_idx = None

    def fit(self, X, y):
        n = X.shape[0]
        self.X = X
        self.y = y.astype(np.float64)
        self.alphas = np.zeros(n)
        K = rbf_kernel(X, X, self.gamma)
        passes = 0
        iters = 0
        while passes < self.max_passes and iters < self.max_iter:
            num_changed = 0
            for i in range(n):
                f_i = (self.alphas * self.y) @ K[:, i] + self.b
                E_i = f_i - self.y[i]
                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (self.y[i] * E_i > self.tol and self.alphas[i] > 0):
                    j = i
                    while j == i:
                        j = np.random.randint(0, n)
                    f_j = (self.alphas * self.y) @ K[:, j] + self.b
                    E_j = f_j - self.y[j]
                    a_i_old = self.alphas[i]
                    a_j_old = self.alphas[j]
                    if self.y[i] != self.y[j]:
                        L = max(0.0, a_j_old - a_i_old)
                        H = min(self.C, self.C + a_j_old - a_i_old)
                    else:
                        L = max(0.0, a_i_old + a_j_old - self.C)
                        H = min(self.C, a_i_old + a_j_old)
                    if L == H:
                        continue
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    self.alphas[j] -= self.y[j] * (E_i - E_j) / eta
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L
                    if abs(self.alphas[j] - a_j_old) < 1e-5:
                        continue
                    self.alphas[i] += self.y[i] * self.y[j] * (a_j_old - self.alphas[j])
                    b1 = self.b - E_i - self.y[i] * (self.alphas[i] - a_i_old) * K[i, i] - self.y[j] * (self.alphas[j] - a_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alphas[i] - a_i_old) * K[i, j] - self.y[j] * (self.alphas[j] - a_j_old) * K[j, j]
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)
                    num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
            iters += 1
        self.support_idx = self.alphas > 1e-6

    def decision_function(self, X):
        sv = self.support_idx
        if sv.sum() == 0:
            return np.full(X.shape[0], self.b)
        K = rbf_kernel(X, self.X[sv], self.gamma)
        return K @ (self.alphas[sv] * self.y[sv]) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

class OVRKernelSVM:
    def __init__(self, num_classes=10, C=1.0, gamma=0.05, tol=1e-3, max_passes=3):
        self.num_classes = num_classes
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.models = []

    def fit(self, X, y):
        self.models = []
        for c in range(self.num_classes):
            yy = np.where(y == c, 1.0, -1.0)
            m = KernelSVMBinary(C=self.C, gamma=self.gamma, tol=self.tol, max_passes=self.max_passes)
            m.fit(X, yy)
            self.models.append(m)

    def predict(self, X):
        scores = []
        for m in self.models:
            scores.append(m.decision_function(X))
        S = np.vstack(scores).T
        return S.argmax(axis=1)
