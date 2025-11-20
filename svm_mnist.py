import os
import numpy as np
import time
from mySVM import OVRKernelSVM, LinearSVM
from Load_data import load_mnist

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "mnist")
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    rng = np.random.default_rng(0)
    n_train_linear = 20000
    n_train_kernel = 3000
    n_test_eval = 5000
    idx_lin = rng.choice(X_train.shape[0], n_train_linear, replace=False)
    idx_k = rng.choice(X_train.shape[0], n_train_kernel, replace=False)
    idx_te = rng.choice(X_test.shape[0], n_test_eval, replace=False)
    X_lin = X_train[idx_lin]
    y_lin = y_train[idx_lin]
    X_k = X_train[idx_k]
    y_k = y_train[idx_k]
    X_te = X_test[idx_te]
    y_te = y_test[idx_te]
    lin = LinearSVM(num_classes=10, reg=1e-4, lr=0.3, epochs=12, batch_size=512, seed=1)
    lin.fit(X_lin, y_lin)
    y_pred_lin = lin.predict(X_te)
    acc_lin = accuracy(y_te, y_pred_lin)
    ker = OVRKernelSVM(num_classes=10, C=2.0, gamma=0.05, tol=1e-3, max_passes=2)
    ker.fit(X_k, y_k)
    y_pred_ker = ker.predict(X_te)
    acc_ker = accuracy(y_te, y_pred_ker)
    print("Linear SVM accuracy:", float(acc_lin))
    print("Kernel SVM accuracy:", float(acc_ker))
    print("Elapsed seconds:", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()