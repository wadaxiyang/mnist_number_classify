import os
import time
import numpy as np
from svm_mnist import load_mnist, LinearSVM, accuracy
from myCNN import softmax_cross_entropy
from myCNN import CNN



def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "mnist")
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    rng = np.random.default_rng(0)
    n_train = 6000
    n_test_eval = 2000
    idx_tr = rng.choice(X_train.shape[0], n_train, replace=False)
    idx_te = rng.choice(X_test.shape[0], n_test_eval, replace=False)
    Xt = X_train[idx_tr].reshape(n_train, 1, 28, 28)
    yt = y_train[idx_tr]
    Xv = X_test[idx_te].reshape(n_test_eval, 1, 28, 28)
    yv = y_test[idx_te]
    model = CNN()
    lr = 0.1
    epochs = 3
    batch = 64
    for e in range(epochs):
        perm = rng.permutation(n_train)
        total = 0.0
        for s in range(0, n_train, batch):
            b = perm[s:s + batch]
            xb = Xt[b]
            yb = yt[b]
            logits = model.forward(xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            grads = model.backward(dlogits)
            model.step(grads, lr)
            total += loss * len(b)
        print("CNN epoch", e + 1, "loss", float(total / n_train))
    
    pred_cnn = model.forward(Xv).argmax(axis=1)
    acc_cnn = accuracy(yv, pred_cnn)
    
    idx_lin = idx_tr
    X_lin = X_train[idx_lin]
    y_lin = y_train[idx_lin]
    X_te = X_test[idx_te]
    y_te = y_test[idx_te]
    lin = LinearSVM(num_classes=10, reg=1e-4, lr=0.3, epochs=8, batch_size=512, seed=1)
    lin.fit(X_lin, y_lin)
    y_pred_lin = lin.predict(X_te)
    acc_lin = accuracy(y_te, y_pred_lin)
    print("CNN accuracy:", float(acc_cnn))
    print("Linear SVM accuracy:", float(acc_lin))
    print("Elapsed seconds:", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()