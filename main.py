import os
import time
import numpy as np
from mySVM import OVRKernelSVM, LinearSVM
from Load_data import load_mnist
from myCNN import softmax_cross_entropy, CNN

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def train_CNN(Xt, yt, n_train, rng, lr: float=0.1, epochs: int=8, batch: int=64):
    model = CNN()
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
    
    return model


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "mnist")
    
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    rng = np.random.default_rng(0)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print("data loaded over!")

    # 训练线性SVM模型
    t0 = time.time()
    print("training linear SVM model...")
    n_train_linear = 20000
    idx_lin = rng.choice(X_train.shape[0], n_train_linear, replace=False)
    X_lin = X_train[idx_lin]
    y_lin = y_train[idx_lin]
    lin = LinearSVM(num_classes=10, reg=1e-4, lr=0.3, epochs=12, batch_size=512, seed=1)
    lin.fit(X_lin, y_lin)
    print("train seconds:", round(time.time() - t0, 2),"s")

    # 训练核SVM模型
    t0 = time.time()
    print("training kernel SVM model...")
    n_train_kernel = 3000
    idx_k = rng.choice(X_train.shape[0], n_train_kernel, replace=False)  
    X_k = X_train[idx_k]
    y_k = y_train[idx_k]
    ker = OVRKernelSVM(num_classes=10, C=2.0, gamma=0.05, tol=1e-3, max_passes=2)
    ker.fit(X_k, y_k)
    print("train seconds:", round(time.time() - t0, 2),"s")

    # 训练CNN模型
    t0 = time.time()
    print("training CNN model...")
    n_train_cnn = 6000
    idx_cnn = rng.choice(X_train.shape[0], n_train_cnn, replace=False)
    X_t_cnn = X_train[idx_cnn].reshape(n_train_cnn, 1, 28, 28)
    y_t_cnn = y_train[idx_cnn]
    cnn = train_CNN(X_t_cnn, y_t_cnn, n_train_cnn, rng)
    print("train seconds:", round(time.time() - t0, 2),"s")


    # 生成测试数据
    n_test_eval = 10000
    idx_te = rng.choice(X_test.shape[0], n_test_eval, replace=False)
    X_te = X_test[idx_te]
    y_te = y_test[idx_te]

    X_test_cnn = X_te.reshape(n_test_eval, 1, 28, 28)
    y_test_cnn = y_te


    ## 评估模型
    print("evaluating models...")
    # 评估线性SVM模型
    y_pred_lin = lin.predict(X_te)
    acc_lin = accuracy(y_te, y_pred_lin)
    
    # 评估核SVM模型
    y_pred_ker = ker.predict(X_te)
    acc_ker = accuracy(y_te, y_pred_ker)

    # 评估CNN模型   
    y_pred_cnn = cnn.forward(X_test_cnn).argmax(axis=1)
    acc_cnn = accuracy(y_test_cnn, y_pred_cnn)
    print("evaluate result:")
    print("Linear SVM accuracy:", float(acc_lin))
    print("Kernel SVM accuracy:", float(acc_ker))
    print("CNN accuracy:", float(acc_cnn))
    
if __name__ == "__main__":
    main()
