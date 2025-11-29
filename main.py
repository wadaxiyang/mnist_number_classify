import os
import time
import numpy as np
from mySVM import OVRKernelSVM, LinearSVM
from Load_data import load_mnist
from myCNN import softmax_cross_entropy, CNN
from digit_gui import DigitGUI
from Evaluate_model import plot_roc_curves, accuracy, precision_recall_f1_macro, softmax,auc_macro_ovr




# 调用训练CNN模型
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

# 主函数
def main(num_lin, num_ker, num_cnn):
    data_dir = os.path.join(os.path.dirname(__file__), "data", "mnist")
    
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    rng = np.random.default_rng(0)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print("data loaded over!")

    
    
    # 训练线性SVM模型
    t0 = time.time()
    print("training linear SVM model...")
    idx_lin = rng.choice(X_train.shape[0], num_lin, replace=False)
    X_lin = X_train[idx_lin]
    y_lin = y_train[idx_lin]
    lin = LinearSVM(num_classes=10, reg=1e-4, lr=0.3, epochs=12, batch_size=512, seed=1)
    lin.fit(X_lin, y_lin)
    print("train seconds:", round(time.time() - t0, 2),"s")

    # 训练核SVM模型
    t0 = time.time()
    print("training kernel SVM model...")
    idx_ker = rng.choice(X_train.shape[0], num_ker, replace=False)
    X_k = X_train[idx_ker]
    y_k = y_train[idx_ker]
    ker = OVRKernelSVM(num_classes=10, C=2.0, gamma=0.05, tol=1e-3, max_passes=2)
    ker.fit(X_k, y_k)
    print("train seconds:", round(time.time() - t0, 2),"s")

    # 训练CNN模型
    t0 = time.time()
    print("training CNN model...")
    idx_cnn = rng.choice(X_train.shape[0], num_cnn, replace=False)
    X_t_cnn = X_train[idx_cnn].reshape(num_cnn, 1, 28, 28)
    y_t_cnn = y_train[idx_cnn]
    cnn = train_CNN(X_t_cnn, y_t_cnn, num_cnn, rng)
    print("train seconds:", round(time.time() - t0, 2),"s")


    # 生成测试数据
    n_test_eval = 10000
    idx_te = rng.choice(X_test.shape[0], n_test_eval, replace=False)
    X_te = X_test[idx_te]
    y_te = y_test[idx_te]
    print("y_te",y_te[10:20])

    X_test_cnn = X_te.reshape(n_test_eval, 1, 28, 28)
    y_test_cnn = y_te


    ## 评估模型
    print("evaluating models...")
    # 评估线性SVM模型
    y_pred_lin = lin.predict(X_te)
    acc_lin = accuracy(y_te, y_pred_lin)
    print("y_pred_lin",y_pred_lin[10:20])
    # 评估核SVM模型
    y_pred_ker = ker.predict(X_te)
    acc_ker = accuracy(y_te, y_pred_ker)
    print("y_pred_ker",y_pred_ker[10:20])

    # 评估CNN模型   
    y_pred_cnn = cnn.forward(X_test_cnn).argmax(axis=1)
    acc_cnn = accuracy(y_test_cnn, y_pred_cnn)
    print("y_pred_cnn",y_pred_cnn[10:20])
    print("evaluate result:")
    print("Linear SVM accuracy:", float(acc_lin))
    print("Kernel SVM accuracy:", float(acc_ker))
    print("CNN accuracy:", float(acc_cnn))

    scores_lin = X_te @ lin.W
    pr_lin, rc_lin, f1_lin = precision_recall_f1_macro(y_te, y_pred_lin)
    auc_lin = auc_macro_ovr(y_te, scores_lin)

    S_ker = np.vstack([m.decision_function(X_te) for m in ker.models]).T
    pr_ker, rc_ker, f1_ker = precision_recall_f1_macro(y_te, y_pred_ker)
    auc_ker = auc_macro_ovr(y_te, S_ker)
    
    logits_cnn = cnn.forward(X_test_cnn)
    probs_cnn = softmax(logits_cnn)
    pr_cnn, rc_cnn, f1_cnn = precision_recall_f1_macro(y_test_cnn, y_pred_cnn)
    auc_cnn = auc_macro_ovr(y_test_cnn, probs_cnn)
    print("Linear SVM precision:", pr_lin)
    print("Linear SVM recall:", rc_lin)
    print("Linear SVM F1:", f1_lin)
    print("Linear SVM AUC:", auc_lin)
    print("Kernel SVM precision:", pr_ker)
    print("Kernel SVM recall:", rc_ker)
    print("Kernel SVM F1:", f1_ker)
    print("Kernel SVM AUC:", auc_ker)
    print("CNN precision:", pr_cnn)
    print("CNN recall:", rc_cnn)
    print("CNN F1:", f1_cnn)
    print("CNN AUC:", auc_cnn)
    plot_roc_curves(y_te, scores_lin, S_ker, probs_cnn)



if __name__ == "__main__":
    print("please select the experiment:")
    print("1. experiment 1")
    print("2. experiment 2")
    sel = int(input("your selection: "))
    if sel == 1:
        # 进行实验1
        main(
        num_lin=20000, 
        num_ker=4000, 
        num_cnn=8000)
    elif sel == 2:
        # 进行实验2
        main( 
        num_lin=10000, 
        num_ker=10000, 
        num_cnn=10000)
    else:
        print("invalid selection")
