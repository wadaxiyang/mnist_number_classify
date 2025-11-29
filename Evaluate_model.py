# 评估模型工具
import matplotlib.pyplot as plt
import numpy as np


# 计算准确率
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

# 计算宏F1分数
def precision_recall_f1_macro(y_true, y_pred):
    K = int(max(int(y_true.max()), int(y_pred.max())) + 1)
    p_list = []
    r_list = []
    f_list = []
    for c in range(K):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    return float(np.mean(p_list)), float(np.mean(r_list)), float(np.mean(f_list))


# 计算宏AUC分数
def auc_macro_ovr(y_true, scores):
    n = scores.shape[0]
    K = scores.shape[1]
    aucs = []
    for c in range(K):
        yb = (y_true == c).astype(np.int64)
        pos = int(yb.sum())
        neg = n - pos
        if pos == 0 or neg == 0:
            continue
        s = scores[:, c]
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        sum_pos = ranks[yb == 1].sum()
        U = sum_pos - pos * (pos + 1) / 2.0
        auc = U / (pos * neg)
        aucs.append(auc)
    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


def softmax(x):
    a = x.max(axis=1, keepdims=True)
    e = np.exp(x - a)
    return e / e.sum(axis=1, keepdims=True)

# 计算微AUC分数
def roc_curve_micro(y_true, scores):
    K = scores.shape[1]
    Y = (np.arange(K)[None, :] == y_true[:, None]).astype(np.int64)
    s = scores.reshape(-1)
    y = Y.reshape(-1)
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    P = y.sum()
    N = y.shape[0] - P
    tpr = tp / (P + 1e-12)
    fpr = fp / (N + 1e-12)
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return fpr, tpr

# 绘制ROC曲线
def plot_roc_curves(y_true, scores_lin, scores_ker, probs_cnn):
    fpr_lin, tpr_lin = roc_curve_micro(y_true, scores_lin)
    fpr_ker, tpr_ker = roc_curve_micro(y_true, scores_ker)
    fpr_cnn, tpr_cnn = roc_curve_micro(y_true, probs_cnn)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_lin, tpr_lin, label="Linear SVM")
    plt.plot(fpr_ker, tpr_ker, label="Kernel SVM")
    plt.plot(fpr_cnn, tpr_cnn, label="CNN")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (micro-average)")
    plt.legend()
    plt.grid(True)
    plt.show()
