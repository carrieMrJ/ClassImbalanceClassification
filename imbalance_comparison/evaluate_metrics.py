import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import math
import numpy as np
from imblearn.metrics import geometric_mean_score


def g_mean(r, tnr):
    return math.sqrt((r * tnr))


def get_precision(c_m):
    return c_m[1][1] / (c_m[1][1] + c_m[0][1])


def get_fbeta_measure(pre, beta, r):
    return (1 + beta * beta) * pre * r / (beta * beta * pre + r)


def get_tnr(c_m):
    return c_m[0][0] / (c_m[0][0] + c_m[0][1])


def plot_confusion_matrix(y_true, y_pred, labels):
    sns.set()
    f, ax = plt.subplots()
    c_bin = metrics.confusion_matrix(np.array(y_true), y_pred, labels=labels)
    print(c_bin)  # 打印出来看看
    # sns.heatmap(c_bin, annot=True, ax=ax)  # 画热力图
    #
    # ax.set_title('confusion matrix')  # 标题
    # ax.set_xlabel('predict')  # x轴
    # ax.set_ylabel('true')  # y轴
    #
    # plt.show()

    return c_bin


def ev_me(Y_test_sm, predict, labels):
    cm_sm = plot_confusion_matrix(Y_test_sm, predict, labels)
    recall_sm = metrics.recall_score(Y_test_sm, predict, pos_label=labels[1])
    precision_sm = metrics.precision_score(Y_test_sm, predict, pos_label=labels[1])
    # tnr_sm = get_tnr(cm_sm)
    # accuracy_score_sm = metrics.accuracy_score(Y_test_sm, predict)
    g_mean_sm = geometric_mean_score(Y_test_sm, predict, pos_label=labels[1])

    # print('tnr', tnr_sm)
    print('recall', recall_sm)
    print('precision', precision_sm)
    # print('accuracy', accuracy_score_sm)
    # print('auc', roc_auc)
    print('g_mean', g_mean_sm)
    print('f1', metrics.f1_score(Y_test_sm, predict, pos_label=labels[1]))



def draw_roc_c(Y_test, y_score, pos_label, alg_name):
    y_score = np.array(y_score)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_score[:, -1], pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    # print(fpr)
    # print(tpr)
    # print(threshold)
    print('AUC',roc_auc)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), linewidth=2, linestyle='-', color='b',
             marker='o')
    plt.fill_between(fpr, y1=tpr, y2=0, step=None, alpha=0.2, color='b')
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve:' + alg_name)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    plt.show()

    # return fig


def draw_apr_c(Y_test_rus, y_score_rus, pos_label, alg):
    y_score_rus = np.array(y_score_rus)
    precision_rus, recall_rus, threshold_rus = metrics.precision_recall_curve(Y_test_rus, y_score_rus[:, -1],
                                                                              pos_label=pos_label)
    # print(recall_rus)
    # print(precision_rus)
    # print('threshold')
    # print(threshold_rus)

    pr_auc_rus = metrics.auc(recall_rus, precision_rus)  # 梯形块分割，建议使用
    pr_auc0_rus = metrics.average_precision_score(Y_test_rus, y_score_rus[:, -1], average='weighted',
                                                  pos_label=pos_label)  # 小矩形块分割

    # print('pr_auc', pr_auc_rus)
    print('AUCPRC', pr_auc0_rus)

    # ======================= PLoting =============================
    plt.figure(1)
    lines = []
    labels = []
    l, = plt.plot(recall_rus, precision_rus,
             linewidth=2, linestyle='-', color='r', marker='o')
    lines.append(l)
    labels.append('Precision-recall (area = {0:0.4f})'.format(pr_auc0_rus))
    plt.fill_between(recall_rus, y1=precision_rus, y2=0, step=None, alpha=0.2, color='b')
    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.title("PR-Curve:" + alg)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    fig = plt.gcf()
    plt.show()

    return fig