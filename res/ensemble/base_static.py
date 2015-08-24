# encoding=utf-8
__author__ = 'Fule Liu'


import math
import os


def base_statics_cv5(fold_name):
    """统计5份交叉验证的平均sn，sp，MCC，ACC。
    """
    files = os.listdir(fold_name)
    ave_sn = 0
    ave_sp = 0
    ave_mcc = 0
    ave_acc = 0
    for file_name in files:
        file_path = fold_name + file_name
        print(file_path)
        sn, sp, mcc, acc = calc_acc(file_path)
        ave_sn += sn
        ave_sp += sp
        ave_mcc += mcc
        ave_acc += acc
        print(sn, sp, mcc, acc)

    ave_sn /= 5
    ave_sp /= 5
    ave_mcc /= 5
    ave_acc /= 5
    print(ave_sn, ave_sp, ave_mcc, ave_acc)


def calc_acc(file_name):
    with open(file_name) as fp:
        lines = fp.readlines()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for idx, line in enumerate(lines):
        line = line.rstrip().split()
        label = line[0]
        # print(pos_prob)
        if idx < 56:
            if label == '1':
                tp += 1
            else:
                fp += 1
        else:
            if label == '-1':
                tn += 1
            else:
                fn += 1

    print(tp, fp, tn, fn)
    sn = tp * 1.0 / (tp + fn)
    sp = tn * 1.0 / (tn + fp)
    mcc = (tp * tn - fp * fn) * 1.0 / math.sqrt((tp + fn) * (tn + fn) * (tp + fp) * (tn + fp))
    acc = (tp + tn) * 1.0 / (tp + fn + tn + fp)

    return sn, sp, mcc, acc


if __name__ == "__main__":
    base_statics_cv5("../cv_psednc_article_predict/")
