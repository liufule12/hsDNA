__author__ = 'Fule Liu'


import math
import os


def statics(fold_name):
    files = os.listdir(fold_name)
    for file_name in files:
        file_path = fold_name + file_name
        print(file_path)
        sn, sp, mcc, acc = calc_acc(file_path)
        print(sn, sp, mcc, acc)


def calc_acc(file_name):
    with open(file_name) as fp:
        lines = fp.readlines()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for idx, line in enumerate(lines):
        line = line.rstrip().split()
        pos_prob = float(line[0])
        # print(pos_prob)
        if idx < 280:
            if pos_prob >= 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if pos_prob < 0.5:
                tn += 1
            else:
                fp += 1

    print(tp, fp, tn, fn)
    sn = tp * 1.0 / (tp + fn)
    sp = tn * 1.0 / (tn + fp)
    mcc = (tp * tn - fp * fn) * 1.0 / math.sqrt((tp + fn) * (tn + fn) * (tp + fp) * (tn + fp))
    acc = (tp + tn) * 1.0 / (tp + fn + tn + fp)

    return sn, sp, mcc, acc


if __name__ == "__main__":
    statics("E:/ensemble_output/")
