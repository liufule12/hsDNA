__author__ = 'Fule Liu'

import operator
import math


def calculate_acc(fold_path, filename_prefix):
    """Return Sn, Sp, MCC, Acc."""
    sn = 0
    sp = 0
    mcc = 0
    acc = 0
    for i in range(5):
        test_file_path = fold_path + filename_prefix + "_test_" + str(i) + ".txt"
        predict_file_path = test_file_path + ".predict"
        with open(test_file_path) as fp:
            test_lines = fp.readlines()
        with open(predict_file_path) as fp:
            predict_lines = fp.readlines()[1:]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for index, predict_line in enumerate(predict_lines):
            if predict_line[0] == '1' and test_lines[index][0] == '1':
                tp += 1
            elif predict_line[0] == '1' and test_lines[index][0] == '-':
                fp += 1
            elif predict_line[0] == '-' and test_lines[index][0] == '-':
                tn += 1
            elif predict_line[0] == '-' and test_lines[index][0] == '1':
                fn += 1
            else:
                raise ValueError("The first element is not '1' or '-'")

        print(tp, fp, tn, fn)
        sn += tp * 1.0 / (tp + fn)
        sp += tn * 1.0 / (tn + fp)
        mcc += (tp * tn - fp * fn) * 1.0 / math.sqrt((tp + fn) * (tn + fn) * (tp + fp) * (tn + fp))
        acc += (tp + tn) * 1.0 / (tp + fn + tn + fp)

    return sn / 5, sp / 5, mcc / 5, acc / 5


def calculate_roc_auc(fold_path, filename_prefix):
    """Calculate auc of ROC curve."""
    average_auc_roc = 0
    for i in range(5):
        test_file_path = fold_path + filename_prefix + "_test_" + str(i) + ".txt"
        predict_file_path = test_file_path + ".predict"
        with open(test_file_path) as fp:
            test_lines = fp.readlines()
        with open(predict_file_path) as fp:
            predict_lines = fp.readlines()[1:]

        label_prob = \
            [(test_lines[index].split()[0], predict_line.split()[1]) for index, predict_line in enumerate(predict_lines)]
        sorted_label_prob = sorted(label_prob, key=operator.itemgetter(1), reverse=True)

        tp = 0
        fp = 0
        auc_roc = 0
        for label, prob in sorted_label_prob:
            if label == "1":
                tp += 1
            else:
                fp += 1
                auc_roc += tp

        if tp == 0:
            auc_roc = 0
        elif fp == 0:
            auc_roc = 1
        else:
            auc_roc /= (tp * fp * 1.0)

        average_auc_roc += auc_roc

    average_auc_roc /= 5
    return average_auc_roc


if __name__ == "__main__":
    # CV5 RevcKmer.
    # print(calculate_acc(fold_path="data/cv5_revc_kmer/", filename_prefix="cv5_revc_kmer"))
    # print(calculate_roc_auc(fold_path="data/cv5_revc_kmer/", filename_prefix="cv5_revc_kmer"))

    # CV5 borderline SMOTE RevcKmer.
    print(calculate_acc(fold_path="data/cv5_borderline_revc_kmer/", filename_prefix="cv5_borderline_revc_kmer"))
    print(calculate_roc_auc(fold_path="data/cv5_borderline_revc_kmer/", filename_prefix="cv5_borderline_revc_kmer"))

    # CV5 PseDNC.
    # print(calculate_acc(fold_path="data/cv5_psednc6_0.2/", filename_prefix="cv5_psednc_6_0.2"))
    # print(calculate_roc_auc(fold_path="data/cv5_psednc6_0.2/", filename_prefix="cv5_psednc_6_0.2"))

    # CV5 SMOTE RevcKmer_PseDNC.
    # print(calculate_acc(fold_path="data/cv5_revc_psednc_smote_6_0.8/", filename_prefix="cv5_smote_revc_psednc_6_0.8"))
    # print(calculate_roc_auc(fold_path="data/cv5_revc_psednc_smote_6_0.8/",
    #                         filename_prefix="cv5_smote_revc_psednc_6_0.8"))

    # CV5 borderline_SMOTE RevcKmer_PseDNC.
    print(calculate_acc(fold_path="data/cv5_borderline_revc_kmer_psednc/",
                        filename_prefix="cv5_smote_borderline_6_0.8"))
    print(calculate_roc_auc(fold_path="data/cv5_borderline_revc_kmer_psednc/",
                            filename_prefix="cv5_smote_borderline_6_0.8"))

    # CV5 psednc_article.
    print()