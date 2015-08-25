# encoding=utf-8
__author__ = 'Fule Liu'

import os

from repDNA.nac import Kmer, RevcKmer
from repDNA.ac import DACC
from repDNA.psenac import PseDNC
from repDNA.util import write_libsvm


def auto_cv5_kmer_tool(k, write_fold):
    for i in range(5):
        test_neg_file = "data/cv5/test_neg_" + str(i)
        test_pos_file = "data/cv5/test_pos_" + str(i)
        train_neg_file = "data/cv5/train_neg_" + str(i)
        train_pos_file = "data/cv5/train_pos_" + str(i)
        test_write_file = write_fold + "cv5_kmer_test_" + str(i)
        train_write_file = write_fold + "cv5_kmer_train_" + str(i)
        cv5_kmer_tool(k=k, test_neg_file=test_neg_file, test_pos_file=test_pos_file,
                      train_neg_file=train_neg_file, train_pos_file=train_pos_file,
                      test_write_file=test_write_file, train_write_file=train_write_file)


def auto_cv5_upto_revckmer_tool(k, write_fold):
    for i in range(5):
        test_neg_file = "data/cv5/test_neg_" + str(i)
        test_pos_file = "data/cv5/test_pos_" + str(i)
        train_neg_file = "data/cv5/train_neg_" + str(i)
        train_pos_file = "data/cv5/train_pos_" + str(i)
        test_write_file = write_fold + "cv5_kmer_test_" + str(i)
        train_write_file = write_fold + "cv5_kmer_train_" + str(i)
        cv5_upto_revckmer_tool(k=k, test_neg_file=test_neg_file, test_pos_file=test_pos_file,
                               train_neg_file=train_neg_file, train_pos_file=train_pos_file,
                               test_write_file=test_write_file, train_write_file=train_write_file)


def auto_cv5_dacc_tool(lag, write_file_prefix):
    for i in range(5):
        print(i)
        test_neg_file = "data/cv5/test_neg_" + str(i)
        test_pos_file = "data/cv5/test_pos_" + str(i)
        train_neg_file = "data/cv5/train_neg_" + str(i)
        train_pos_file = "data/cv5/train_pos_" + str(i)
        test_write_file = write_file_prefix + "_test_" + str(i)
        train_write_file = write_file_prefix + "_train_" + str(i)
        cv5_dacc_tool(lag=lag, test_neg_file=test_neg_file, test_pos_file=test_pos_file,
                      train_neg_file=train_neg_file, train_pos_file=train_pos_file,
                      test_write_file=test_write_file, train_write_file=train_write_file)


def auto_cv5_psednc_tool(lamada, w, write_file_prefix):
    for i in range(5):
        print(i)
        test_neg_file = "data/cv5/test_neg_" + str(i)
        test_pos_file = "data/cv5/test_pos_" + str(i)
        train_neg_file = "data/cv5/train_neg_" + str(i)
        train_pos_file = "data/cv5/train_pos_" + str(i)
        test_write_file = write_file_prefix + "_test_" + str(i)
        train_write_file = write_file_prefix + "_train_" + str(i)
        cv5_psednc_tool(lamada=lamada, w=w, test_neg_file=test_neg_file, test_pos_file=test_pos_file,
                        train_neg_file=train_neg_file, train_pos_file=train_pos_file,
                        test_write_file=test_write_file, train_write_file=train_write_file)


def cv5_kmer_tool(k, test_neg_file, test_pos_file, train_neg_file, train_pos_file, test_write_file, train_write_file):
    kmer = Kmer(k=k, normalize=True)
    with open(test_neg_file) as fp:
        test_neg_vecs = kmer.make_kmer_vec(fp)
    with open(test_pos_file) as fp:
        test_pos_vecs = kmer.make_kmer_vec(fp)
    with open(train_pos_file) as fp:
        train_pos_vecs = kmer.make_kmer_vec(fp)
    with open(train_neg_file) as fp:
        train_neg_vecs = kmer.make_kmer_vec(fp)

    train_vecs = train_pos_vecs + train_neg_vecs
    test_vecs = test_pos_vecs + test_neg_vecs
    train_labels = [1] * len(train_pos_vecs) + [-1] * len(train_neg_vecs)
    test_labels = [1] * len(test_pos_vecs) + [-1] * len(test_neg_vecs)

    # Write file.
    write_libsvm(train_vecs, train_labels, train_write_file)
    write_libsvm(test_vecs, test_labels, test_write_file)


def cv5_upto_revckmer_tool(k, test_neg_file, test_pos_file, train_neg_file, train_pos_file, test_write_file,
                           train_write_file):
    kmer = RevcKmer(k=k, upto=True, normalize=True)
    with open(test_neg_file) as fp:
        test_neg_vecs = kmer.make_revckmer_vec(fp)
    with open(test_pos_file) as fp:
        test_pos_vecs = kmer.make_revckmer_vec(fp)
    with open(train_pos_file) as fp:
        train_pos_vecs = kmer.make_revckmer_vec(fp)
    with open(train_neg_file) as fp:
        train_neg_vecs = kmer.make_revckmer_vec(fp)

    train_vecs = train_pos_vecs + train_neg_vecs
    test_vecs = test_pos_vecs + test_neg_vecs
    train_labels = [1] * len(train_pos_vecs) + [-1] * len(train_neg_vecs)
    test_labels = [1] * len(test_pos_vecs) + [-1] * len(test_neg_vecs)

    # Write file.
    write_libsvm(train_vecs, train_labels, train_write_file)
    write_libsvm(test_vecs, test_labels, test_write_file)


def cv5_dacc_tool(lag, test_neg_file, test_pos_file, train_neg_file, train_pos_file, test_write_file, train_write_file):
    dacc = DACC(lag=lag)
    with open(test_neg_file) as fp:
        test_neg_vecs = dacc.make_dacc_vec(fp, phyche_index=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    with open(test_pos_file) as fp:
        test_pos_vecs = dacc.make_dacc_vec(fp, phyche_index=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    with open(train_pos_file) as fp:
        train_pos_vecs = dacc.make_dacc_vec(fp, phyche_index=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    with open(train_neg_file) as fp:
        train_neg_vecs = dacc.make_dacc_vec(fp, phyche_index=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])

    train_vecs = train_pos_vecs + train_neg_vecs
    test_vecs = test_pos_vecs + test_neg_vecs
    train_labels = [1] * len(train_pos_vecs) + [-1] * len(train_neg_vecs)
    test_labels = [1] * len(test_pos_vecs) + [-1] * len(test_neg_vecs)

    # Write file.
    write_libsvm(train_vecs, train_labels, train_write_file)
    write_libsvm(test_vecs, test_labels, test_write_file)


def cv5_psednc_tool(lamada, w, test_neg_file, test_pos_file, train_neg_file, train_pos_file, test_write_file,
                    train_write_file):
    psednc = PseDNC(lamada=lamada, w=w)
    with open(test_neg_file) as fp:
        test_neg_vecs = psednc.make_psednc_vec(fp)
    with open(test_pos_file) as fp:
        test_pos_vecs = psednc.make_psednc_vec(fp)
    with open(train_pos_file) as fp:
        train_pos_vecs = psednc.make_psednc_vec(fp)
    with open(train_neg_file) as fp:
        train_neg_vecs = psednc.make_psednc_vec(fp)

    train_vecs = train_pos_vecs + train_neg_vecs
    test_vecs = test_pos_vecs + test_neg_vecs
    train_labels = [1] * len(train_pos_vecs) + [-1] * len(train_neg_vecs)
    test_labels = [1] * len(test_pos_vecs) + [-1] * len(test_neg_vecs)

    # Write file.
    write_libsvm(train_vecs, train_labels, train_write_file)
    write_libsvm(test_vecs, test_labels, test_write_file)


def kmer_tool(k, pos_file, neg_file, write_file):
    kmer = Kmer(k=k, normalize=True)
    with open(pos_file) as fp:
        pos_vecs = kmer.make_kmer_vec(fp)
    with open(neg_file) as fp:
        neg_vecs = kmer.make_kmer_vec(fp)

    vecs = pos_vecs + neg_vecs
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

    # Write file.
    write_libsvm(vecs, labels, write_file)


def dacc_tool(lag, pos_file, neg_file, write_file):
    dacc = DACC(lag=lag)
    with open(pos_file) as fp:
        pos_vecs = dacc.make_dacc_vec(fp, phyche_index=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    # print(pos_vecs)
    with open(neg_file) as fp:
        neg_vecs = dacc.make_dacc_vec(fp, phyche_index=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])

    vecs = pos_vecs + neg_vecs
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

    # Write file.
    write_libsvm(vecs, labels, write_file)


def psednc_tool(lamada, w, pos_file, neg_file, write_file):
    psednc = PseDNC(lamada=lamada, w=w)
    with open(pos_file) as fp:
        pos_vecs = psednc.make_psednc_vec(fp)
    with open(neg_file) as fp:
        neg_vecs = psednc.make_psednc_vec(fp)

    vecs = pos_vecs + neg_vecs
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

    # Write file.
    write_libsvm(vecs, labels, write_file)


def cv5_libsvm(c, g, train_prefix, test_prefix):
    for i in range(5):
        # Train. Usage: svm-predict [options] test_file model_file output_file
        cmd = "libsvm\svm-train.exe -c " + str(c) + " -g " + str(g) + " -b 1 " + \
              train_prefix + "_train_" + str(i) + " " + train_prefix + "_train_" + str(i) + ".model"
        print(cmd)
        os.system(cmd)

        # Test. Usage: svm-predict [options] test_file model_file output_file
        cmd = "libsvm\svm-predict.exe -b 1 " + test_prefix + "_test_" + str(i) + " " + \
              train_prefix + "_train_" + str(i) + ".model" + " " + test_prefix + "_test_" + str(i) + ".predict"
        print(cmd)
        os.system(cmd)

    pass


if __name__ == "__main__":
    # kmer_tool(k=2, pos_file="data/hs.fasta", neg_file="data/non-hs.fasta", write_file="res/kmer_2")
    # psednc_tool(lamada=3, w=0.2, pos_file="data/hs.fasta", neg_file="data/non-hs.fasta", write_file="res/psednc_3_0.2")
    # dacc_tool(lag=1, pos_file="data/hs.fasta", neg_file="data/non-hs.fasta", write_file="res/dacc_1")

    # cv5_kmer_tool(k=2, test_neg_file="data/cv5/test_neg_0", test_pos_file="data/cv5/test_pos_0",
    # train_neg_file="data/cv5/train_neg_0", train_pos_file="data/cv5/train_pos_0",
    # test_write_file="res/cv5_kmer_test", train_write_file="res/cv5_kmer_train")

    # cv5_dacc_tool(lag=1, test_neg_file="data/cv5/test_neg_0", test_pos_file="data/cv5/test_pos_0",
    # train_neg_file="data/cv5/train_neg_0", train_pos_file="data/cv5/train_pos_0",
    # test_write_file="res/cv5_dacc_test", train_write_file="res/cv5_dacc_train")

    # cv5_psednc_tool(lamada=3, w=0.2, test_neg_file="data/cv5/test_neg_0", test_pos_file="data/cv5/test_pos_0",
    # train_neg_file="data/cv5/train_neg_0", train_pos_file="data/cv5/train_pos_0",
    # test_write_file="res/cv5_psednc_test", train_write_file="res/cv5_psednc_train")

    # 各方法最优参数生成特征向量
    # auto_cv5_kmer_tool(k=2, write_file_prefix="res/cv5_kmer/cv5_kmer")
    # auto_cv5_dacc_tool(lag=1, write_file_prefix="res/cv5_dacc/cv5_dacc")
    # auto_cv5_psednc_tool(lamada=3, w=0.2, write_file_prefix="res/cv5_psednc/cv5_psednc")

    # 各特征向量五份交叉验证。
    # cv5_libsvm(c=512, g=2, train_prefix="res/cv5_kmer/cv5_kmer", test_prefix="res/cv5_kmer/cv5_kmer")
    # cv5_libsvm(c=32768, g=0.0078125, train_prefix="res/cv5_dacc/cv5_dacc", test_prefix="res/cv5_dacc/cv5_dacc")
    # cv5_libsvm(c=2048, g=2, train_prefix="res/cv5_psednc/cv5_psednc", test_prefix="res/cv5_psednc/cv5_psednc")

    # 论文PseDNC最优参数生成特征向量并5份交叉验证。
    # auto_cv5_psednc_tool(lamada=6, w=0.2, write_file_prefix="res/cv5_psednc_article/cv5_psednc")
    # cv5_libsvm(c=512, g=0.0078125, train_prefix="res/cv5_psednc_article/cv5_psednc",
    #            test_prefix="res/cv5_psednc_article/cv5_psednc")

    # 论文kmer最优参数生成特征向量并5份交叉验证。
    # auto_cv5_upto_revckmer_tool(k=6, write_fold="res/cv5_revckmer_article/cv5_kmer")
    # cv5_libsvm(c=2, g=0.001953125, train_prefix="res/cv5_revckmer_article/cv5_kmercv5_kmer",
    #            test_prefix="res/cv5_revckmer_article/cv5_kmercv5_kmer")

    pass