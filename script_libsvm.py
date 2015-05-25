__author__ = 'Fule Liu'

import subprocess
import time
import os
import logging


SVM_SCALE = "libsvm/svm-scale.exe"
SVM_TRAIN = "libsvm/svm-train.exe"
SVM_PREDICT = "libsvm/svm-predict.exe"


def main_revc_kmer():
    cmd = " ".join(["python", "easy.py", "data/revc/res.txt"])
    print(cmd)
    subprocess.Popen(cmd, shell=True).wait()


def main_revc_kmer_loo():
    pass


def main_psednc():
    listdir = os.listdir("data/pseudo")
    for cur_dir in listdir:
        cmd = " ".join(["python", "easy.py", "data/pseudo/" + cur_dir])
        print(cmd)
        subprocess.Popen(cmd, shell=True).wait()


def main_revc_kmer_psednc():
    listdir = os.listdir("data/revc_psednc")
    for cur_dir in listdir:
        cmd = " ".join(["python", "easy.py", "data/revc_psednc/" + cur_dir])
        print(cmd)
        subprocess.Popen(cmd, shell=True).wait()


def main_borderline_revc_kmer():
    cmd = " ".join(["python", "easy.py", "data/borderline_revc_kmer/borderline_smote_revc_kmer.txt"])
    print(cmd)
    subprocess.Popen(cmd, shell=True).wait()


def main_borderline_revc_psednc():
    cmd = " ".join(["python", "easy.py", "data/borderline_revc_kmer_psednc/6_8"])
    print(cmd)
    subprocess.Popen(cmd, shell=True).wait()


def main_cv5_smote_revc_psednc(fold_path):
    n_lamada = "6_0.8"
    for i in range(5):
        train_file = fold_path + "train_" + n_lamada + "_" + str(i) + ".txt"
        test_file = fold_path + "test_" + n_lamada + "_" + str(i) + ".txt"
        cmd = " ".join(["python", "easy.py", ""])


if __name__ == "__main__":
    print("Computing...")
    start_time = time.time()

    # main_revc_kmer()
    # main_psednc()
    # main_revc_kmer_psednc()
    # main_borderline_revc_kmer()
    # main_borderline_revc_psednc()

    main_cv5_smote_revc_psednc("data/cv5/")

    print("End, used time: %s s." % (time.time() - start_time))
