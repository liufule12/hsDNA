__author__ = 'Fule Liu'


import time
import numpy as np

import smote
from repDNA.nac import RevcKmer
from repDNA.psenac import PseDNC
from repDNA.util import get_data


def write_libsvm(vector_list, label_list, write_file):
    """Write the vectors into disk in livSVM format."""
    len_vector_list = len(vector_list)
    len_label_list = len(label_list)
    if len_vector_list == 0:
        raise ValueError("The vector is none.")
    if len_label_list == 0:
        raise ValueError("The label is none.")
    if len_vector_list != len_label_list:
        raise ValueError("The length of vector and label is different.")

    with open(write_file, 'wb') as f:
        for ind1, vec in enumerate(vector_list):
            write_line = str(label_list[ind1])
            for ind2, val in enumerate(vec):
                write_ind_val = ":".join([str(ind2+1), str(vec[ind2])])
                write_line = " ".join([write_line, write_ind_val])
            f.write(write_line)
            f.write('\n')


def whole_revc_kmer(pos_file, neg_file, k):
    """Generate revc_kmer into a file combined positive and negative file."""
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    with open(pos_file) as fp:
        pos_vecs = revc_kmer.make_revckmer_vec(fp)
    with open(neg_file) as fp:
        neg_vecs = revc_kmer.make_revckmer_vec(fp)
    vecs = pos_vecs + neg_vecs
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

    # Write file.
    write_file = "data/whole_revc_kmer.txt"
    write_libsvm(vecs, labels, write_file)


def whole_revc_kmer_psednc(pos_file, neg_file, k):
    """Generate revc_kmer and psednc feature into a file combined positive and negative file."""

    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    with open(pos_file) as fp:
        revc_kmer_pos_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
    with open(neg_file) as fp:
        revc_kmer_neg_vecs = np.array(revc_kmer.make_revckmer_vec(fp))

    lamada = 6
    w = 0.8
    psednc = PseDNC(lamada, w)
    with open(pos_file) as fp:
        psednc_pos_vecs = np.array(psednc.make_psednc_vec(fp))
    with open(neg_file) as fp:
        psednc_neg_vecs = np.array(psednc.make_psednc_vec(fp))

    pos_vecs = np.column_stack((revc_kmer_pos_vecs, psednc_pos_vecs[:, -lamada:]))
    neg_vecs = np.column_stack((revc_kmer_neg_vecs, psednc_neg_vecs[:, -lamada:]))
    vecs = pos_vecs.tolist() + neg_vecs.tolist()
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

    # Write file.
    write_file = "data/whole_revc_kmer_psednc.txt"
    write_libsvm(vecs, labels, write_file)


def whole_revc_kmer_psednc_choose_args(pos_file, neg_file, k):
    """Generate revc_kmer and psednc feature into a file combined positive and negative file."""

    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    with open(pos_file) as fp:
        revc_kmer_pos_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
    with open(neg_file) as fp:
        revc_kmer_neg_vecs = np.array(revc_kmer.make_revckmer_vec(fp))

    for lamada in range(1, 2):
        w = 0.1
        while w < 1:
            psednc = PseDNC(lamada, w)
            with open(pos_file) as fp:
                psednc_pos_vecs = np.array(psednc.make_psednc_vec(fp))
            with open(neg_file) as fp:
                psednc_neg_vecs = np.array(psednc.make_psednc_vec(fp))

            pos_vecs = np.column_stack((revc_kmer_pos_vecs, psednc_pos_vecs[:, -lamada:]))
            neg_vecs = np.column_stack((revc_kmer_neg_vecs, psednc_neg_vecs[:, -lamada:]))
            vecs = pos_vecs.tolist() + neg_vecs.tolist()
            labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

            # Write file.
            lamada_w = str(lamada) + '_' + str(w)
            write_file = "data/whole_revc_kmer_psednc_" + lamada_w + ".txt"
            print(write_file)
            write_libsvm(vecs, labels, write_file)

            w += 0.1


def cv5_revc_kmer(fold_path, filename, k):
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    for i in range(5):
        # Generate RevcKmer vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)

        # Write test file.
        write_file = fold_path + filename + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_revc_kmer_vecs + test_neg_revc_kmer_vecs
        test_vecs_labels = [1] * len(test_pos_revc_kmer_vecs) + [-1] * len(test_neg_revc_kmer_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + "_train_" + str(i) + ".txt"
        train_vecs = train_pos_revc_kmer_vecs + train_neg_revc_kmer_vecs
        train_vecs_labels = [1] * len(train_pos_revc_kmer_vecs) + [-1] * len(train_neg_revc_kmer_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


def borderline_smote_revc_kmer():
    revc_kmer = RevcKmer(k=6, normalize=True, upto=True)
    with open("data/hs.fasta") as f:
        pos_vecs = np.array(revc_kmer.make_revckmer_vec(f))
    with open("data/non-hs.fasta") as f:
        neg_vecs = np.array(revc_kmer.make_revckmer_vec(f))

    vecs = np.row_stack((pos_vecs, neg_vecs))
    vecs_labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)
    _1, synthetic1, _2 = smote.borderline_smote(vecs, vecs_labels, 1, N=300, k=5)

    pos_vecs = pos_vecs.tolist() + synthetic1.tolist()
    vecs = pos_vecs + neg_vecs.tolist()
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)
    write_libsvm(vecs, labels, "borderline_smote_revc_kmer.txt")


def borderline_smote_revc_psednc(fold_path):
    revc_kmer = RevcKmer(k=2, normalize=True, upto=True)
    with open("data/hs.fasta") as f:
        pos_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(f))
    with open("data/non-hs.fasta") as f:
        neg_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(f))

    lamada = 6
    w = 0.8
    psednc = PseDNC(lamada, w)
    with open("data/hs.fasta") as f:
        pos_psednc_vecs = np.array(psednc.make_psednc_vec(f))
    with open("data/non-hs.fasta") as f:
        neg_psednc_vecs = np.array(psednc.make_psednc_vec(f))

    pos_vecs = np.column_stack((pos_revc_kmer_vecs, pos_psednc_vecs[:, -lamada:]))
    neg_vecs = np.column_stack((neg_revc_kmer_vecs, neg_psednc_vecs[:, -lamada:]))
    vecs = np.row_stack((pos_vecs,  neg_vecs))
    vecs_labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)
    _1, synthetic, _2 = (smote.borderline_smote(vecs, vecs_labels, 1, N=300, k=5))
    pos_vecs = pos_vecs.tolist() + synthetic.tolist()
    vecs = pos_vecs + neg_vecs.tolist()
    labels = [1] * len(pos_vecs) + [-1] * len(neg_vecs)

    lamada_n = "_".join([str(lamada), str(w)])
    write_file = "/".join([fold_path, lamada_n])
    print(write_file)
    write_libsvm(vecs, labels, write_file)


def cv5():
    """Write 5 fold cross split.
    Because if others want to repeat this experiment, they need this split data.

    Return
    ------
    pos_cv, array, shape(5, num_every_fold)
    neg_cv, array, shape(5, num_every_fold)
    """
    # 5 fold cross split.
    with open("data/hs.fasta") as fp:
        pos = np.array(get_data(fp, desc=True))
    with open("data/non-hs.fasta") as fp:
        neg = np.array(get_data(fp, desc=True))
    len_pos = 280
    len_neg = 737
    pos_random = np.random.permutation(len_pos)
    neg_random = np.random.permutation(len_neg)
    pos_cv = [pos[pos_random[:len_pos:5]], pos[pos_random[1:len_pos:5]], pos[pos_random[2:len_pos:5]],
              pos[pos_random[3:len_pos:5]], pos[pos_random[4:len_pos:5]]]
    neg_cv = [neg[neg_random[:len_neg:5]], neg[neg_random[1:len_neg:5]], neg[neg_random[2:len_neg:5]],
              neg[neg_random[3:len_neg:5]], neg[neg_random[4:len_neg:5]]]

    # Write 5 fold file.
    for i in range(5):
        # Write test file.
        write_file = "data/cv5/test_pos_" + str(i)
        with open(write_file, 'w') as fp:
            for seq in pos_cv[i]:
                seq_desc = "".join([">", seq.name, "\n"])
                fp.write(seq_desc)
                fp.write(seq.seq)
                fp.write("\n")
                fp.write("\n")

        write_file = "data/cv5/test_neg_" + str(i)
        with open(write_file, 'w') as fp:
            for seq in neg_cv[i]:
                seq_desc = "".join([">", seq.name, "\n"])
                fp.write(seq_desc)
                fp.write(seq.seq)
                fp.write("\n")
                fp.write("\n")

        # Write train file.
        write_file = "data/cv5/train_pos_" + str(i)
        with open(write_file, 'w') as fp:
            for j in range(i):
                for seq in pos_cv[j]:
                    seq_desc = "".join([">", seq.name, "\n"])
                    fp.write(seq_desc)
                    fp.write(seq.seq)
                    fp.write("\n")
                    fp.write("\n")
            for j in range(i+1, 5):
                for seq in pos_cv[j]:
                    seq_desc = "".join([">", seq.name, "\n"])
                    fp.write(seq_desc)
                    fp.write(seq.seq)
                    fp.write("\n")
                    fp.write("\n")

        write_file = "data/cv5/train_neg_" + str(i)
        with open(write_file, 'w') as fp:
            for j in range(i):
                for seq in neg_cv[j]:
                    seq_desc = "".join([">", seq.name, "\n"])
                    fp.write(seq_desc)
                    fp.write(seq.seq)
                    fp.write("\n")
                    fp.write("\n")
            for j in range(i+1, 5):
                for seq in neg_cv[j]:
                    seq_desc = "".join([">", seq.name, "\n"])
                    fp.write(seq_desc)
                    fp.write(seq.seq)
                    fp.write("\n")
                    fp.write("\n")

    return pos_cv, neg_cv


def cv5_borderline_smote_revc_kmer(fold_path, filename, k):
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    for i in range(5):
        # Generate RevcKmer vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)

        # Generate borderline SMOTE synthetic vecs from train_vecs.
        train_vecs = np.row_stack((train_pos_revc_kmer_vecs, train_neg_revc_kmer_vecs))
        train_vecs_labels = [1] * len(train_pos_revc_kmer_vecs) + [-1] * len(train_neg_revc_kmer_vecs)
        _1, synthetic, _2 = smote.borderline_smote(train_vecs, train_vecs_labels, 1, N=300, k=5)

        # Write test file.
        write_file = fold_path + filename + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_revc_kmer_vecs + test_neg_revc_kmer_vecs
        test_vecs_labels = [1] * len(test_pos_revc_kmer_vecs) + [-1] * len(test_neg_revc_kmer_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + "_train_" + str(i) + ".txt"
        train_pos_revc_kmer_vecs = train_pos_revc_kmer_vecs + synthetic.tolist()
        train_vecs = train_pos_revc_kmer_vecs + train_neg_revc_kmer_vecs
        train_vecs_labels = [1] * len(train_pos_revc_kmer_vecs) + [-1] * len(train_neg_revc_kmer_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


def cv5_smote_revc_kmer(fold_path, filename, k):
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    for i in range(5):
        # Generate RevcKmer vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))

        # Generate SMOTE synthetic vecs from train_pos_vecs.
        synthetic = smote.smote(train_pos_revc_kmer_vecs, N=200, k=5)

        # Write test file.
        write_file = fold_path + filename + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_revc_kmer_vecs + test_neg_revc_kmer_vecs
        test_vecs_labels = [1] * len(test_pos_revc_kmer_vecs) + [-1] * len(test_neg_revc_kmer_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + "_train_" + str(i) + ".txt"
        train_pos_revc_kmer_vecs = train_pos_revc_kmer_vecs.tolist() + synthetic.tolist()
        train_vecs = train_pos_revc_kmer_vecs + train_neg_revc_kmer_vecs
        train_vecs_labels = [1] * len(train_pos_revc_kmer_vecs) + [-1] * len(train_neg_revc_kmer_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


def cv5_loop_borderline_smote_revc_kmer(fold_path, filename, k):
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    for i in range(5):
        # Generate RevcKmer vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_revc_kmer_vecs = revc_kmer.make_revckmer_vec(fp)

        # Generate borderline SMOTE synthetic vecs from train_vecs.
        train_vecs = np.row_stack((train_pos_revc_kmer_vecs, train_neg_revc_kmer_vecs))
        train_vecs_labels = [1] * len(train_pos_revc_kmer_vecs) + [-1] * len(train_neg_revc_kmer_vecs)
        synthetic = smote.loop_borderline_smote(train_vecs, train_vecs_labels, 1, -1, N=200, k=5)

        # Write test file.
        write_file = fold_path + filename + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_revc_kmer_vecs + test_neg_revc_kmer_vecs
        test_vecs_labels = [1] * len(test_pos_revc_kmer_vecs) + [-1] * len(test_neg_revc_kmer_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + "_train_" + str(i) + ".txt"
        train_pos_revc_kmer_vecs = train_pos_revc_kmer_vecs + synthetic
        train_vecs = train_pos_revc_kmer_vecs + train_neg_revc_kmer_vecs
        train_vecs_labels = [1] * len(train_pos_revc_kmer_vecs) + [-1] * len(train_neg_revc_kmer_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


def cv5_psednc(fold_path, filename):
    """Contrast experiment by psednc in article
    Prediction of DNase I Hypersensitive Sites by Using Pseudo Nucleotide Compositions.
    """
    lamada = 6
    w = 0.2
    psednc = PseDNC(lamada, w)
    for i in range(5):
        # Generate RevcKmer_PseDNC vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_psednc_vecs = psednc.make_psednc_vec(fp)
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_psednc_vecs = psednc.make_psednc_vec(fp)
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_psednc_vecs = psednc.make_psednc_vec(fp)
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_psednc_vecs = psednc.make_psednc_vec(fp)

        n_lamada = "_".join([str(lamada), str(w)])
        # Write test file.
        write_file = fold_path + filename + "_" + n_lamada + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_psednc_vecs + test_neg_psednc_vecs
        test_vecs_labels = [1] * len(test_pos_psednc_vecs) + [-1] * len(test_neg_psednc_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + "_" + n_lamada + "_train_" + str(i) + ".txt"
        train_vecs = train_pos_psednc_vecs + train_neg_psednc_vecs
        train_vecs_labels = [1] * len(train_pos_psednc_vecs) + [-1] * len(train_neg_psednc_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


def cv5_smote_revc_psednc(fold_path, filename, k):
    # Generate pos and neg vecs and SMOTE synthetic vecs.
    lamada = 6
    w = 0.8
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    psednc = PseDNC(lamada, w)
    for i in range(5):
        # Generate RevcKmer_PseDNC vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        test_neg_revc_psednc_vecs = np.column_stack((test_neg_revc_kmer_vecs, test_neg_psednc_vecs[:, -lamada:]))

        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        test_pos_revc_psednc_vecs = np.column_stack((test_pos_revc_kmer_vecs, test_pos_psednc_vecs[:, -lamada:]))

        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        train_neg_revc_psednc_vecs = np.column_stack((train_neg_revc_kmer_vecs, train_neg_psednc_vecs[:, -lamada:]))

        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        train_pos_revc_psednc_vecs = np.column_stack((train_pos_revc_kmer_vecs, train_pos_psednc_vecs[:, -lamada:]))

        # Generate synthetic vecs from pos_vecs.
        synthetic1 = (smote.smote(train_pos_revc_psednc_vecs, N=100, k=5)).tolist()
        synthetic2 = (smote.smote(train_pos_revc_psednc_vecs, N=50, k=5)).tolist()
        synthetic = np.row_stack((synthetic1, synthetic2))

        n_lamada = "_".join([str(lamada), str(w)])
        # Write test file.
        write_file = fold_path + filename + '_' + n_lamada + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_revc_psednc_vecs.tolist() + test_neg_revc_psednc_vecs.tolist()
        test_vecs_labels = [1] * len(test_pos_revc_psednc_vecs) + [-1] * len(test_neg_revc_psednc_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + '_' + n_lamada + "_train_" + str(i) + ".txt"
        train_pos_vecs = train_pos_revc_psednc_vecs.tolist() + synthetic.tolist()
        train_vecs = train_pos_vecs + train_neg_revc_psednc_vecs.tolist()
        train_vecs_labels = [1] * len(train_pos_vecs) + [-1] * len(train_neg_revc_psednc_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


def cv5_borderline_smote_revc_psednc(fold_path, filename, k):
    lamada = 6
    w = 0.8
    revc_kmer = RevcKmer(k=k, normalize=True, upto=True)
    psednc = PseDNC(lamada, w)
    for i in range(5):
        # Generate RevcKmer_PseDNC vecs.
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "test_neg_" + str(i)) as fp:
            test_neg_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        test_neg_revc_psednc_vecs = np.column_stack((test_neg_revc_kmer_vecs, test_neg_psednc_vecs[:, -lamada:]))

        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_revc_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "test_pos_" + str(i)) as fp:
            test_pos_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        test_pos_revc_psednc_vecs = np.column_stack((test_pos_revc_vecs, test_pos_psednc_vecs[:, -lamada:]))

        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "train_neg_" + str(i)) as fp:
            train_neg_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        train_neg_revc_psednc_vecs = np.column_stack((train_neg_revc_kmer_vecs, train_neg_psednc_vecs[:, -lamada:]))

        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_revc_kmer_vecs = np.array(revc_kmer.make_revckmer_vec(fp))
        with open(fold_path + "train_pos_" + str(i)) as fp:
            train_pos_psednc_vecs = np.array(psednc.make_psednc_vec(fp))
        train_pos_revc_psednc_vecs = np.column_stack((train_pos_revc_kmer_vecs, train_pos_psednc_vecs[:, -lamada:]))

        # Generate borderline SMOTE synthetic vecs from train_vecs.
        train_vecs = np.row_stack((train_pos_revc_psednc_vecs, train_neg_revc_psednc_vecs))
        train_vecs_labels = [1] * len(train_pos_revc_psednc_vecs) + [-1] * len(train_neg_revc_psednc_vecs)
        _1, synthetic, _2 = smote.borderline_smote(train_vecs, train_vecs_labels, 1, N=300, k=5)

        n_lamada = "_".join([str(lamada), str(w)])
        # Write test file.
        write_file = fold_path + filename + '_' + n_lamada + "_test_" + str(i) + ".txt"
        test_vecs = test_pos_revc_psednc_vecs.tolist() + test_neg_revc_psednc_vecs.tolist()
        test_vecs_labels = [1] * len(test_pos_revc_psednc_vecs) + [-1] * len(test_neg_revc_psednc_vecs)
        write_libsvm(test_vecs, test_vecs_labels, write_file)

        # Write train file.
        write_file = fold_path + filename + '_' + n_lamada + "_train_" + str(i) + ".txt"
        train_pos_vecs = train_pos_revc_psednc_vecs.tolist() + synthetic.tolist()
        train_vecs = train_pos_vecs + train_neg_revc_psednc_vecs.tolist()
        train_vecs_labels = [1] * len(train_pos_vecs) + [-1] * len(train_neg_revc_psednc_vecs)
        write_libsvm(train_vecs, train_vecs_labels, write_file)


if __name__ == "__main__":
    print("Begin.")
    start_time = time.time()

    # Experiment.
    # whole_revc_kmer(pos_file="data/hs.fasta", neg_file="data/non-hs.fasta", k=6)
    # whole_revc_kmer_psednc(pos_file="data/hs.fasta", neg_file="data/non-hs.fasta", k=6)
    # cv5_revc_kmer(fold_path="data/cv5/", filename="cv5_revc_kmer", k=6)
    # cv5_smote_revc_kmer(fold_path="data/cv5/", filename="cv5_smote_revc_kmer", k=6)
    # cv5_borderline_smote_revc_kmer(fold_path="data/cv5/", filename="cv5_borderline_revc_kmer", k=6)
    # cv5_loop_borderline_smote_revc_kmer(fold_path="data/cv5/", filename="cv5_loop_borderline_revc_kmer", k=6)
    # cv5_psednc("data/cv5/", "cv5_psednc")
    whole_revc_kmer_psednc_choose_args(pos_file="data/hs.fasta", neg_file="data/non-hs.fasta", k=2)
    # cv5_smote_revc_psednc(fold_path="data/cv5/", filename="cv5_smote_revc_psednc", k=6)
    # cv5_borderline_smote_revc_psednc(fold_path="data/cv5/", filename="cv5_smote_borderline", k=6)

    print("End. Used %s s." % (time.time() - start_time))