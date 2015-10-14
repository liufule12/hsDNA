__author__ = 'Fule Liu'


def add_label(file_name, write_name):
    with open(file_name) as fp:
        lines = fp.readlines()

    with open(write_name, 'wb') as fp:
        # fp.write("real_label pos_prob neg_prob\n")
        fp.write("real_label pos_prob\n")
        for idx, line in enumerate(lines):
            line = line.rstrip().split()
            if idx < 280:
                # fp.write(" ".join(["1", line]))
                fp.write(" ".join(["1", line[0]+'\n']))
            else:
                # fp.write(" ".join(["0", line]))
                fp.write(" ".join(["0", line[0]+'\n']))


def ensemble_roc(files, write_name):
    lines_list = []
    for file_name in files:
        with open(file_name) as fp:
            lines_list.append(fp.readlines())

    with open(write_name, 'wb') as fp:
        fp.write("read_label kmer revcKmer DACC PseDNC\n")
        for i in range(1017):
            kmer_line = lines_list[0][i].rstrip().split()
            revc_kmer_line = lines_list[1][i].rstrip().split()
            dacc_line = lines_list[2][i].rstrip().split()
            psednc_line = lines_list[3][i].rstrip().split()
            if i < 280:
                fp.write(" ".join(["1", kmer_line[1], revc_kmer_line[1], dacc_line[1], psednc_line[1]+'\n']))
            else:
                fp.write(" ".join(["0", kmer_line[1], revc_kmer_line[1], dacc_line[1], psednc_line[1]+'\n']))


if __name__ == "__main__":
    # add_label("0.4_0.1_0.5", "0.4_0.1_0.5_label")
    # add_label("roc/0.1_0.2_0.05_0.65", "roc/0.1_0.2_0.05_0.65_label")
    # add_label("../cv_revckmer_article_predict/cv5_revckmer_article_predict", "roc/cv5_revckmer_label")
    # add_label("../cv5_psednc/cv5_psednc_test.predict", "roc/cv5_psednc_label")
    # add_label("../cv5_dacc/cv5_dacc_test.predict", "roc/cv5_dacc_label")

    files = ["../cv5_kmer/cv5_kmer_test.predict", "../cv5_revckmer_article/cv5_revckmer_article_predict",
             "../cv5_dacc/cv5_dacc_test.predict", "../cv5_psednc/cv5_psednc_test.predict"]
    ensemble_roc(files, "roc/cv5_ensemble4_label")