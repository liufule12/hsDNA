__author__ = 'Fule Liu'


def ensemble_res4():
    with open("../cv5_kmer/cv5_kmer_test.predict") as fp:
        kmer_prob = [(line.rstrip().split()[1], line.rstrip().split()[2]) for line in fp.readlines()]
    with open("../cv_revckmer_article_predict/cv5_revckmer_article_predict") as fp:
        revc_kmer_prob = [(line.rstrip().split()[1], line.rstrip().split()[2]) for line in fp.readlines()]
    with open("../cv5_dacc/cv5_dacc_test.predict") as fp:
        dacc_prob = [(line.rstrip().split()[1], line.rstrip().split()[2]) for line in fp.readlines()]
    with open("../cv5_psednc/cv5_psednc_test.predict") as fp:
        psednc_prob = [(line.rstrip().split()[1], line.rstrip().split()[2]) for line in fp.readlines()]

    w1 = 0.0
    while w1 <= 1:
        w2 = 0.0
        if w1 + w2 > 1:
            w1 += 0.05
            continue
        while w2 <= 1:
            w3 = 0.0
            if w1 + w2 + w3 > 1:
                w2 += 0.05
                continue
            while w3 <= 1:
                w4 = 1 - w1 - w2 - w3
                if w4 < 0:
                    w3 += 0.05
                    continue

                print(w1, w2, w3, w4)
                ensemble_prob = []
                for idx, kmer_val in enumerate(kmer_prob):
                    pos_prob = w1 * float(kmer_val[0]) + w2 * float(revc_kmer_prob[idx][0]) + \
                               w3 * float(dacc_prob[idx][0]) + w4 * float(psednc_prob[idx][0])
                    neg_prob = w1 * float(kmer_val[1]) + w2 * float(revc_kmer_prob[idx][1]) + \
                               w3 * float(dacc_prob[idx][1]) + w4 * float(psednc_prob[idx][1])
                    ensemble_prob.append((pos_prob, neg_prob))

                write_name = "E:/ensemble_output4/" + str(round(w1, 2)) + '_' + str(round(w2, 2)) +\
                             '_' + str(round(w3, 2)) + '_' + str(round(w4, 2))
                print(write_name)
                with open(write_name, 'wb') as fp:
                    for pos_prob, neg_prob in ensemble_prob:
                        fp.write(str(pos_prob) + ' ' + str(neg_prob) + '\n')

                w3 += 0.05
            w2 += 0.05
        w1 += 0.05


if __name__ == "__main__":
    ensemble_res4()
