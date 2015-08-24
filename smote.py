#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The MIT License (MIT)
Copyright (c) 2012-2013 Karsten Jeschkies <jeskar@web.de>
Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Created on 24.11.2012
@author: karsten jeschkies <jeskar@web.de>
Modified on 12.05.2015
@changer: Fule liu <liufule12@gmail.com>
This is an implementation of the SMOTE Algorithm. 
See: "SMOTE: synthetic minority over-sampling technique" by Chawla, N.V et al.
     "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning" by Hui Han et al.
"""

from random import choice
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(T, N, k, h=1.0):
    """Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_attrs]
        Holds the minority samples
    N : percentage of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.

    Returns
    -------
    synthetic : synthetic samples. array,
        shape = [(N/100) * n_minority_samples, n_attrs].

    Example
    -------


    """
    n_minority_samples, n_attrs = T.shape

    if N < 100:
        # create synthetic samples only for a subset of T.
        n_sub_minority_samples = int(n_minority_samples * (N * 1.0 / 100))
        if n_sub_minority_samples <= 1:
            raise ValueError("int(n_minority_samples * (N * 1.0 / 100)) must be > 1.")
        random_indices = np.random.random_integers(low=0, high=n_minority_samples - 1,
                                                   size=(1, n_sub_minority_samples))
        T = T[random_indices[0], :]
        n_minority_samples = T.shape[0]
        N = 100

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100.")

    N /= 100
    n_synthetic_samples = N * n_minority_samples
    synthetic = np.zeros(shape=(n_synthetic_samples, n_attrs))

    # Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(T)

    # Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in range(N):
            nn_index = choice(nn[0])
            # NOTE: nn includes T[i], we don't want to select it
            while nn_index == i:
                nn_index = choice(nn[0])

            dif = T[nn_index] - T[i]
            gap = np.random.random() * h
            synthetic[i * N + n, :] = T[i, :] + gap * dif

    return synthetic


def borderline_smote(X, y, minority_target, N, k):
    """Returns synthetic minority samples.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Holds the minority and majority samples
    y : array-like, shape = [n_samples]
        Holds the class targets for samples
    minority_target : value for minority class
    N : percentage of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.
    h : high in random.uniform to scale dif of synthetic sample

    Returns
    -------
    safe : Safe minorities
    synthetic : Synthetic sample of minorities in danger zone
    danger : Minorities of danger zone
    """

    n_samples, _ = X.shape

    # Learn nearest neighbours on complete training set
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)

    safe_minority_indices = []
    danger_minority_indices = []

    for i in range(n_samples):
        if y[i] != minority_target:
            continue

        nn = neigh.kneighbors(X[i], return_distance=False)
        majority_neighbours = 0
        for n in nn[0]:
            if y[n] != minority_target:
                majority_neighbours += 1
        if majority_neighbours == len(nn[0]):
            continue
        elif majority_neighbours < (len(nn[0]) / 2):
            safe_minority_indices.append(i)
        else:
            # DANGER zone
            danger_minority_indices.append(i)

    # SMOTE danger minority samples
    synthetic_samples = smote(X[danger_minority_indices], N, k, h=0.5)

    return (X[safe_minority_indices, :],
            synthetic_samples,
            X[danger_minority_indices, :])


def borderline_smote_indices(X, y, minority_target, N, k):
    """Returns safe_minority indices, synthetic minority samples and danger minority indices.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Holds the minority and majority samples
    y : array-like, shape = [n_samples]
        Holds the class targets for samples
    minority_target : value for minority class
    N : percentage of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.
    h : high in random.uniform to scale dif of synthetic sample

    Returns
    -------
    safe_minority_indices : Safe minorities indices
    synthetic : Synthetic sample of minorities in danger zone
    danger_minority_indices : Minorities of danger zone indices
    """

    n_samples, _ = X.shape

    # Learn nearest neighbours on complete training set
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)

    safe_minority_indices = []
    danger_minority_indices = []

    for i in range(n_samples):
        if y[i] != minority_target:
            continue

        nn = neigh.kneighbors(X[i], return_distance=False)
        majority_neighbours = 0
        for n in nn[0]:
            if y[n] != minority_target:
                majority_neighbours += 1
        if majority_neighbours == len(nn[0]):
            continue
        elif majority_neighbours < (len(nn[0]) / 2):
            safe_minority_indices.append(i)
        else:
            # DANGER zone
            danger_minority_indices.append(i)

    # SMOTE danger minority samples
    synthetic_samples = smote(X[danger_minority_indices], N, k, h=0.5)

    return (safe_minority_indices,
            synthetic_samples,
            danger_minority_indices)


def loop_borderline_smote(X, y, minority_target, majority_targer, N, k):
    """To generate more average data.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Holds the minority and majority samples
    y : array-like, shape = [n_samples]
        Holds the class targets for samples
    minority_target : value for minority class
    majority_target : value for majority class
    N : percentage of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.
    h : high in random.uniform to scale dif of synthetic sample

    Returns
    -------
    synthetic : Synthetic sample of minorities in danger zone
    """

    synthetic_samples = []

    new_y = np.array(y)
    new_minority_target = minority_target

    n_minority_samples = count_label(y, minority_target)
    # n_majority_samples = count_label(y, majority_targer)

    while n_minority_samples > 80:
        safe_minority_indices, temp_synthetic_samples, danger_minority_indices =\
            borderline_smote_indices(X, new_y, new_minority_target, N, k)
        print("safe_minority_indices", len(safe_minority_indices))
        print("temp_synthetic_samples", temp_synthetic_samples.shape)
        print("danger_minority_indices", len(danger_minority_indices))
        print("\n")
        synthetic_samples += temp_synthetic_samples.tolist()
        new_y[safe_minority_indices] = minority_target
        new_y[danger_minority_indices] = majority_targer
        n_minority_samples = len(safe_minority_indices)

    return synthetic_samples


def count_label(y, target):
    """Count target label.

    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Holds the class label for samples
    target : value for target label.

    Returns
    -------
    count : the number of target label
    """

    n_target = 0
    for label in y:
        if label == target:
            n_target += 1

    return n_target


def main():
    from repDNA.nac import Kmer
    from repDNA.util import write_libsvm
    kmer = Kmer(k=1, normalize=True)
    with open("hs.fasta") as fp:
        pos = kmer.make_kmer_vec(fp)
    with open("non-hs.fasta") as fp:
        neg = kmer.make_kmer_vec(fp)
    print(len(pos))
    print(len(neg))
    data = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [-1] * len(neg))
    synthetic1 = loop_borderline_smote(data, labels, 1, -1, N=100, k=5)
    synthetic2 = loop_borderline_smote(data, labels, 1, -1, N=65, k=5)
    synthetic = synthetic1 + synthetic2
    for e in synthetic:
        print(e)
    print(len(synthetic))

    pos += synthetic
    data = pos + neg
    labels = [1] * len(pos) + [-1] * len(neg)
    write_libsvm(data, labels, "loop_borderline_smote.txt")


if __name__ == "__main__":
    main()
