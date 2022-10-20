from itertools import combinations
from collections import defaultdict
import numpy as np
import random
import time
import os
import sys

def write_csv(h, fname):
    with open(fname, 'w') as f:
        for edge in h:
            f.write(','.join([str(node) for node in edge])+'\n')

def read_csv(fname):
    feature_mat = []
    with open(fname, 'r+') as f:
        for line in f.readlines():
            # print([value for value in line.split(',')])
            feature_mat.append([float(value) for value in line.split(',') if not value =='\n'])
    feature_mat = np.array(feature_mat, dtype=np.float)
    return feature_mat


def extract_motif_features(candidates, H, node_degree, use_c=True, jobs=1, args=None):
    '''
    :param candidates: a list of clique candidates , each of which has a motif fingerprint to extract
    :param max_cliques: the pseudo hypergraph (a list of sorted tuple)
    :return: feature matrix (len(candidates), #motifs*4+1), feature dict (name of each feature)
    '''
    if use_c:
        # save as csv, run c code, read results
        fname = 'data/{}/features{}.csv'.format(args.dataset, len(candidates))
        if os.path.exists(fname):
            return read_csv(fname=fname), None
        tt = time.time()
        print("Using C code to accelerate motif extraction", end=', ')
        write_csv(candidates, fname='data/{}/candidates.csv'.format(args.dataset))
        write_csv(H, fname='data/{}/H.csv'.format(args.dataset))
        slash = '/'
        if sys.platform.startswith('win'):
            slash = '\\'
        if jobs > 1:
            exit_code = os.system(".{}cmotif {} {} {}".format(slash, args.dataset,  args.downsample, jobs))
        else:
            exit_code = os.system(".{}cmotif {} {}".format(slash, args.dataset, args.downsample))
        print("exit code: ", exit_code, end=", ")
        feature_mat = read_csv(fname=fname)
        print('time usage:', str(time.time()-tt)[:6], 's')
        return feature_mat, None

    def intersect(he1, he2, exclude):
        return len((set(H[he1]) & set(H[he2])) - exclude) > 0

    def contain(he, subset):
        return len(subset - set(H[he])) == 0

    H = sorted(H)
    feature_mat = []
    feature_dict = defaultdict(list)
    node2hes = get_node_he_neighbors(H)   #{node: index of the he in max_clique}
    for clique_i, clique in enumerate(candidates):

        # print(clique_i, end=' ')
        motif2distribution = defaultdict(list) #{motif index: distribution (list of counts) of this motif}
        # counting motif 1-3
        degs = []
        for v in clique:
            degs.append(node_degree[v])
            for i in range(1, 4):
                motif2distribution[str(i)].append(0)
            hes = node2hes[v]
            motif2distribution['1'][-1] = len(hes)
            for he1, he2 in combinations(hes, 2):
                flag = not intersect(he1, he2, {v})
                motif2distribution['2'][-1] += flag
                motif2distribution['3'][-1] += not flag

        # counting motif 4-13
        if len(clique) == 1:
            for i in range(4, 14):
                motif2distribution[str(i)].append(0)
        else:
            for v1, v2 in combinations(clique, 2):
                # if random.random() < 0.5:
                #     continue
                for i in range(4, 14):
                    motif2distribution[str(i)].append(0)
                hes = set(node2hes[v1]) | set(node2hes[v2])
                for he in hes:
                    flag = contain(he, {v1, v2})
                    # flag = 1
                    motif2distribution['4'][-1] += flag
                    motif2distribution['5'][-1] += not flag
                for he1, he2 in combinations(hes, 2):
                    its = intersect(he1, he2, {v1, v2})
                    c11 = contain(he1, {v1})
                    c12 = (not c11) or contain(he1, {v2})
                    c21 = contain(he2, {v1})
                    c22 = (not c21) or contain(he2, {v2})
                    flags = (its, c11, c12, c21, c22)
                    motif2distribution['6'][-1] += (flags == (0, 1, 0, 0, 1)) or (flags == (0, 0, 1, 1, 0))
                    motif2distribution['7'][-1] += (flags == (0, 1, 0, 1, 0)) or (flags == (0, 0, 1, 0, 1))
                    motif2distribution['8'][-1] += (flags == (1, 1, 0, 0, 1)) or (flags == (1, 0, 1, 1, 0))
                    motif2distribution['9'][-1] += flags[0] == 0 and sum(flags[1:]) == 3
                    motif2distribution['10'][-1] += (flags == (1, 1, 0, 1, 0)) or (flags == (1, 0, 1, 0, 1))
                    motif2distribution['11'][-1] += flags[0] == 0 and all(flags[1:])
                    motif2distribution['12'][-1] += flags[0] == 1 and sum(flags[1:]) == 3
                    motif2distribution['13'][-1] += all(flags)
                # print(len(hes), end=' ')
        print(clique_i, end=' ')
        feature = [len(clique), np.array(degs).mean()] + vectorize_distribution(motif2distribution)
        feature_mat.append(feature)
        feature_dict[clique_i].append(motif2distribution)
        write_csv(feature_mat, fname='features_python.csv')



    feature_mat = np.array(feature_mat, dtype=float)
    write_csv(candidates, fname='candidates.csv')
    write_csv(H, fname='H.csv')
    if jobs >1:
        exit_code = os.system(".\cmotif {}".format(jobs))
    else:
        exit_code = os.system(".\cmotif")
    feature_mat2 = read_csv(fname='features.csv')
    print("diff between 2 implementations", np.abs(feature_mat - feature_mat2).sum())
    # write_csv(feature_mat - feature_mat2, fname='diff.csv')
    np.save('feature_mat', feature_mat)
    np.save('feature_mat2', feature_mat2)
    return feature_mat, feature_dict


def vectorize_distribution(motif2distribution):
    feature = []
    for i in range(1, 1+len(motif2distribution)):
        distribution = np.array(motif2distribution[str(i)], dtype=float)
        feature += [distribution.min(), distribution.max(), distribution.mean(), distribution.std()]
    return feature


def get_node_he_neighbors(H):
    node2hes = defaultdict(list)
    for he_i, he in enumerate(H):
        for node in he:
            node2hes[node].append(he_i)
    return node2hes


def filter_top_motifs(X):
    start = 4
    step = 4
    end = len(X.shape[0])
    mean_columns = list(range(start, end, step))
    k = 5
    good_motifs = X[:, mean_columns].std(axis=0).argmax()[:5]


