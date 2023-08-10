import numpy as np
from collections import defaultdict
import pickle
import operator as op
from functools import reduce
from itertools import combinations
import random
from numpy.linalg import norm
import networkx as nx

def compute_clique_distribution(max_cliques, smaller_cliques):
    max_cliques = list(max_cliques)
    smaller_cliques = list(smaller_cliques)
    size2max_cliques_i = group_by_size(max_cliques)
    size2smaller_cliques_i = group_by_size(smaller_cliques)
    clique_distribution = {}
    unique_cliques = {}
    for max_clique_size in sorted(size2max_cliques_i.keys()):

        max_clique_count = len(size2max_cliques_i[max_clique_size])
        smaller_cliques_sizes = [x for x in sorted(size2smaller_cliques_i.keys()) if x<=max_clique_size]

        if len(smaller_cliques_sizes) == 0:
            continue
        clique_distribution[max_clique_size] = {}
        for smaller_clique_size in smaller_cliques_sizes:
            clique_distribution[max_clique_size][smaller_clique_size] = [0, max_clique_count, 0, []]
#             print(max_clique_size, max_clique_count, len(size2smaller_cliques_i[smaller_clique_size]))
            for max_clique_i in size2max_cliques_i[max_clique_size]:
                max_clique = max_cliques[max_clique_i]
                #############################
                for smaller_clique_i in size2smaller_cliques_i[smaller_clique_size]:
                    smaller_clique = smaller_cliques[smaller_clique_i]
                    #                     print(smaller_clique, max_clique)
                    if len(smaller_clique - max_clique) == 0:
                        clique_distribution[max_clique_size][smaller_clique_size][0] += 1
                        clique_distribution[max_clique_size][smaller_clique_size][3].append(smaller_clique_i)
    fill_in_efficiency(clique_distribution)
    return clique_distribution


def group_by_size(cliques):
    size2cliques = defaultdict(list)
    for i, clique in enumerate(cliques):
        size2cliques[len(clique)].append(i)
    return size2cliques


def print_distribution(clique_distribution):
    for mc_size in sorted(clique_distribution.keys()):
        for sc_size in sorted(clique_distribution[mc_size].keys()):
            print(mc_size, sc_size, clique_distribution[mc_size][sc_size][:3])


def fill_in_efficiency(clique_distribution):
    for mc_size in sorted(clique_distribution.keys()):
        for sc_size in sorted(clique_distribution[mc_size].keys()):
            v = clique_distribution[mc_size][sc_size]
            # e = v[0] / v[1]
            # e = len(set([tuple(sorted(x)) for x in v[3]])) / v[1]
            e = len(set(v[3])) / v[1]
            clique_distribution[mc_size][sc_size][2] = e


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  

def rank_efficiency(dis):
    l = []
    for mc_size in dis.keys():
        for sc_size in dis[mc_size].keys():
            l.append((dis[mc_size][sc_size][2], (mc_size, sc_size)))
    return sorted(l, reverse=1)
def get_efficiency_matrix(dis, normalize=True):
    m = np.zeros((max(dis.keys()), max([max(dis[k].keys()) for k in dis])))
    for mc_size in dis.keys():
        for sc_size in dis[mc_size].keys():
            m[mc_size-1, sc_size-1] = dis[mc_size][sc_size][2]
#             m[mc_size-1, sc_size-1] = dis[mc_size][sc_size][0]/dis[mc_size][sc_size][1]
            if normalize:
                m[mc_size-1, sc_size-1] = m[mc_size-1, sc_size-1] / ncr(mc_size, sc_size)
    return m

from scipy.special import kl_div, rel_entr
eps = 1e-26
def kl(ref, com):
#     m1 = ref[:16, :16]
#     m2 = com
    h = max(ref.shape[0], com.shape[0])
    w = max(ref.shape[1], com.shape[1])
    m1 = np.zeros((h,w))
    m2 = np.zeros((h,w))
    
    m1[:ref.shape[0], :ref.shape[1]] = ref
    m2[:com.shape[0], :com.shape[1]] = com
    
    a = min(ref.shape[0], com.shape[0])
    b = min(ref.shape[1], com.shape[1])
    m3 = m1[:a, :b]
    m4 = m2[:a, :b]
    
#     for i in range(ref.shape[0]):
#         m1[i, :ref.shape[1]] += ref[i]
#     for i in range(com.shape[0]):
#         m2[i, :com.shape[1]] += com[i]
    
    return 0.5*(dist(m1, m2)/norm(m1)**2 + dist(m3, m4)/norm(m3)**2)
def dist(m1, m2):
    m1/=m1.sum()
    m2/=m2.sum()
    return ((m1-m2)**2).mean()
#     m1 = m1.flatten() + eps
#     m2 = m2.flatten() + eps
#     m1/=m1.sum()
#     m2/=m2.sum()
#     return sum(rel_entr(m1, m2))
def load_m(dataset, split):
    s = 'all'
    with open('../repo/data/{}/{}_cliques_distribution_{}.pkl'.format(dataset, s, split), 'rb') as f:
        dis1 = pickle.load(f)
    return get_efficiency_matrix(dis1)

def make_graph(H):
    G = nx.Graph()
    for e in H:
        for u, v in combinations(e, 2):
            G.add_edge(u, v)
    return G
def detect_max_cliques(graph):
    cliques = nx.algorithms.clique.find_cliques(graph)
    cliques = set(sorted([tuple(sorted(clique)) for clique in cliques]))
    return cliques

def to_list_of_set(H):
    return [set(x) for x in H]

def get_efficiency_matrix_from_hypergraph(H):
    G = make_graph(H)
    max_cliques = detect_max_cliques(G)
    dis = compute_clique_distribution(to_list_of_set(max_cliques), to_list_of_set(H))
    efficiency_matrix = get_efficiency_matrix(dis, normalize=True)
    return efficiency_matrix
def compare(H1, H2):
    m1 = get_efficiency_matrix_from_hypergraph(H1)
    m2 = get_efficiency_matrix_from_hypergraph(H2)
    return kl(m1, m2)


def create_random_hypergraph(n, m_arr, K):
    H = []
    for i, k in enumerate(range(2, K+1)):
        m = m_arr[i]
        size_k_hyperedges = random.sample(list(combinations(range(n), k)), m)
        H += size_k_hyperedges
    return H

def get_stability(n, m_arr, K, eps=0.05, repeats=10):
    diffs = []
    H_base = create_random_hypergraph(n, m_arr, K)
    for _ in range(repeats):
        delta = (np.random.random(K-1) -1)*norm(m_arr)*0.05
        m_arr_new = m_arr + (np.rint(delta)).astype(int)
        H_new = create_random_hypergraph(n, m_arr_new, K)
        diff = compare(H_base, H_new)
        diffs.append(diff)
        
    diffs = np.array(diffs)
    return diffs.mean()/eps, diffs


for n in range(10, 60, 10):
    stability = get_stability(n=30, m_arr=np.array([n, n, n, n]), K=5, eps=0.05, repeats=10)
    print('n={}, instability={:.2f};'.format(n, stability[0]), end= ' ')