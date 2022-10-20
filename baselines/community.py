import numpy as np
from cdlib.algorithms import demon, kclique


def get_demon_communities(G):
    communities = demon(G, min_com_size=1, epsilon=1).communities
    return set([tuple(sorted(x)) for x in communities])


def get_kclique_communities(G, G_train, H_train):
    # Warning: H_train is only 10% in semi-supervised setting
    j_k = []
    for q in range(1, 6):
        k = int(np.quantile([len(e) for e in H_train], 0.1*q)) + 1
        communities = kclique(G_train, k=k).communities
        communities = set([tuple(sorted(x)) for x in communities])
        jaccard = len(H_train) / len(communities | H_train)
        j_k.append((jaccard, k))
    best_k = sorted(j_k)[-1][1]
    communities = kclique(G, k=best_k).communities
    communities = set([tuple(sorted(x)) for x in communities])
    return communities, best_k

