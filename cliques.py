import networkx as nx
from copy import deepcopy
from itertools import combinations
from collections import defaultdict, Counter
import pickle
import os
import numpy as np
import operator as op
from functools import reduce
import random


def compute_cliques(graphs, args, logger):
    logger.info('Start computing cliques')
    cliques = {}
    cliques['max_cliques_train'] = detect_max_cliques(graphs['G_train'], 'train', args, logger)
    cliques['support_cliques_train'], _ = find_support(cliques['max_cliques_train'])
    cliques['max_cliques_test'] = detect_max_cliques(graphs['G_test'], 'test', args, logger)
    cliques['support_cliques_test'], _ = find_support(cliques['max_cliques_test'])
    lengths = np.array([len(x) for x in graphs['simplicies_test']])
    c = Counter([n for h in graphs['simplicies_test'] for n in h])
    avg_deg = np.array(list(c.values())).mean() if len(c) > 0 else 0
    logger.info('test nodes {} hyperedges {} avg size {} std {} avg degree {} max cliques {}'.format(graphs['G_test'].number_of_nodes(),
                                                                                             len(graphs['simplicies_test']),
                                                                                             lengths.mean(),
                                                                                             lengths.std(),
                                                                                             avg_deg,
                                                                                             len(cliques['max_cliques_test'])))
    logger.info('Breakdown of test hyperedges (max+small) {} {} {}'.format(len(graphs['simplicies_test'] & cliques['max_cliques_test']), '+',
          len(graphs['simplicies_test']) - len(graphs['simplicies_test'] & cliques['max_cliques_test'])))
    logger.info('Intersection {} out of {} {}'.format(len(graphs['simplicies_test'] & cliques['support_cliques_test']),
          len(cliques['support_cliques_test']), len(cliques['max_cliques_test'])))
    logger.info('support cliques train:{} - test: {}'.format(len(cliques['support_cliques_train']),
                                                             len(cliques['support_cliques_test'])))
    # with open('data/stats.csv', 'a') as f:
    #     f.write(','.join([str(x) for x in (args.dataset, graphs['G_test'].number_of_nodes(), len(graphs['simplicies_test']), round(lengths.mean(), 1), round(lengths.std(), 1), round(nx.average_clustering(graphs['G_test']),2), len(cliques['max_cliques_test']))])+'\n')

    # finishing computing max cliques
    logger.info('Optimizing clique sampler .. ')
    cliques['sampler'] = CliqueSampler(cliques['max_cliques_train'], graphs['simplicies_train'], args.beta, logger, args)
    cliques['children_cliques_train'] = cliques['sampler'].find_children(cliques['max_cliques_train'], graphs['simplicies_train'])
    cliques['children_cliques_test'] = cliques['sampler'].find_children(cliques['max_cliques_test'], graphs['simplicies_test'])

   # cliques['children_cliques_train'] = find_children(cliques['max_cliques_train'], max_size=args.max_child_size) # the children cliques are only children of the support
    # cliques['children_cliques_test'] = find_children(cliques['max_cliques_test'], max_size=args.max_child_size)

    logger.info('Clique analysis done.')

    return cliques


def detect_max_cliques(graph, mode, args, logger):

    cache = args.data_dir + args.dataset + '/cliques_{}.pkl'.format(mode)

    if os.path.exists(cache):
        logger.info('Found cache for max cliques {}'.format(mode))
        with open(cache, 'rb') as c:
            cliques = pickle.load(c)
    else:
        logger.info('Couldn\'t find cache for max cliques {}'.format(mode))
        cliques = nx.algorithms.clique.find_cliques(graph)
        cliques = set(sorted([tuple(sorted(clique)) for clique in cliques]))
        with open(cache, 'wb') as c:
            pickle.dump(cliques, c)
    logger.info('Number of maximum cliques:{}'.format(len(cliques)))
    return cliques


def find_support(cliques_):
    cliques = deepcopy(cliques_)
    edge2count = {}
    for clique in cliques:
        for edge in combinations(clique, 2):
            edge_ = tuple(sorted(edge))
            if edge_ not in edge2count:
                edge2count[edge_] = 1
            else:
                edge2count[edge_] += 1
    iteration = 0
    while True:
        iteration += 1
        to_remove = set()
        for clique in cliques:
            if len(clique) == 1:
                continue
            if sum([int(edge2count[tuple(sorted(edge))] > 1) for edge in combinations(clique, 2)]) == int(
                    len(clique) * (len(clique) - 1) / 2):
                for edge in combinations(clique, 2):
                    edge_ = tuple(sorted(edge))
                    edge2count[edge_] -= 1
                to_remove.add(clique)
        cliques -= to_remove
        if len(to_remove) == 0:
            break
    return cliques, iteration-1





class CliqueSampler:
    def __init__(self, max_cliques, hyperedges, beta, logger, args):
        self.args = args
        self.max_cliques = [set(x) for x in max_cliques]
        self.hyperedges = [set(x) for x in hyperedges]

        self.beta = beta
        self.logger = logger
        if self.args.ablation > 0:
            return

        self.rho = self.compute_clique_distribution(self.max_cliques, self.hyperedges)

        self.N = max(list(self.rho.keys()))
        self.r, self.seq, self.n_collected = self.optimize(beta, self.N, self.rho)
        recall = self.n_collected / len(hyperedges)
        efficiency = self.n_collected / self.beta
        self.logger.info(self.seq)
        self.logger.info('Optimize Clique Sampler: #hyperedges collected:{}, recall: {}, efficiency:{}'.format(self.n_collected, recall, efficiency) )

    def compute_clique_distribution(self, max_cliques, smaller_cliques):
        cache_name = 'data/{}/rho.pkl'.format(self.args.dataset)
        if os.path.exists(cache_name):
            self.logger.info('Found cache for rho.')
            with open(cache_name, 'rb') as f:
                rho = pickle.load(f)
            return rho

        self.logger.info('Cache for rho not found. Computing rho ...')
        size2max_cliques_i = self.group_by_size(max_cliques)
        size2smaller_cliques_i = self.group_by_size(smaller_cliques)
        clique_distribution = {}
        for max_clique_size in sorted(size2max_cliques_i.keys()):

            max_clique_count = len(size2max_cliques_i[max_clique_size])
            smaller_cliques_sizes = [x for x in sorted(size2smaller_cliques_i.keys()) if x <= max_clique_size]

            if len(smaller_cliques_sizes) == 0:
                continue
            clique_distribution[max_clique_size] = {}
            for smaller_clique_size in smaller_cliques_sizes:
                clique_distribution[max_clique_size][smaller_clique_size] = [0, max_clique_count, 0, []]
                for max_clique_i in size2max_cliques_i[max_clique_size]:
                    max_clique = max_cliques[max_clique_i]
                    for smaller_clique_i in size2smaller_cliques_i[smaller_clique_size]:
                        smaller_clique = smaller_cliques[smaller_clique_i]
                        if len(smaller_clique - max_clique) == 0:
                            clique_distribution[max_clique_size][smaller_clique_size][0] += 1
                            clique_distribution[max_clique_size][smaller_clique_size][3].append(smaller_clique_i)
        self.fill_in_efficiency(clique_distribution)
        rho = self.get_rho(clique_distribution)

        with open(cache_name, 'wb') as f:
            pickle.dump(rho, f)
        return rho

    def group_by_size(self, cliques):
        size2cliques = defaultdict(list)
        for i, clique in enumerate(cliques):
            size2cliques[len(clique)].append(i)
        return size2cliques

    def print_distribution(self, clique_distribution):
        for mc_size in sorted(clique_distribution.keys()):
            for sc_size in sorted(clique_distribution[mc_size].keys()):
                self.logger.info(mc_size, sc_size, clique_distribution[mc_size][sc_size][:3])

    def fill_in_efficiency(self, clique_distribution):
        for mc_size in sorted(clique_distribution.keys()):
            for sc_size in sorted(clique_distribution[mc_size].keys()):
                v = clique_distribution[mc_size][sc_size]
                e = len(set(v[3])) / v[1]
                clique_distribution[mc_size][sc_size][2] = e

    def get_rho(self, dis):
        rho = {}
        N = max(list(dis.keys()))
        for i in range(1, N + 1):
            rho[i] = {}
            for j in range(1, N + 1):
                rho[i][j] = [set(), 1e8]
        for mc_size in dis.keys():
            for sc_size in dis[mc_size].keys():
                rho[mc_size][sc_size][0] = set(dis[mc_size][sc_size][3])
                rho[mc_size][sc_size][1] = dis[mc_size][sc_size][1] * ncr(mc_size, sc_size)

        return rho

    def optimize(self, beta, N, rho):
        # rho contains both E and Q
        Gamma = {}
        omega = {}
        Delta = {}
        n = {}

        r = np.zeros((N + 1, N + 1))
        for k in range(1, N + 1):
            Gamma[k] = set()
            omega[k] = set(range(k, N + 1))
            Delta[k], n[k] = self.Update(k, omega[k], Gamma[k], rho)
        seq = []
        d = 50
        beta = deepcopy(beta)
        while beta > 0:
            k = sorted((delta, k) for k, delta in Delta.items())[-1][1]
            seq.append((n[k], k))
            r[n[k]][k] = min([beta, rho[n[k]][k][1]]) / rho[n[k]][k][1]
            if r[n[k]][k] < 1:
                Gamma[k] = Gamma[k] | set(random.sample(list(rho[n[k]][k][0]), int(r[n[k]][k]*len(rho[n[k]][k][0]))))
            else:
                Gamma[k] = Gamma[k] | rho[n[k]][k][0]
            omega[k] = omega[k] - {n[k]}
            #         r[n[k]][k] = d
            d -= 1
            beta -= rho[n[k]][k][1]
            Delta[k], n[k] = self.Update(k, omega[k], Gamma[k], rho)
            if max(Delta.values()) == 0:
                break

        return r, seq, sum([len(x) for x in Gamma.values()])

    def Update(self, k, omega_k, Gamma_k, rho):
        if len(omega_k) == 0:
            return 0, 0
        best_e = -1
        best_n = -1
        for n in omega_k:
            e = (len(Gamma_k | rho[n][k][0]) - len(Gamma_k)) / rho[n][k][1]
            if e > best_e:
                best_e = e
                best_n = n
        return best_e, best_n

    def find_children(self, max_cliques, hyperedges):
        max_cliques = list(max_cliques)
        if self.args.ablation == 1:
            return self.find_children_random(max_cliques)
        if self.args.ablation == 2:
            return self.find_children_head_tail(max_cliques, max_size=2, tail=False)
        if self.args.ablation == 3:
            return self.find_children_head_tail(max_cliques, max_size=2, tail=True)

        size2max_cliques_i = self.group_by_size(max_cliques)
        children = defaultdict(list)
        for i, (n, k) in enumerate(self.seq):
            print_hits(children, hyperedges)
            end = int(len(size2max_cliques_i[n]) * self.r[n, k])
            for clique_i in size2max_cliques_i[n][:end]:
                max_clique = max_cliques[clique_i]
                for child in combinations(max_clique, k):
                    children[max_clique].append(tuple(sorted(child)))
        print()
        return children



    def find_children_head_tail(self, cliques, max_size=2, tail=True):
        # children {parent_clique: [child1, child2, ...]}
        children = {}
        count = 0
        for clique in cliques:
            children[clique] = []
            for size in range(1, min([max_size+1, len(clique)])):
                if size == 1:
                    continue
                for child in combinations(clique, size):
                    children[clique].append(tuple(sorted(child)))
                    count += 1
                    if count > self.beta:
                        return children
            if not tail or len(clique)-1 <= max_size:
                continue
            for size in range(max([max_size + 1, len(clique) - max_size + 1]), len(clique)+1):
                if size == 1:
                    continue
                for child in combinations(clique, size):
                    children[clique].append(tuple(sorted(child)))
                    count += 1
                    if count > self.beta:
                        return children
        return children

    def find_children_random(self, cliques):
        weights = [min(2**len(c), 2**30) for c in cliques]
        clique_indicies = random.choices(range(len(cliques)), weights=weights, k=self.beta)
        children = defaultdict(list)
        for i in clique_indicies:
            clique = cliques[i]
            n = len(clique)
            subset_weights = [ncr(n, r) for r in range(1, n+1)]
            subset_size = random.choices(range(1, n+1), weights=subset_weights, k=1)[0]
            children[clique].append(tuple(sorted(random.sample(clique, subset_size))))
        return children



def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def print_hits(dic, hyperedges):
    guess = set([x for l in dic.values() for x in l])
    total_samples = sum([len(l) for l in dic.values()])
    print((total_samples, (len(guess & hyperedges))), end=', ')