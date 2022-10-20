import networkx as nx
from itertools import combinations
import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
niter_beyesian = 200


def make_gt_graph(G: nx.Graph):
    gt_G = gt.Graph()
    vlist = list(gt_G.add_vertex(G.number_of_nodes()))
    for s, t in G.edges:
        gt_G.add_edge(s, t)

    return gt_G


def get_beyesian_best(G: nx.Graph, args, logger):
    gt_G = make_gt_graph(G)
    cliques, nflips, mdl,_ = get_state(gt_G, niter_beyesian, seed=args.seed)
    return cliques


def check_beyesian_best(G: nx.Graph, args, logger):
    gt_G = make_gt_graph(G)
    nflips_l = [0]
    n_cliques_l = []
    gt.seed_rng(args.seed)
    np.random.seed(args.seed)
    state = gt.CliqueState(gt_G)
    cliques = set()
    for i in range(niter_beyesian):
        new_cliques, nflips, mdl, state = renew_state(state)
        n_cliques_l.append(len(new_cliques))
        nflips_l.append(nflips)
        if nflips > 0:
            logger.info('Iteration {}'.format(i))
            logger.info('{} nflips {} hyperedges in this iteration (#added + #deleted)'.format(nflips, len(new_cliques)))
            logger.info('total nflips {}'.format(nflips))
            logger.info('added cliques')
            logger.info(len(new_cliques - cliques))
            logger.info('deleted cliques')
            logger.info(len(cliques - new_cliques))
        cliques = new_cliques

    logger.info(n_cliques_l)
    plt.figure()
    plt.plot(n_cliques_l, '-o', ms=2)
    plt.xlabel('#iterations')
    plt.ylabel('#hyperedges')
    plt.savefig('log/{}/beyesian.png'.format(args.dataset))
    return cliques


def get_state(gt_G, niter, seed):
    gt.seed_rng(seed)
    np.random.seed(seed)
    state = gt.CliqueState(gt_G)
    mdl, nflips = state.mcmc_sweep(niter=niter)
    cliques = []
    for v in state.f.vertices():      # iterate through factor graph
        if state.is_fac[v]:
            continue                  # skip over factors
#         print(tuple(sorted(state.c[v])), state.x[v]) # clique occupation
        if state.x[v] > 0:
            cliques.append(tuple(sorted(state.c[v])))
    cliques = set(cliques)
    return cliques, nflips, mdl, state


def renew_state(state):
    mdl, nflips = state.mcmc_sweep(niter=1)
    cliques = []
    for v in state.f.vertices():  # iterate through factor graph
        if state.is_fac[v]:
            continue  # skip over factors
        if state.x[v] > 0:
            cliques.append(tuple(sorted(state.c[v])))
    cliques = set(cliques)
    return cliques, nflips, mdl, state

