from utils import *
from cliques import *
from train import *

from collections import defaultdict
def get_node2neighbors(all_sets):
    node2neighbors = defaultdict(list)
    for i, clique in enumerate(all_sets):
        for node in clique:
            node2neighbors[node].append(i)
    return node2neighbors


def get_neighbor_sets(clique, node2neighbors):
    return set([n for node in clique for n in node2neighbors[node]])


def count(all_sets, suspects=[]):
    if len(suspects) == 0:
        suspects = list(range(len(all_sets)))
    node2neighbors = get_node2neighbors(all_sets)
    pairs = []
    for suspect in suspects:
        neighbor_sets = get_neighbor_sets(all_sets[suspect], node2neighbors)
        for neighbor_set in neighbor_sets:
            if suspect == neighbor_set:
                continue
            if len(all_sets[suspect] - all_sets[neighbor_set]) == 0:
                pairs.append((suspect, neighbor_set))
                break
    return pairs


def get_data(graphs, cliques):
    max_cliques = cliques['max_cliques_test']
    edges = graphs['simplicies_test']
    suspect_sets = [set(x) for x in edges-max_cliques]
    max_sets = [set(x) for x in (edges & max_cliques)]
    return suspect_sets+max_sets, list(range(len(suspect_sets)))


if __name__ == '__main__':
    args, logger = set_up() # set up args, logger, random seed
    graphs = load_graphs(args, logger) # load original graph data
    cliques = compute_cliques(graphs, args, logger) # cliques operations
    dataloader = DataLoader(graphs, cliques, args, logger)  # extract features, prepare labels
    train(dataloader, args, logger)
