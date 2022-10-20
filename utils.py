import argparse
import networkx as nx
from itertools import combinations
import numpy as np
import random
import os
import sys
import logging
import time


def set_up():
    args, sys_argv = get_args()
    logger = set_up_logger(args, sys_argv)
    set_random_seed(args.seed)
    return args, logger


def get_args():
    parser = argparse.ArgumentParser('Interface for hypergraphs reconstruction framework')

    # key parameters: dataset, features, model, and sampling budget
    parser.add_argument('--dataset', '--d', type=str, default='dblp', help='dataset name')
    parser.add_argument('--beta', type=int, default=1e6, help='sampling budget')
    parser.add_argument('--features', type=str, default='count', help='type of features used to characterize structural property, heuristic or motif-based')
    parser.add_argument('--model', type=str, default='mlp', help='Ml model to use')
    # parser.add_argument('--gnn_model', type=str, default='GIN', help='GNN model to use, valid only when model=gnn')

    # moderate training process
    parser.add_argument('--epochs', type=int, default=2000, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='random seed')

    # more experiments
    parser.add_argument('--setting', type=str, default='f', help='fully supervised (f) or semi-supervised with 10% labels (s)')
    parser.add_argument('--label_rate', type=float, default=0.1, help='fraction of labeling in semi-supervised learning')
    parser.add_argument('--ablation', type=int, default=0, help='alternative ways to sample cliques, for ablation study')

    # parameters that have no effect on results
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--use_c', type=int, default=1, help='whether to use c++ code for extracting motif')
    parser.add_argument('--jobs', type=int, default=1, help='number of cores to use for extracting motifs')
    parser.add_argument('--save_features', type=int, default=0, help='epochs')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--log_dir', type=str, default='log', help='log root directory')
    parser.add_argument('--downsample', type=int, default=0, help='downsample neighborhood in motif extraction')

    # deprecated

    parser.add_argument('--max_child_size', type=int, default=1, help='maximum size of the child')
    parser.add_argument('--ext', type=int, default=0, help='whether to extend features')

    try:
        args = parser.parse_args()
        if args.setting == 's':
            args.dataset = args.dataset + '-s'
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def load_graphs(args, logger):
    dataset_dir = args.data_dir+args.dataset+'/'
    graphs = {}
    graphs['simplicies_train'] = read_simplicies(dataset_dir, mode='train')
    graphs['simplicies_test'] = read_simplicies(dataset_dir,  mode='test')
    graphs['G_train'] = construct_graph(graphs['simplicies_train'])
    graphs['G_test'] = construct_graph(graphs['simplicies_test'])
    logger.info('Finish loading graphs.')


    logger.info('Nodes train: {}, test: {}'.format(graphs['G_train'].number_of_nodes(), graphs['G_test'].number_of_nodes()))
    logger.info('Simplicies train: {}, test: {}'.format(len(graphs['simplicies_train']), len(graphs['simplicies_test'])))
    return graphs


def read_simplicies(file_dir, mode='train'):
    simplicies = []
    with open(file_dir + '{}.txt'.format(mode), 'r') as f:
        for line in f.readlines():
            simplicies.append(tuple(sorted(set([int(node) for node in line.strip().split(' ')]))))

    try:
        assert (len(set(simplicies)) == len(simplicies)) # no duplicate
        nodes = set([node for simplex in simplicies for node in simplex])
        assert(min(nodes) == 0)
        assert (max(nodes) == len(nodes)-1)  # compact indexing

    # sanity check
    except AssertionError:
        print('Sanity check failed, reindexing the hypergraphs ...')
        all_nodes = sorted(set(n for s in simplicies for n in s))
        node2i = {node: i for i, node in enumerate(all_nodes)}
        simplicies = [tuple(sorted(set([node2i[n] for n in s]))) for s in simplicies]

    return set(simplicies)


def construct_graph(simplicies):
    G = nx.Graph()
    for s in simplicies:
        if len(s) == 1:
            G.add_node(s[0])
            continue
        for e in combinations(s, 2):
            G.add_edge(*e)
    print('number of nodes in construct graph', G.number_of_nodes())
    return G


def set_random_seed(seed):
    seed = seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_up_logger(args, sys_argv):
    # set up running log
    runtime_id = '{}-{}-{}-{}'.format(args.dataset, str(args.beta), args.features[:3], str(time.time()))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_dir = '{}/{}/'.format(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = log_dir + runtime_id + '.log'
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger