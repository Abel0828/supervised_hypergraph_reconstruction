import os.path
import sys

from model import get_model
from copy import deepcopy
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict, Counter
from motif import extract_motif_features
import pickle
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
epsilon = 1e-8
from baselines import ecc, community
from tqdm import tqdm


def train(dataloader, args, logger):
    model1 = get_model(args, dataloader)
    model2 = deepcopy(model1)

    (X_train1, y_train1), (X_train2, y_train2) = dataloader.split_train()

    if X_train1.shape[0] > 0:
        model1.fit(X_train1, y_train1)
    print(X_train2.shape, y_train2.shape,  y_train2.sum())
    if X_train2.shape[0] > 0:
        model2.fit(X_train2, y_train2)
    models = (model1, model2)
    evaluate(models, dataloader, logger)


def evaluate(models, dataloader, logger):

    if len(models) > 1:
        model1, model2 = models
    else:
        model1 = models[0]
        model2 = models[0]
    X_test, y_test, X_train, y_train = dataloader.X_test, dataloader.y_test, dataloader.X_train, dataloader.y_train
    num_max_cliques_test = dataloader.get_num_max_candidates('test')
    y_hat_test1, y_hat_test2 = np.array([]), np.array([])
    if X_test[:num_max_cliques_test].shape[0] > 0:
        y_hat_test1 = model1.predict(X_test[:num_max_cliques_test])
    if X_test[num_max_cliques_test:].shape[0] > 0:
        y_hat_test2 = model2.predict(X_test[num_max_cliques_test:])
    y_hat_test = np.hstack((y_hat_test1, y_hat_test2))
    reconstructed_cliques = set(clique for pred, clique in zip(y_hat_test, dataloader.cliques['final_cliques_test']) if pred > 0.5)
    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(reconstructed_cliques, dataloader.graphs['simplicies_test'])
    logger.info('Our Performance: precision {:.4f}, recall {:.4f}, f1 {:.4f} jaccard {:.4f}'.format(precision, recall, f1, jaccard))

    # baselines:
    if not sys.platform.startswith('win'):
        from baselines import beyesian
        beyesian_cliques = beyesian.get_beyesian_best(dataloader.graphs['G_test'], dataloader.args, dataloader.logger)
        print(len(beyesian_cliques))
    else:
        beyesian_cliques = {(-1,)}
        logger.info('Beyesian Baseline package not supported.')
    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(beyesian_cliques, dataloader.graphs['simplicies_test'])
    logger.info('Baseline: Beyesian precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(precision, recall, f1, jaccard))

    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(dataloader.cliques['max_cliques_test'], dataloader.graphs['simplicies_test'])
    logger.info('Baseline: Max Clique precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(precision, recall, f1, jaccard))

    ecc_covering = ecc.get_edge_clique_cover(dataloader.graphs['G_test'])
    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(ecc_covering,  dataloader.graphs['simplicies_test'])
    logger.info('Baseline: ECC precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(precision, recall, f1, jaccard))

    communities_demon = community.get_demon_communities(dataloader.graphs['G_test'])
    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(communities_demon, dataloader.graphs['simplicies_test'])
    logger.info('Baseline: Demon precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(precision, recall, f1, jaccard))

    communities_kclique, best_k = community.get_kclique_communities(dataloader.graphs['G_test'], dataloader.graphs['G_test'], dataloader.graphs['simplicies_train'])
    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(communities_kclique, dataloader.graphs['simplicies_test'])
    logger.info('Baseline: CFinder (k={}) precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(best_k, precision, recall, f1, jaccard))


def get_performance_wrt_ground_truth(reconstructed, ground_truth):
    correct_cliques = reconstructed & ground_truth
    precision = len(correct_cliques) / len(reconstructed) if len(reconstructed)>0 else 0
    recall = len(correct_cliques) / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if precision * recall > 0 else 0
    jaccard = len(correct_cliques) / len(reconstructed | ground_truth)
    return precision, recall, f1, jaccard


class DataLoader:
    def __init__(self, graphs, cliques, args, logger):
        self.args = args
        self.graphs = graphs
        self.cliques = cliques
        self.logger = logger
        self.feature_names = ['clique_type', 'clique_size', 'edge_degree_all_dup', 'edge_degree_mean',
                              'node_degree_mean', 'node_degree_mean_recur', 'node_degree_mean2', 'cluster_coef_mean', 'parent_cliques']
        self.node_features, self.max_candidates = {}, {}
        self.X_train, self.cliques['final_cliques_train'], self.feature_dict_train, self.node_features['train'] = self.extract_features('train')
        self.X_test, self.cliques['final_cliques_test'], self.feature_dict_test, self.node_features['test'] = self.extract_features('test')
        self.y_train = self.create_labels(mode='train')
        self.y_test = self.create_labels(mode='test')
        self.k_best = 10

        if self.k_best > 0 and self.args.features == 'motif':
            self.X_train, self.X_test = self.select_k_best(self.X_train, self.y_train, self.X_test)
        self.logger.info('test max cliques {}, nested cliques {}'.format(self.get_num_max_candidates('test'),
                                                               len(self.y_test) - self.get_num_max_candidates('test')))

        self.logger.info('test real max cliques in search universe: {}'.format(self.y_test[:self.get_num_max_candidates('test')].sum()))
        self.logger.info('test real smaller cliques in search universe: {} {}'.format(self.y_test[self.get_num_max_candidates('test'):].sum(),
              len(set(self.cliques['final_cliques_test'][self.get_num_max_candidates('test'):]) & self.graphs['simplicies_test'])))
        if self.args.save_features == 1:
            self.save_features_to_csv()

    def save_features(self):
        with open('data/{}/feature_labels.pkl'.format(self.args.dataset), 'wb') as f:
            pickle.dump({'features': self.X_test, 'label': self.y_test, 'feature_dict': self.feature_dict_test,
                         'feature_names': self.feature_names, 'cut': self.get_num_max_candidates('test')}, f)
        self.logger.info('features saved.')

    def extract_features(self, mode):
        max_cliques = self.cliques['max_cliques_{}'.format(mode)]
        child2parents = self.get_child_parents_map(mode)
        candidates_ = set(child2parents.keys())
        if self.args.ablation > 0:
            max_candidates = candidates_ & max_cliques
        else:
            max_candidates = max_cliques
        self.max_candidates[mode] = max_candidates
        nested_candidates = candidates_ - max_candidates
        final_cliques = list(max_candidates) + list(nested_candidates)

        node_degree = self.get_node_degree(mode)

        if self.args.features == 'count':
            # Now preparing for feature extraction:
            # get node degree w.r.t. max clique view

            node_degree_recur = self.get_node_degree_recur(mode)
            node_degree2 = self.get_node_degree2(mode)
            # get edge degree w.r.t. graph view
            edge_degree = self.get_edge_degree(mode)
            cluster_coef = self.get_cluster_coef(mode)

            # Now extract features for cliques
            feature_names = self.feature_names
            feature_dict = {feature_name: [] for feature_name in feature_names}
            for i, clique in tqdm(enumerate(final_cliques)):
                size = len(clique)
                feature_dict['clique_type'].append(int(i >= len(max_candidates)))
                feature_dict['clique_size'].append(size)
                edge_degrees = [edge_degree[edge] for edge in combinations(clique, 2)]
                feature_dict['edge_degree_all_dup'].append(all(d>1 for d in edge_degrees) if len(clique) > 1 else False)
                feature_dict['edge_degree_mean'].append(sum(edge_degrees) / max([(size*(size-1)/2), 1])) #
                feature_dict['node_degree_mean'].append(sum(node_degree[node] for node in clique) / size)
                feature_dict['node_degree_mean_recur'].append(sum(node_degree_recur[node] for node in clique) / size)
                feature_dict['node_degree_mean2'].append(sum(node_degree2[node] for node in clique)/size)
                feature_dict['cluster_coef_mean'].append(sum(cluster_coef[node] for node in clique)/size)
                feature_dict['parent_cliques'].append(len(child2parents[clique]) if clique in child2parents else 0)


            X = np.zeros(shape=(len(final_cliques), len(feature_dict)), dtype=np.float)
            for i, feature_name in enumerate(feature_names):
                X[:, i] = feature_dict[feature_name]

        else:
            # motif based features
            X, feature_dict = extract_motif_features(candidates=final_cliques,
                                                     H=self.cliques['max_cliques_{}'.format(mode)],
                                                     node_degree=node_degree,
                                                     use_c=self.args.use_c,
                                                     jobs=self.args.jobs,
                                                     args=self.args)
        if self.args.ext:
            X = self.extend_features(X, final_cliques, child2parents, mode)
        split = self.get_num_max_candidates(mode)
        X= (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + epsilon)  # normalize
        node_degree_arr = np.zeros(len(node_degree))
        for i in range(len(node_degree_arr)):
            node_degree_arr[i] = node_degree[i]
        one_hot_dim = int(np.quantile(node_degree_arr, 0.2) if mode == 'train' else self.node_features['train'].shape[-1])
        node_degree_arr = np.clip(node_degree_arr, a_min=None, a_max=one_hot_dim-1).astype(np.int)
        node_features = np.eye(one_hot_dim)[node_degree_arr]

        return X, final_cliques, feature_dict, node_features

    def get_child_parents_map(self, mode):
        child2parents = defaultdict(list)  # {child:[parent1, parent2, ...], child2: }
        count = 0
        for parent, children in self.cliques['children_cliques_{}'.format(mode)].items():
            n_children = len(children)
            count += n_children
            for child in children:
                child2parents[child].append(parent)
        self.logger.info('{}: {} unique candidates out of {} sampled candidates'.format(mode, len(child2parents), count))

        return child2parents

    def get_edge_degree(self, mode):
        edge2count = {}
        for clique in self.cliques['max_cliques_{}'.format(mode)]:
            for edge in combinations(clique, 2):
                edge_ = tuple(sorted(edge))
                if edge_ not in edge2count:
                    edge2count[edge_] = 1
                else:
                    edge2count[edge_] += 1
        return edge2count

    def get_node_degree(self, mode):
        return self.graphs['G_{}'.format(mode)].degree

    def get_node_degree_recur(self, mode):
        node_degree = self.get_node_degree(mode)
        node_degree_recur = {}
        for node in self.graphs['G_{}'.format(mode)].nodes:
            neighbor_degrees = [node_degree[neighbor] for neighbor in self.graphs['G_{}'.format(mode)].neighbors(node)]
            node_degree_recur[node] = np.array(neighbor_degrees).mean() if len(neighbor_degrees) > 0 else 0.0
        return node_degree_recur

    def get_node_degree2(self, mode):
        node2count = {}
        for clique in self.cliques['max_cliques_{}'.format(mode)]:
            for node in clique:
                if node not in node2count:
                    node2count[node] = 1
                else:
                    node2count[node] += 1
        return node2count

    def get_cluster_coef(self, mode):
        cluster_coef = nx.algorithms.cluster.clustering(self.graphs['G_{}'.format(mode)])
        return cluster_coef

    def create_labels(self, mode):
        Y = [int(clique in self.graphs['simplicies_{}'.format(mode)]) \
                  for clique in self.cliques['final_cliques_{}'.format(mode)]]
        Y = np.array(Y)
        return Y

    def get_preliminary_results(self, mode):
        if mode == 'train':
            Y = self.y_train
        else:
            Y = self.y_test
        precision = Y.sum() / len(Y)
        recall = Y.sum() / len(self.graphs['simplicies_{}'.format(mode)])
        return precision, recall

    def get_num_max_candidates(self, mode):
        return len(self.max_candidates[mode])

    def get_num_final_cliques(self, mode):
        return len(self.cliques['final_cliques_{}'.format(mode)])

    def get_num_nodes(self, mode):
        return len(self.graphs['G_{}'.format(mode)])

    def save_features_to_csv(self):
        a = np.concatenate([self.y_test[:, None], self.X_test], axis=1)
        np.savetxt("tmp/features.csv", a, fmt='%.3f', delimiter=",")

    def split_train(self):
        num_max_cliques_train = self.get_num_max_candidates('train')
        X_train1, y_train1 = self.X_train[:num_max_cliques_train], self.y_train[:num_max_cliques_train]
        X_train2, y_train2 = self.X_train[num_max_cliques_train:], self.y_train[num_max_cliques_train:]
        upsampled = False
        if upsampled:
            return self.upsample(X_train1, y_train1), self.upsample(X_train2, y_train2)
        else:
            return (X_train1, y_train1), (X_train2, y_train2)

    def extend_features(self, X, final_cliques, child2parents, mode):
        X_ext = deepcopy(X)
        start = self.get_num_max_candidates(mode)
        parent2features = {k: v for k, v in zip(final_cliques, X[:start])}
        for i in range(start, X.shape[0]):
            X_ext[i] = np.array([parent2features[parent] for parent in child2parents[final_cliques[i]]]).mean(axis=0)
        return np.concatenate((X_ext, X), axis=1)

    def upsample(self, X, y):
        if 2*y.sum() >= len(y):
            return X, y
        X_upsampled, y_upsampled = resample(X[y==1], y[y==1], n_samples=int((y==0).sum()/3), replace=True,
                                            random_state=self.args.seed)
        return np.vstack((X_upsampled, X[y==0])), np.hstack((y_upsampled, y[y==0]))

    def select_k_best(self, X_train, y_train, X_test):
        s = SelectKBest(k=self.k_best)
        X_train = np.hstack((X_train[:, :9], s.fit_transform(X_train[:, 9:], y_train)))
        X_test = np.hstack((X_test[:, :9], s.transform(X_test[:,9:])))

        return X_train, X_test


def get_reconstruction_properties(reconstruction):
    lengths = np.array([len(h) for h in reconstruction])
    c = Counter([n for h in reconstruction for n in h])
    avg_deg = np.array(c.values()).mean()
    return [len(reconstruction), lengths.mean(), lengths.std(), avg_deg]

