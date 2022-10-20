import networkx
from copy import deepcopy
from itertools import combinations


def get_edge_clique_cover(G):
    G_uncovered = deepcopy(G)
    edges = set(G.edges)
    cliques = []
    while len(edges) > 0:
        u, v = edges.pop()  # select uncovered edge
        G_uncovered.remove_edge(u, v)

        clique = find_clique_of(G, (u, v), G_uncovered)
        cliques.append(clique)

        new_coverage = set(combinations(clique, 2))
        edges -= new_coverage
        edges -= set([(j, i) for i, j in new_coverage])
        G_uncovered.remove_edges_from(new_coverage)
    return set([tuple(sorted(c)) for c in cliques])


def find_clique_of(G, edge, G_uncovered):
    u, v = edge
    R = {u, v}
    P = set(G.neighbors(u)) & set(G.neighbors(v))
    z = extract_node(P, G_uncovered, R)
    while z is not None:
        R.add(z)
        P = P & set(G.neighbors(z))
        z = extract_node(P, G_uncovered, R)
    return set(R)


def extract_node(P, G_uncovered, R):
    z = None
    best_coverage = 0
    for p in P:
        coverage = sum([G_uncovered.has_edge(p, r) for r in R])
        if coverage > best_coverage:
            z = p
            best_coverage = coverage
    return z


if __name__ == '__main__':
    G = networkx.Graph([(1,2), (2,3) , (3,1), (3,4), (4, 5), (5,6), (6,4), (1,6)])
    print(get_edge_clique_cover(G))




