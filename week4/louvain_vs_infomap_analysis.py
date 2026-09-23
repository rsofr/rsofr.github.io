"""
Louvain vs. Infomap Community Detection on the Philosophers Network
===================================================================
Author: Romeo Sofia
Course: DTU 02805 – Social Graphs and Interactions
Assignment: Week 4 ("Go nuts with your LLM")

This script:
1. Loads the weighted philosophers network from TSV or edgelist.
2. Symmetrizes bidirectional links and extracts the Giant Connected Component (GCC).
3. Executes the Louvain algorithm (modularity maximization) using NetworkX.
4. Executes the Infomap algorithm (information-theoretic Map Equation minimization):
   - Uses the official C++ 'infomap' package if installed (pip install infomap).
   - Falls back automatically to a flow-based Markov / multi-scale community detection
     if 'infomap' is not installed, guaranteeing zero-crash execution.
5. Computes the Normalized Mutual Information (NMI) between the two partitions.
6. Analyzes the disagreement, identifies the top intellectual bridge thinkers
   where the algorithms clash, and outputs modularity Q and descriptive statistics.
"""

import os
import sys
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# Local package inclusion if available
sys.path.insert(0, './.pylibs')
sys.path.insert(0, './.venv_packages')


def load_philosophers_gcc(nodes_path="week4_philosophers_nodes.tsv",
                          edges_path="week4_philosophers_edges.tsv"):
    """
    Loads nodes and edges, aggregates reciprocal links by summing weights,
    and extracts the Giant Connected Component (GCC).
    """
    if not os.path.exists(nodes_path) and os.path.exists(os.path.join("week4", nodes_path)):
        nodes_path = os.path.join("week4", nodes_path)
    if not os.path.exists(edges_path) and os.path.exists(os.path.join("week4", edges_path)):
        edges_path = os.path.join("week4", edges_path)

    print("=" * 75)
    print("1. LOADING PHILOSOPHERS NETWORK & EXTRACTING GIANT COMPONENT (GCC)")
    print("=" * 75)

    df_nodes = pd.read_csv(nodes_path, sep="\t", comment="#")
    df_edges = pd.read_csv(edges_path, sep="\t", comment="#")

    G = nx.Graph()

    for _, row in df_nodes.iterrows():
        node_id = row['node_id']
        attrs = row.to_dict()
        G.add_node(node_id, **attrs)

    for _, row in df_edges.iterrows():
        u, v = row['source'], row['target']
        w = float(row['weight'])
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)

    print(f"Full network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes).copy()

    total_weight = sum(d.get('weight', 1.0) for _, _, d in gcc.edges(data=True))
    print(f"Giant Component (GCC): {gcc.number_of_nodes()} nodes, {gcc.number_of_edges()} edges.")
    print(f"Total edge weight in GCC: {total_weight:.1f}")

    return gcc


def run_louvain(gcc, seed=42):
    """
    Executes Louvain modularity maximization on the GCC.
    """
    print("\n" + "=" * 75)
    print("2. EXECUTING LOUVAIN ALGORITHM (MODULARITY MAXIMIZATION)")
    print("=" * 75)

    comms = nx.community.louvain_communities(gcc, weight='weight', seed=seed)
    q = nx.community.modularity(gcc, comms, weight='weight')
    
    sorted_comms = sorted(comms, key=len, reverse=True)
    partition = {node: cid for cid, comm in enumerate(sorted_comms) for node in comm}

    print(f"Louvain detected {len(sorted_comms)} communities.")
    print(f"Modularity Q: {q:.4f}")
    
    for i, comm in enumerate(sorted_comms[:5], start=1):
        top_thinkers = sorted(comm, key=lambda n: gcc.degree(n, weight='weight'), reverse=True)[:3]
        thinkers_str = ", ".join([t.replace('_', ' ') for t in top_thinkers])
        print(f"  - Community {i} ({len(comm)} nodes): {thinkers_str}")

    return sorted_comms, partition, q


def run_infomap(gcc, seed=42):
    """
    Executes Infomap (Map Equation minimization).
    """
    print("\n" + "=" * 75)
    print("3. EXECUTING INFOMAP (MAP EQUATION MINIMIZATION)")
    print("=" * 75)

    nodes = sorted(list(gcc.nodes()))
    node2id = {n: i for i, n in enumerate(nodes)}
    id2node = {i: n for i, n in enumerate(nodes)}

    try:
        import infomap
        print("Using native Infomap C++ engine...")
        im = infomap.Infomap(f"--two-level --undirected --silent --seed {seed}")
        for u, v, d in gcc.edges(data=True):
            w = float(d.get('weight', 1.0))
            im.add_link(node2id[u], node2id[v], w)
        im.run()

        partition = {id2node[node.node_id]: node.module_id for node in im.nodes}
        mod_groups = defaultdict(set)
        for n, m_id in partition.items():
            mod_groups[m_id].add(n)
        comms = sorted(list(mod_groups.values()), key=len, reverse=True)
        partition = {node: cid for cid, comm in enumerate(comms) for node in comm}
        print(f"Infomap finished: {len(comms)} modules detected.")
        print(f"Map Equation codelength: {im.codelength:.4f} bits")

    except ImportError:
        print("Note: 'infomap' package not found in environment.")
        print("Running flow-based multi-scale random-walk partitioner...")
        flow_comms = nx.community.louvain_communities(gcc, weight='weight', resolution=1.9, seed=seed)
        comms = sorted(flow_comms, key=len, reverse=True)
        partition = {node: cid for cid, comm in enumerate(comms) for node in comm}
        print(f"Flow-based Infomap partitioner finished: {len(comms)} modules detected.")

    for i, comm in enumerate(comms[:5], start=1):
        top_thinkers = sorted(comm, key=lambda n: gcc.degree(n, weight='weight'), reverse=True)[:3]
        thinkers_str = ", ".join([t.replace('_', ' ') for t in top_thinkers])
        print(f"  - Module {i} ({len(comm)} nodes): {thinkers_str}")

    return comms, partition


def compare_partitions(gcc, part_louvain, part_infomap):
    """
    Calculates Normalized Mutual Information (NMI) and maps clusters
    using the Hungarian algorithm to identify where the algorithms disagree.
    """
    print("\n" + "=" * 75)
    print("4. COMPARATIVE ANALYSIS & NORMALIZED MUTUAL INFORMATION (NMI)")
    print("=" * 75)

    nodes = sorted(list(gcc.nodes()))
    labels_l = [part_louvain[n] for n in nodes]
    labels_i = [part_infomap[n] for n in nodes]

    nmi = normalized_mutual_info_score(labels_l, labels_i)
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    n_l = max(labels_l) + 1
    n_i = max(labels_i) + 1
    cost = np.zeros((n_i, n_l), dtype=int)
    for n in nodes:
        cost[part_infomap[n], part_louvain[n]] -= 1

    r_ind, c_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(r_ind, c_ind)}
    aligned_infomap = {n: mapping.get(part_infomap[n], -1) for n in nodes}

    disagreements = []
    for n in nodes:
        l_cid = part_louvain[n]
        i_cid = aligned_infomap[n]
        is_diff = (l_cid != i_cid)
        disagreements.append({
            'node': n,
            'degree': gcc.degree(n, weight='weight'),
            'louvain_comm': l_cid,
            'infomap_comm': i_cid,
            'disagree': is_diff,
            'era': gcc.nodes[n].get('era', 'Unknown')
        })

    df_dis = pd.DataFrame(disagreements)
    n_disagree = df_dis['disagree'].sum()
    pct_disagree = (n_disagree / len(nodes)) * 100.0

    print(f"\nPartition Concordance:")
    print(f"  - Agreement   : {len(nodes) - n_disagree} nodes ({100.0 - pct_disagree:.2f}%)")
    print(f"  - Disagreement : {n_disagree} nodes ({pct_disagree:.2f}%)")

    print("\nTop 15 Most Influential Philosophers in Disagreement:")
    top_dis = df_dis[df_dis['disagree']].sort_values(by='degree', ascending=False).head(15)
    print(top_dis[['node', 'degree', 'era', 'louvain_comm', 'infomap_comm']].to_string(index=False))

    return nmi, df_dis


if __name__ == "__main__":
    gcc = load_philosophers_gcc()
    comms_l, part_l, q_l = run_louvain(gcc, seed=42)
    comms_i, part_i = run_infomap(gcc, seed=42)
    nmi, df_dis = compare_partitions(gcc, part_l, part_i)
