"""
Visualization of Disagreement Between Louvain and Infomap
=========================================================
Author: Romeo Sofia
Course: DTU 02805 – Social Graphs and Interactions
Assignment: Week 4 ("Go nuts with your LLM")

This script:
1. Loads the philosophers GCC network and runs both Louvain and Infomap.
2. Identifies all nodes where the partitions diverge using Hungarian matching.
3. Generates a publication-grade, dark-themed multi-panel visualization:
   - Panel A: Network subgraph highlighting the "Nodes of Discord" (disagreement)
     in vibrant coral/orange with degree-scaled sizes and crisp annotations for key thinkers.
   - Panel B: Modularity resolution limit breakdown (Louvain clusters vs. Infomap splits).
   - Panel C: Method disagreement rate across historical eras.
4. Saves high-resolution figures (300 DPI) to PNG.
"""

import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local path inclusion
sys.path.insert(0, './.pylibs')
sys.path.insert(0, './.venv_packages')

from louvain_vs_infomap_analysis import load_philosophers_gcc, run_louvain, run_infomap


def generate_disagreement_visualization(gcc, part_louvain, part_infomap,
                                       save_paths=["week4/louvain_vs_infomap_disagreement.png",
                                                   "rsofr.github.io/assets/louvain_vs_infomap_disagreement.png"]):
    """
    Renders and exports the multi-panel disagreement visualization.
    """
    print("\n" + "=" * 75)
    print("GENERATING PUBLICATION-QUALITY DISAGREEMENT VISUALIZATION")
    print("=" * 75)

    nodes = sorted(list(gcc.nodes()))
    labels_l = [part_louvain[n] for n in nodes]
    labels_i = [part_infomap[n] for n in nodes]

    # Kuhn-Munkres alignment
    n_l = max(labels_l) + 1
    n_i = max(labels_i) + 1
    cost = np.zeros((n_i, n_l), dtype=int)
    for n in nodes:
        cost[part_infomap[n], part_louvain[n]] -= 1

    r_ind, c_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(r_ind, c_ind)}
    aligned_infomap = {n: mapping.get(part_infomap[n], -1) for n in nodes}

    disagree_nodes = {n for n in nodes if part_louvain[n] != aligned_infomap[n]}

    # Create figure layout
    fig = plt.figure(figsize=(20, 10), facecolor='#0d1117')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.32, wspace=0.25)
    ax_net = fig.add_subplot(gs[:, 0], facecolor='#0d1117')
    ax_heat = fig.add_subplot(gs[0, 1], facecolor='#161b22')
    ax_era = fig.add_subplot(gs[1, 1], facecolor='#161b22')

    # 1. NETWORK SUBGRAPH OF TOP 350 THINKERS
    top_nodes = sorted(nodes, key=lambda n: gcc.degree(n, weight='weight'), reverse=True)[:350]
    subG = gcc.subgraph(top_nodes)
    pos = nx.spring_layout(subG, k=0.19, seed=42, iterations=100)

    # Subtle edges
    nx.draw_networkx_edges(subG, pos, ax=ax_net, alpha=0.12, edge_color='#8b949e', width=0.8)

    sub_agree = [n for n in top_nodes if n not in disagree_nodes]
    sub_disagree = [n for n in top_nodes if n in disagree_nodes]

    # Concordant nodes (subdued scientific blue)
    nx.draw_networkx_nodes(subG, pos, nodelist=sub_agree, ax=ax_net,
                           node_color='#388bfd', node_size=[gcc.degree(n)*0.6 + 25 for n in sub_agree],
                           alpha=0.45, edgecolors='#1f6feb', linewidths=0.8)

    # Discordant nodes (vibrant coral with glowing orange border)
    nx.draw_networkx_nodes(subG, pos, nodelist=sub_disagree, ax=ax_net,
                           node_color='#ff7b72', node_size=[gcc.degree(n)*0.9 + 40 for n in sub_disagree],
                           alpha=0.90, edgecolors='#f0883e', linewidths=1.8)

    # Curated, non-overlapping annotations for key philosophical giants
    offsets = {
        'Aristotle': (0.10, 0.04),
        'David_Hume': (0.08, -0.02),
        'Edmund_Husserl': (0.06, -0.04),
        'Augustine_of_Hippo': (-0.22, -0.05),
        'Martin_Heidegger': (-0.18, 0.02),
        'Henri_Bergson': (-0.22, 0.07),
        'Jean-Jacques_Rousseau': (-0.20, -0.05),
        'Albert_Einstein': (0.06, 0.06),
        'Adam_Smith': (-0.18, 0.04),
        'Galileo_Galilei': (0.08, -0.03)
    }

    for g, (dx, dy) in offsets.items():
        if g in pos:
            x, y = pos[g]
            label = g.replace('_', ' ')
            ax_net.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                            fontsize=9, fontweight='bold', color='#ffffff',
                            bbox=dict(boxstyle='round,pad=0.28', facecolor='#21262d', edgecolor='#ff7b72', alpha=0.92),
                            arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.3, connectionstyle='arc3,rad=0.08'))

    ax_net.set_title('Network of Disagreement: Louvain vs. Infomap\n(Coral = Method Discordance | Blue = Method Concordance)',
                     color='#ffffff', fontsize=14, fontweight='bold', pad=14)
    ax_net.axis('off')

    # 2. RESOLUTION LIMIT BAR PLOT (LOUVAIN CLUSTERS VS INFOMAP SPLITS)
    comms_l_groups = defaultdict(list)
    for n, cid in part_louvain.items():
        comms_l_groups[cid].append(n)

    top_louvain_names = {}
    for cid in range(5):
        top_2 = sorted(comms_l_groups[cid], key=lambda n: gcc.degree(n, weight='weight'), reverse=True)[:2]
        top_louvain_names[cid] = ' / '.join([n.replace('_', ' ') for n in top_2])

    cross_data = []
    for cid in range(5):
        c_nodes = comms_l_groups[cid]
        match_cnt = (pd.Series([aligned_infomap[n] for n in c_nodes]) == cid).sum()
        diverge_cnt = len(c_nodes) - match_cnt
        cross_data.append({'Cluster': top_louvain_names[cid], 'Concordant': match_cnt, 'Divergent (Shattered)': diverge_cnt})

    df_cross = pd.DataFrame(cross_data).set_index('Cluster')
    y_pos = np.arange(len(df_cross))
    ax_heat.barh(y_pos - 0.18, df_cross['Concordant'], height=0.35, color='#388bfd', label='Concordant (Core School)')
    ax_heat.barh(y_pos + 0.18, df_cross['Divergent (Shattered)'], height=0.35, color='#ff7b72', label='Divergent (Shattered by Flow)')
    ax_heat.set_yticks(y_pos)
    ax_heat.set_yticklabels(df_cross.index, color='#c9d1d9', fontsize=9.5)
    ax_heat.set_xlabel('Number of Philosophers', color='#c9d1d9', fontsize=10.5)
    ax_heat.set_title('Modularity Resolution Limit: Louvain Blobs Shattered by Infomap Flow', color='#ffffff', fontsize=12.5, fontweight='bold', pad=10)
    ax_heat.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#ffffff', fontsize=9.5)
    ax_heat.tick_params(colors='#8b949e')
    ax_heat.grid(axis='x', alpha=0.15, color='#8b949e')

    # 3. DISAGREEMENT RATE BY HISTORICAL ERA
    era_series = pd.Series([gcc.nodes[n].get('era', 'Unknown') for n in nodes])
    era_dis = pd.DataFrame({'era': era_series, 'disagree': [n in disagree_nodes for n in nodes]})
    top_eras = era_dis['era'].value_counts()
    top_eras = top_eras[top_eras >= 25].index.tolist()
    era_stats = era_dis[era_dis['era'].isin(top_eras)].groupby('era')['disagree'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

    y_era = np.arange(len(era_stats))
    bars = ax_era.barh(y_era, era_stats['mean'] * 100, color='#f0883e', alpha=0.85, edgecolor='#ff7b72', height=0.55)
    ax_era.set_yticks(y_era)
    clean_era_labels = [e.replace('centuries', 'c.').replace('through', '-').replace('century', 'c.') for e in era_stats.index]
    ax_era.set_yticklabels(clean_era_labels, color='#c9d1d9', fontsize=9.5)
    ax_era.set_xlabel('Method Disagreement Rate (%)', color='#c9d1d9', fontsize=10.5)
    ax_era.set_title('Where Historiography Clashes: Disagreement Rate Across Eras', color='#ffffff', fontsize=12.5, fontweight='bold', pad=10)
    ax_era.tick_params(colors='#8b949e')
    ax_era.grid(axis='x', alpha=0.15, color='#8b949e')

    for bar, (idx, row) in zip(bars, era_stats.iterrows()):
        w = bar.get_width()
        ax_era.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}% (n={int(row["count"])})',
                    va='center', color='#ffffff', fontsize=8.5, fontweight='bold')

    ax_era.set_xlim(0, max(era_stats['mean']*100) + 18)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08, wspace=0.35, hspace=0.32)

    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Figure saved to: {p}")

    plt.close()


if __name__ == "__main__":
    gcc = load_philosophers_gcc()
    _, part_l, _ = run_louvain(gcc, seed=42)
    _, part_i = run_infomap(gcc, seed=42)
    generate_disagreement_visualization(gcc, part_l, part_i)
