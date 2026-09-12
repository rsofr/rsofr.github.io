#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DTU SOCIAL GRAPHS & INTERACTIONS - WEEK 2
VISUALISATION D'UN RÉSEAU ALÉATOIRE PROCHE DU POINT CRITIQUE (⟨k⟩ = 1.5)
===============================================================================

Ce script génère un réseau d'Erdős-Rényi G(n=200, ⟨k⟩=1.5) au-dessus du seuil
critique de percolation (⟨k⟩_c = 1.0), extrait sa composante géante, sélectionne
un nœud source (root_node) et met en évidence ses voisins à distance géodésique 2.

Codage couleur des nœuds :
  - Nœud de départ (root_node, d = 0) : Noir (black)
  - Nœuds à exactement 2 étapes (d = 2) : Rouge (red)
  - Tous les autres nœuds de la composante : Bleu pâle (lightblue)
===============================================================================
"""

import os
import sys
import types
import random
import numpy as np

# Configuration du cache Matplotlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')

# Compatibilité et shims légers pour environnements minimaux
try:
    import packaging
except ImportError:
    try:
        import setuptools._vendor.packaging as pkg
        sys.modules['packaging'] = pkg
        sys.modules['packaging.version'] = pkg.version
    except ImportError:
        pass

try:
    import dateutil
    import dateutil.parser
except ImportError:
    dateutil = types.ModuleType('dateutil')
    dateutil.__version__ = '2.8.2'
    parser = types.ModuleType('dateutil.parser')
    parser.parse = lambda s, **kwargs: None
    dateutil.parser = parser
    rrule = types.ModuleType('dateutil.rrule')
    for const in [
        'rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU',
        'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY', 'SECONDLY'
    ]:
        setattr(rrule, const, 0)
    dateutil.rrule = rrule
    relativedelta_mod = types.ModuleType('dateutil.relativedelta')
    class relativedelta: pass
    relativedelta_mod.relativedelta = relativedelta
    dateutil.relativedelta = relativedelta_mod
    tz_mod = types.ModuleType('dateutil.tz')
    tz_mod.tzlocal = lambda: None
    dateutil.tz = tz_mod
    sys.modules['dateutil'] = dateutil
    sys.modules['dateutil.parser'] = parser
    sys.modules['dateutil.rrule'] = rrule
    sys.modules['dateutil.relativedelta'] = relativedelta_mod
    sys.modules['dateutil.tz'] = tz_mod

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visualize_critical_network(
    n=200,
    target_k=1.5,
    graph_seed=42,
    root_seed=42,
    layout_seed=42,
    output_img='week2/critical_network_visualization.png'
):
    """
    Génère, analyse et visualise le réseau aléatoire et la composante géante.
    """
    print("\n" + "=" * 78)
    print("🌐 VISUALISATION D'UN RÉSEAU ALÉATOIRE AU VOISINAGE DU POINT CRITIQUE")
    print("=" * 78)

    # 1. Calcul de la probabilité de liaison
    p = target_k / float(n - 1)
    print(f"  • Nombre total de nœuds (n)  : {n}")
    print(f"  • Degré moyen visé ⟨k⟩       : {target_k}")
    print(f"  • Probabilité de lien (p)    : {p:.6f} (soit {target_k}/{n-1})")

    # 2. Génération du graphe d'Erdős-Rényi
    G = nx.erdos_renyi_graph(n, p, seed=graph_seed)
    num_edges = G.number_of_edges()
    print(f"  • Arêtes générées            : {num_edges}")

    # 3. Extraction de la composante géante (Largest Connected Component)
    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    largest_cc = max(connected_components, key=len)
    subG = G.subgraph(largest_cc).copy()

    gcc_size = subG.number_of_nodes()
    gcc_edges = subG.number_of_edges()
    gcc_fraction = gcc_size / float(n)

    print(f"  • Nombre total de composantes: {num_components}")
    print(f"  • Taille de la composante GCC: {gcc_size} nœuds ({gcc_fraction:.1%} du réseau)")
    print(f"  • Arêtes dans la GCC         : {gcc_edges}")

    # 4. Choix d'un nœud source (root_node) au hasard dans la composante géante
    rng = random.Random(root_seed)
    # Tri préalable pour reproductibilité
    nodes_list = sorted(list(subG.nodes()))
    root_node = rng.choice(nodes_list)
    root_degree = subG.degree[root_node]

    print(f"\n🎯 Nœud de départ (root_node)  : Nœud #{root_node} (degré: {root_degree})")

    # 5. Calcul des distances géodésiques depuis root_node
    # nx.single_source_shortest_path_length retourne {node: distance, ...}
    distances = nx.single_source_shortest_path_length(subG, root_node)

    # Identification des nœuds selon leur distance
    dist_0_nodes = [node for node, d in distances.items() if d == 0]  # [root_node]
    dist_1_nodes = [node for node, d in distances.items() if d == 1]
    dist_2_nodes = [node for node, d in distances.items() if d == 2]
    dist_other_nodes = [node for node, d in distances.items() if d not in (0, 2)]

    print(f"  • Nœuds à distance 1 (voisins directs) : {len(dist_1_nodes)}")
    print(f"  • Nœuds à distance 2 (2 étapes)        : {len(dist_2_nodes)}")
    print(f"  • Nœuds à distance >= 3                : {len(dist_other_nodes) - len(dist_1_nodes)}")
    print(f"  • Distance max dans la GCC             : {max(distances.values())} étapes")

    # 6. Attribution des couleurs et des tailles
    # - Nœud de départ : Noir (black)
    # - Nœuds à distance 2 : Rouge (red)
    # - Tous les autres nœuds : Bleu pâle (lightblue)
    node_colors = []
    node_sizes = []
    node_edge_colors = []
    node_linewidths = []

    for node in subG.nodes():
        d = distances.get(node, -1)
        if d == 0:
            node_colors.append('black')
            node_sizes.append(160)
            node_edge_colors.append('#FFFFFF')
            node_linewidths.append(1.8)
        elif d == 2:
            node_colors.append('red')
            node_sizes.append(90)
            node_edge_colors.append('#991B1B')
            node_linewidths.append(0.8)
        else:
            node_colors.append('lightblue')
            node_sizes.append(70)
            node_edge_colors.append('#64748B')
            node_linewidths.append(0.6)

    # 7. Disposition spatiale (Layout)
    print("\n📐 Calcul du layout spatial (Fruchterman-Reingold / spring layout)...")
    pos = nx.spring_layout(subG, seed=layout_seed, k=0.28, iterations=120)

    # 8. Visualisation avec Matplotlib
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300, facecolor='white')
    ax.set_facecolor('#F8FAFC')

    # Tracé des arêtes discrètes et fines
    nx.draw_networkx_edges(
        subG,
        pos,
        ax=ax,
        edge_color='#94A3B8',
        width=0.75,
        alpha=0.45
    )

    # Tracé des nœuds
    nx.draw_networkx_nodes(
        subG,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=node_linewidths
    )

    # Étiquette discrète pour le nœud racine
    root_x, root_y = pos[root_node]
    ax.text(
        root_x, root_y + 0.045,
        f"Racine #{root_node}",
        fontsize=9,
        fontweight='bold',
        ha='center',
        va='bottom',
        color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#0F172A', alpha=0.85, linewidth=0.8),
        zorder=10
    )

    # Titre explicite
    ax.set_title(
        f"Composante géante d'un réseau aléatoire critique $G(n={n}, p=1.5/199)$\n"
        f"Degré moyen $\\langle k \\rangle = 1.5$ — Nœud racine et voisins à distance 2",
        fontsize=13,
        fontweight='bold',
        pad=16,
        color='#0F172A'
    )

    # Légende personnalisée avec code couleur strict
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'Nœud de départ (root_node #{root_node}) : Noir',
            markerfacecolor='black',
            markeredgecolor='white',
            markeredgewidth=1.2,
            markersize=11
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'Nœuds à distance géodésique 2 ({len(dist_2_nodes)} nœuds) : Rouge',
            markerfacecolor='red',
            markeredgecolor='#991B1B',
            markersize=9
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'Autres nœuds ({len(dist_other_nodes)} nœuds) : Bleu pâle (lightblue)',
            markerfacecolor='lightblue',
            markeredgecolor='#64748B',
            markersize=8
        ),
        Line2D(
            [0], [0],
            color='#94A3B8',
            lw=1.5,
            alpha=0.6,
            label=f'Arêtes ({gcc_edges} liens)'
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper left',
        framealpha=0.95,
        edgecolor='#CBD5E1',
        fontsize=9.5
    )

    # Cartouche d'informations statistiques
    info_text = (
        "Propriétés du réseau :\n"
        f" • Taille globale $n = {n}$\n"
        f" • Degré moyen $\\langle k \\rangle = {target_k:.1f}$\n"
        f" • Composante géante : {gcc_size}/{n} nœuds ({gcc_fraction:.1%})\n"
        f" • Distance 1 (voisins) : {len(dist_1_nodes)}\n"
        f" • Distance 2 (cibles) : {len(dist_2_nodes)}\n"
        f" • Distance max (rayon) : {max(distances.values())} pas"
    )
    ax.text(
        0.98, 0.03,
        info_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='white', edgecolor='#CBD5E1', alpha=0.92),
        zorder=5
    )

    # Supprimer les bordures d'axes et ajouter une marge d'aération
    ax.margins(0.08)
    ax.axis('off')
    plt.tight_layout()

    # Sauvegarde
    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
    fig.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"  💾 Graphique exporté avec succès : {output_img}")

    return {
        'G': G,
        'subG': subG,
        'root_node': root_node,
        'dist_2_nodes': dist_2_nodes,
        'distances': distances,
        'output_img': output_img
    }


def main():
    visualize_critical_network(
        n=200,
        target_k=1.5,
        graph_seed=42,
        root_seed=42,
        layout_seed=42,
        output_img='week2/critical_network_visualization.png'
    )
    print("\n" + "=" * 78)
    print("✅ VISUALISATION DU RÉSEAU CRITIQUE TERMINÉE AVEC SUCCÈS")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
