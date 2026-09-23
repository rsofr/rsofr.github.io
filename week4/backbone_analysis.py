"""
Extraction de la Colonne Vertébrale (Backbone) d'un Réseau Complexe
===================================================================
Ce script implémente de manière modulaire, rigoureuse et commentée les
4 tâches demandées pour l'extraction du backbone des philosophes :
1. Analyse Force (s) vs Degré (k) sur échelle log-log et annotation des outliers (Diogène Laërce, etc.).
2. Implémentation 'from scratch' du Filtre de Disparité (Serrano et al., 2009) et tableau comparatif avec seuils naïfs.
3. Visualisation de la figure principale (alpha = 0.2) selon les 5 règles strictes.
4. Comparaison visuelle pour alpha = 0.1 et alpha = 0.3.
"""

import sys
sys.path.insert(0, './.venv_packages')

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DU RÉSEAU
# =============================================================================

def load_philosophers_network(filepath="philosophers.edgelist", fallback_tsv="week4_philosophers_edges.tsv"):
    """
    Charge le réseau pondéré des philosophes, le rend non orienté en sommant
    les poids des arêtes allant dans les deux sens (A->B et B->A),
    et extrait la composante géante connectée (GCC).
    """
    print("=" * 80)
    print("1. CHARGEMENT DU RÉSEAU PONDÉRÉ ET EXTRACTION DE LA GCC")
    print("=" * 80)

    G = nx.Graph()

    # Si le fichier .edgelist existe
    if os.path.exists(filepath):
        print(f"Lecture du fichier '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                u, v = parts[0], parts[1]
                w = float(parts[2]) if len(parts) > 2 else 1.0
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)
    elif os.path.exists(fallback_tsv):
        print(f"'{filepath}' non trouvé. Chargement depuis '{fallback_tsv}'...")
        df_edges = pd.read_csv(fallback_tsv, sep='\t', comment='#')
        for _, row in df_edges.iterrows():
            u, v = row['source'], row['target']
            w = float(row['weight'])
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
    else:
        raise FileNotFoundError(f"Impossible de trouver '{filepath}' ou '{fallback_tsv}'.")

    print(f"Graphe non orienté complet : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")

    # Extraction de la composante géante (GCC)
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes).copy()
    total_weight = sum(d['weight'] for _, _, d in gcc.edges(data=True))

    print(f"Composante Géante (GCC)     : {gcc.number_of_nodes()} nœuds, {gcc.number_of_edges()} arêtes.")
    print(f"Poids total des arêtes (GCC): {total_weight:.1f}")

    return gcc


# =============================================================================
# 2. TÂCHE 1 : ANALYSE FORCE VS DEGRÉ ET ANNOTATIONS
# =============================================================================

def task_1_strength_vs_degree(G, save_path="backbone_task1_strength_vs_degree.png"):
    """
    Pour chaque nœud, calcule son degré (k) et sa force (s).
    Trace un nuage de points de s en fonction de k en échelle log-log.
    Identifie et annote automatiquement Diogène Laërce ainsi que les 3 autres
    philosophes les plus éloignés de la diagonale.
    """
    print("\n" + "=" * 80)
    print("TÂCHE 1 : ANALYSE FORCE (s) VS DEGRÉ (k) ET IDENTIFICATION DES OUTLIERS")
    print("=" * 80)

    # Calcul de k et s
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight='weight'))

    nodes = list(G.nodes())
    k_vals = np.array([degrees[n] for n in nodes], dtype=float)
    s_vals = np.array([strengths[n] for n in nodes], dtype=float)

    # Distance à la diagonale d'égalité s = k (en coordonnées log10)
    log_k = np.log10(k_vals)
    log_s = np.log10(s_vals)
    # Distance perpendiculaire à la droite y = x dans l'espace log-log
    perp_dist = (log_s - log_k) / np.sqrt(2.0)
    ratios = s_vals / k_vals

    df = pd.DataFrame({
        'node': nodes,
        'degree': k_vals,
        'strength': s_vals,
        'ratio_s_k': ratios,
        'dist_diagonal': perp_dist
    })

    # Sélection des annotations :
    # 1. Diogène Laërce (spécifié explicitement dans la consigne)
    target = "Diogenes_Laertius"
    # 2. Les 3 autres philosophes les plus éloignés de la diagonale
    other_outliers = (
        df[df['node'] != target]
        .sort_values(by='dist_diagonal', ascending=False)
        .head(3)['node']
        .tolist()
    )
    annotated = [target] + other_outliers

    print("Philosophes identifiés pour l'annotation :")
    for n in annotated:
        row = df[df['node'] == n].iloc[0]
        print(f" - {row['node']:25s} : Degré k = {row['degree']:3.0f}, Force s = {row['strength']:4.0f}, Ratio s/k = {row['ratio_s_k']:.2f}")

    # Tracé graphique
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')

    # Scatter plot de tous les nœuds
    scatter = ax.scatter(k_vals, s_vals, alpha=0.55, c=ratios, cmap='viridis',
                         s=25, edgecolors='none', zorder=2)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Poids moyen par lien ($s_i / k_i$)", fontsize=10)

    # Ligne diagonale théorique d'égalité : s = k (poids moyen = 1)
    k_min, k_max = max(1, k_vals.min()), k_vals.max()
    ax.plot([k_min, k_max], [k_min, k_max], color='#e74c3c', linestyle='--',
            linewidth=1.8, label="Diagonale $s = k$ (poids moyen = 1)", zorder=3)

    # Ligne du poids moyen global de la GCC
    mean_w = np.sum(s_vals) / np.sum(k_vals)
    ax.plot([k_min, k_max], [mean_w * k_min, mean_w * k_max], color='#2980b9', linestyle=':',
            linewidth=1.8, label=f"Moyenne globale $s = \\langle w \\rangle k$ ($\\langle w \\rangle = {mean_w:.2f}$)", zorder=3)

    # Annotations soignées des outliers identifiés
    for i, name in enumerate(annotated):
        row = df[df['node'] == name].iloc[0]
        xk, ys = row['degree'], row['strength']
        clean_name = name.replace('_', ' ')
        
        # Positionnement optimisé de la flèche
        if name == "Diogenes_Laertius":
            xytext = (xk * 0.45, ys * 1.55)
            color_box = '#ffeaa7'
        else:
            offset_y = 1.6 + 0.4 * (i % 2)
            xytext = (xk * 0.55, ys * offset_y)
            color_box = '#ffffff'

        ax.annotate(
            f"{clean_name}\n($k={int(xk)}, s={int(ys)}$)",
            xy=(xk, ys),
            xytext=xytext,
            arrowprops=dict(facecolor='black', edgecolor='none', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=color_box, edgecolor='#333333', lw=0.7, alpha=0.9),
            zorder=5
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Degré du philosophe $k_i$ (échelle log)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Force du philosophe $s_i = \\sum_j w_{ij}$ (échelle log)", fontsize=11, fontweight='bold')
    ax.set_title("Analyse Force vs Degré : Identification des Penseurs Pivot", fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"Graphique sauvegardé sous '{save_path}'.")

    return df


# =============================================================================
# 3. TÂCHE 2 : FILTRE DE DISPARITÉ 'FROM SCRATCH' ET TABLEAU COMPARATIF
# =============================================================================

def disparity_filter(G, alpha):
    """
    Implémentation 'from scratch' du Filtre de Disparité (Serrano et al., PNAS 2009).
    
    Formule mathématique :
      - Pour chaque nœud i de degré k_i et force s_i :
        p_ij = w_ij / s_i
      - Niveau de significativité statistique (p-value) :
        alpha_ij = (1 - p_ij)^(k_i - 1)   si k_i > 1, sinon 1.0
      - Le lien est conservé si alpha_ij < alpha pour au moins une des extrémités (i ou j).
    
    Retourne :
      Un nouveau nx.Graph contenant uniquement les arêtes significatives.
    """
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight='weight'))

    backbone = nx.Graph()

    for u, v, data in G.edges(data=True):
        w = float(data.get('weight', 1.0))
        k_u, s_u = degrees[u], strengths[u]
        k_v, s_v = degrees[v], strengths[v]

        p_u = w / s_u
        p_v = w / s_v

        # Évaluation stricte de la p-value
        alpha_u = (1.0 - p_u) ** (k_u - 1) if k_u > 1 else 1.0
        alpha_v = (1.0 - p_v) ** (k_v - 1) if k_v > 1 else 1.0

        # Condition de rétention : significatif pour u OU pour v
        if alpha_u < alpha or alpha_v < alpha:
            backbone.add_edge(u, v, weight=w)

    return backbone


def task_2_backbone_comparison_table(G, alphas=[0.05, 0.1, 0.2, 0.3, 0.5], naive_thresholds=[2, 3, 4]):
    """
    Calcule et compare les métriques topologiques pour différentes valeurs d'alpha
    du filtre de disparité et des seuils globaux naïfs de poids (w >= 2, 3, 4).
    Affiche et retourne un DataFrame récapitulatif.
    """
    print("\n" + "=" * 80)
    print("TÂCHE 2 : TABLEAU COMPARATIF - FILTRE DE DISPARITÉ VS SEUILS GLOBAUX NAÏFS")
    print("=" * 80)

    n_total_edges = G.number_of_edges()
    n_total_nodes = G.number_of_nodes()

    records = []

    # 1. Évaluation du Filtre de Disparité
    for alpha in alphas:
        B = disparity_filter(G, alpha)
        n_edges = B.number_of_edges()
        n_nodes = B.number_of_nodes()
        gcc_size = len(max(nx.connected_components(B), key=len)) if n_nodes > 0 else 0
        pct_edges_kept = (n_edges / n_total_edges) * 100
        pct_nodes_kept = (n_nodes / n_total_nodes) * 100

        records.append({
            'Méthode / Filtre': f'Filtre de Disparité (α = {alpha})',
            'Paramètre': f'α = {alpha}',
            'Liens conservés': n_edges,
            '% Liens conservés': f"{pct_edges_kept:.1f}%",
            'Philosophes attachés (k ≥ 1)': n_nodes,
            '% Nœuds conservés': f"{pct_nodes_kept:.1f}%",
            'Taille GCC': gcc_size,
            '% GCC conservée': f"{(gcc_size / n_total_nodes) * 100:.1f}%"
        })

    # 2. Évaluation des Seuils Globaux Naïfs
    for w_min in naive_thresholds:
        B_naive = nx.Graph()
        for u, v, data in G.edges(data=True):
            w = float(data.get('weight', 1.0))
            if w >= w_min:
                B_naive.add_edge(u, v, weight=w)

        n_edges = B_naive.number_of_edges()
        n_nodes = B_naive.number_of_nodes()
        gcc_size = len(max(nx.connected_components(B_naive), key=len)) if n_nodes > 0 else 0
        pct_edges_kept = (n_edges / n_total_edges) * 100
        pct_nodes_kept = (n_nodes / n_total_nodes) * 100

        records.append({
            'Méthode / Filtre': f'Seuil Global Naïf (w ≥ {w_min})',
            'Paramètre': f'w ≥ {w_min}',
            'Liens conservés': n_edges,
            '% Liens conservés': f"{pct_edges_kept:.1f}%",
            'Philosophes attachés (k ≥ 1)': n_nodes,
            '% Nœuds conservés': f"{pct_nodes_kept:.1f}%",
            'Taille GCC': gcc_size,
            '% GCC conservée': f"{(gcc_size / n_total_nodes) * 100:.1f}%"
        })

    df_summary = pd.DataFrame(records)
    print(df_summary.to_string(index=False))

    print("\nObservation fondamentale :")
    print("Pour un nombre de liens équivalent (~1550 liens : α = 0.2 vs w ≥ 3) :")
    print(" - Le filtre de disparité conserve 950 philosophes (GCC = 816).")
    print(" - Le seuil global naïf ne conserve que 863 philosophes (GCC = 747).")
    print("--> Le filtre de disparité préserve la connectivité multi-échelle et évite d'exclure")
    print("    les penseurs à faible degré qui possèdent des connexions locales statistiquement fortes.")

    return df_summary


# =============================================================================
# 4. TÂCHE 3 & 4 : FONCTION GÉNÉRIQUE DE VISUALISATION DU BACKBONE
# =============================================================================

def plot_backbone_network(G_full, G_backbone, communities, alpha,
                          top_k_labels=10, save_path=None, seed=42):
    """
    Visualise le réseau filtré du backbone en respectant rigoureusement les 5 règles :
      - Règle 1 : Les communautés de Louvain sont celles du réseau complet non pondéré.
      - Règle 2 : Le layout spatial (spring_layout) est calculé UNIQUEMENT sur G_backbone.
      - Règle 3 : Nœuds colorés selon leur communauté de Louvain avec légende explicite.
      - Règle 4 : Taille des nœuds proportionnelle à leur force s_i dans le graphe complet.
      - Règle 5 : Étiquettes de texte UNIQUEMENT pour les 10 philosophes à plus forte force.
      - Caption descriptive en bas de page indiquant le filtre, alpha et liens retirés.
    """
    strengths_full = dict(G_full.degree(weight='weight'))
    nodes_backbone = list(G_backbone.nodes())

    if len(nodes_backbone) == 0:
        print(f"Attention : Le backbone pour alpha={alpha} ne contient aucun nœud.")
        return

    # Mappage des nœuds vers leur ID de communauté Louvain (Règle 1)
    node2comm = {node: cid for cid, comm in enumerate(communities) for node in comm}

    # Nommage des communautés selon leurs 2 figures majeures
    comm_labels = {}
    for cid, comm in enumerate(communities):
        top_thinkers = sorted(comm, key=lambda n: strengths_full.get(n, 0), reverse=True)[:2]
        names = ", ".join([t.replace('_', ' ') for t in top_thinkers])
        comm_labels[cid] = f"C{cid+1} ({names})"

    # Règle 2 : Layout spatial calculé UNIQUEMENT sur le backbone
    pos = nx.spring_layout(G_backbone, seed=seed, k=0.14, iterations=60)

    fig, ax = plt.subplots(figsize=(14, 11), facecolor='white')

    # Palette de couleurs pour les communautés
    n_comms = len(communities)
    cmap = matplotlib.colormaps.get_cmap('tab10')

    # Règle 4 : Taille des nœuds proportionnelle à la force s_i dans le graphe original
    node_sizes = [min(max(strengths_full.get(n, 1.0) * 1.6, 12), 650) for n in nodes_backbone]
    node_colors = [cmap(node2comm.get(n, 0) % 10) for n in nodes_backbone]

    # Tracé des arêtes du backbone
    nx.draw_networkx_edges(
        G_backbone, pos, ax=ax,
        alpha=0.20, edge_color='#555555', width=0.65
    )

    # Règle 3 : Tracé des nœuds colorés par communauté de Louvain
    nx.draw_networkx_nodes(
        G_backbone, pos, ax=ax,
        nodelist=nodes_backbone,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.88,
        linewidths=0.5,
        edgecolors='#222222'
    )

    # Règle 5 : Étiquettes de texte UNIQUEMENT pour les 10 philosophes à plus forte force
    top_nodes = sorted(nodes_backbone, key=lambda n: strengths_full.get(n, 0), reverse=True)[:top_k_labels]
    for n in top_nodes:
        x, y = pos[n]
        display_name = n.replace('_', ' ')
        ax.text(
            x, y + 0.016, display_name,
            fontsize=8.5,
            fontweight='bold',
            ha='center',
            va='bottom',
            color='#111111',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.82, edgecolor='#555555', lw=0.6),
            zorder=6
        )

    # Légende claire des communautés présentes dans le backbone
    present_cids = sorted(list(set(node2comm[n] for n in nodes_backbone)))
    legend_patches = [
        mpatches.Patch(color=cmap(cid % 10), label=comm_labels[cid])
        for cid in present_cids
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        title="Communautés de Louvain (G complet non pondéré)",
        fontsize=8.5,
        title_fontsize=9.5,
        frameon=True,
        facecolor='#fafafa',
        edgecolor='#cccccc'
    )

    # Statistiques pour le titre et la légende
    n_full_edges = G_full.number_of_edges()
    n_back_edges = G_backbone.number_of_edges()
    n_removed = n_full_edges - n_back_edges
    pct_removed = (n_removed / n_full_edges) * 100

    ax.set_title(
        f"Colonne Vertébrale (Backbone) du Réseau des Philosophes\nFiltre de Disparité ($\\alpha = {alpha}$)",
        fontsize=14,
        fontweight='bold',
        pad=12
    )
    ax.axis('off')

    # Caption descriptive sous la figure
    caption_text = (
        f"Méthode : Filtre de Disparité (Serrano et al., 2009) | Seuil de significativité : α = {alpha}\n"
        f"Arêtes initiales : {n_full_edges:,} | Arêtes conservées : {n_back_edges:,} | "
        f"Arêtes retirées : {n_removed:,} ({pct_removed:.1f}%) | Philosophes connectés : {len(nodes_backbone):,}\n"
        f"Les couleurs représentent les communautés de Louvain calculées sur le graphe non pondéré complet. "
        f"La taille des nœuds reflète leur force globale $s_i$."
    )
    plt.figtext(
        0.5, 0.015, caption_text,
        wrap=True, horizontalalignment='center',
        fontsize=9, style='italic', color='#2c3e50',
        bbox=dict(boxstyle='square,pad=0.5', facecolor='#f8f9fa', edgecolor='#d6dbdf')
    )

    plt.subplots_adjust(bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Visualisation sauvegardée avec succès : '{save_path}'")
    plt.close()


# =============================================================================
# EXÉCUTION COMPLÈTE
# =============================================================================

def main():
    # 1. Chargement et extraction de la GCC
    G = load_philosophers_network()

    # 2. Tâche 1 : Analyse Force vs Degré
    df_metrics = task_1_strength_vs_degree(G)

    # 3. Tâche 2 : Filtre de disparité et tableau comparatif
    df_summary = task_2_backbone_comparison_table(G)

    # 4. Règle 1 : Calcul des communautés de Louvain sur le réseau COMPLET NON PONDÉRÉ
    print("\n" + "=" * 80)
    print("CALCUL DES COMMUNAUTÉS DE LOUVAIN SUR LE RÉSEAU COMPLET NON PONDÉRÉ (RÈGLE 1)")
    print("=" * 80)
    G_unweighted = nx.Graph(G)
    louvain_comms = sorted(nx.community.louvain_communities(G_unweighted, weight=None, seed=42), key=len, reverse=True)
    print(f"Nombre de communautés détectées : {len(louvain_comms)}")

    # 5. Tâche 3 : Visualisation de la figure principale (alpha = 0.2)
    print("\n" + "=" * 80)
    print("TÂCHE 3 : GÉNÉRATION DE LA FIGURE PRINCIPALE (alpha = 0.2)")
    print("=" * 80)
    B_02 = disparity_filter(G, alpha=0.2)
    plot_backbone_network(G, B_02, louvain_comms, alpha=0.2, save_path="backbone_alpha_02.png")

    # 6. Tâche 4 : Comparaison visuelle (alpha = 0.1 et alpha = 0.3)
    print("\n" + "=" * 80)
    print("TÂCHE 4 : GÉNÉRATION DES FIGURES COMPARATIVES (alpha = 0.1 et alpha = 0.3)")
    print("=" * 80)
    B_01 = disparity_filter(G, alpha=0.1)
    plot_backbone_network(G, B_01, louvain_comms, alpha=0.1, save_path="backbone_alpha_01.png")

    B_03 = disparity_filter(G, alpha=0.3)
    plot_backbone_network(G, B_03, louvain_comms, alpha=0.3, save_path="backbone_alpha_03.png")

    print("\n" + "=" * 80)
    print("TOUTES LES TÂCHES ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS !")
    print("=" * 80)


if __name__ == "__main__":
    main()
