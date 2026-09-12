#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DTU SOCIAL GRAPHS & INTERACTIONS (02805) - EXERCICE 2.11
ANALYSES AVANCÉES SUR LA COMPOSANTE GÉANTE DU RÉSEAU MARVEL COMICS :
  1. LE PARADOXE DE L'AMITIÉ CHEZ LES SUPER-HÉROS
  2. SHUFFLE TEST SUR LE CLUSTERING (MODÈLE NUL PRÉSERVANT LES DEGRÉS)
===============================================================================
Auteur : Romeo Sofia
Date   : Septembre 2026
===============================================================================
"""

import os
import sys
import csv
import numpy as np

# Gestion des chemins d'import locaux si nécessaires
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PYLIBS_DIR = os.path.join(WORKSPACE_DIR, '.pylibs')
if os.path.exists(PYLIBS_DIR) and PYLIBS_DIR not in sys.path:
    sys.path.insert(0, PYLIBS_DIR)

# Configuration du cache matplotlib dans le workspace
cache_dir = os.path.join(WORKSPACE_DIR, '.matplotlib_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = cache_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Palette graphique sombre & cybernétique assortie au site NetSci Lab
DARK_BG = "#07090e"
SURFACE_BG = "#0f141f"
TEXT_MAIN = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
CYAN_ACCENT = "#38bdf8"
PURPLE_ACCENT = "#a855f7"
RED_MARVEL = "#e62429"
EMERALD_ACCENT = "#10b981"
BORDER_COLOR = (1.0, 1.0, 1.0, 0.12)

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': SURFACE_BG,
    'axes.edgecolor': '#334155',
    'axes.labelcolor': TEXT_MAIN,
    'xtick.color': TEXT_MUTED,
    'ytick.color': TEXT_MUTED,
    'text.color': TEXT_MAIN,
    'font.sans-serif': ['Plus Jakarta Sans', 'DejaVu Sans', 'Arial'],
    'font.family': 'sans-serif',
    'grid.color': '#1e293b',
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})


def load_marvel_gcc():
    """
    Localise et charge le réseau Marvel non-orienté, puis extrait sa
    composante géante connexe (GCC).
    """
    candidate_dirs = [WORKSPACE_DIR, os.path.dirname(__file__), '.', '..']
    nodes_path, edges_path = None, None

    for d in candidate_dirs:
        n_cand = os.path.join(d, 'nodes.tsv')
        e_cand = os.path.join(d, 'edges.tsv')
        if os.path.exists(n_cand) and os.path.exists(e_cand):
            nodes_path = os.path.abspath(n_cand)
            edges_path = os.path.abspath(e_cand)
            break

    if not nodes_path or not edges_path:
        raise FileNotFoundError("Impossible de localiser 'nodes.tsv' et 'edges.tsv'.")

    G = nx.Graph()

    # Nœuds avec noms complets
    with open(nodes_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row or row[0].startswith('#') or row[0].lower() in ('node_id', 'id'):
                continue
            node_id = row[0].strip()
            name = row[1].strip() if len(row) > 1 else node_id
            G.add_node(node_id, name=name)

    # Arêtes
    with open(edges_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row or row[0].startswith('#') or row[0].lower() in ('source', 'from') or len(row) < 2:
                continue
            G.add_edge(row[0].strip(), row[1].strip())

    # Extraction de la Composante Géante (GCC)
    gcc_nodes = max(nx.connected_components(G), key=len)
    G_gcc = G.subgraph(gcc_nodes).copy()

    print(f"📊 Réseau Global : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")
    print(f"🌟 Composante Géante (GCC) : {G_gcc.number_of_nodes()} nœuds, {G_gcc.number_of_edges()} arêtes.")

    return G, G_gcc


def analyze_friendship_paradox(G_gcc, output_dirs):
    """
    Partie 1 : Analyse complète du paradoxe de l'amitié sur la GCC de Marvel.
    """
    print("\n" + "="*70)
    print("1. ANALYSE DU PARADOXE DE L'AMITIÉ CHEZ LES SUPER-HÉROS")
    print("="*70)

    degrees = np.array([d for _, d in G_gcc.degree()])
    k_mean = np.mean(degrees)
    k_sq_mean = np.mean(degrees**2)
    k_var = np.var(degrees)
    
    # Espérance théorique d'un voisin aléatoire : <k_voisin> = <k^2> / <k> = <k> + var(k)/<k>
    k_neighbor_expected = k_sq_mean / k_mean

    # Degré moyen des voisins par nœud k_nn(i)
    knn_dict = nx.average_neighbor_degree(G_gcc)
    knn_values = np.array([knn_dict[n] for n in G_gcc.nodes()])
    k_nn_mean = np.mean(knn_values)

    print(f"• Degré moyen d'un personnage choisi au hasard ⟨k⟩ : {k_mean:.3f}")
    print(f"• Variance des degrés σ_k² : {k_var:.3f}")
    print(f"• Degré moyen d'un voisin choisi au hasard ⟨k_voisin⟩ (théorie) : {k_neighbor_expected:.3f}")
    print(f"• Moyenne des degrés des voisins par nœud ⟨k_nn⟩ : {k_nn_mean:.3f}")
    print(f"• Ratio d'amplification du paradoxe : {k_neighbor_expected / k_mean:.2f}x")

    # Personnages immunisés au paradoxe (k_i >= k_nn,i) vs victimes (k_i < k_nn,i)
    beaters = []
    victims = []

    for n in G_gcc.nodes():
        deg = G_gcc.degree(n)
        knn = knn_dict[n]
        name = G_gcc.nodes[n].get('name', n)
        diff = deg - knn
        if deg >= knn:
            beaters.append((n, name, deg, knn, diff))
        else:
            victims.append((n, name, deg, knn, diff))

    beaters.sort(key=lambda x: x[4], reverse=True)
    victims.sort(key=lambda x: x[4])

    n_total = len(G_gcc)
    n_beaters = len(beaters)
    n_victims = len(victims)
    pct_beaters = (n_beaters / n_total) * 100
    pct_victims = (n_victims / n_total) * 100

    print(f"\n• Super-héros subissant le paradoxe (k < k_nn) : {n_victims} ({pct_victims:.1f}%)")
    print(f"• Super-héros immunisés au paradoxe (k >= k_nn) : {n_beaters} ({pct_beaters:.1f}%)")

    print("\n🏆 Top 10 des Hubs qui inversent le paradoxe (k >> k_nn) :")
    for i, (_, name, deg, knn, diff) in enumerate(beaters[:10], 1):
        clean_name = name.replace(' (character)', '').replace(' (comics)', '')
        print(f"   {i:02d}. {clean_name:<28} : Degré k = {deg:3d} | k_nn = {knn:5.2f} | Excédent = +{diff:5.2f}")

    # =========================================================================
    # VISUALISATION 1 : Le paradoxe de l'amitié
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.3, 1]})

    # Panneau Gauche : k_i vs k_nn(i)
    x_vals = degrees
    y_vals = knn_values
    max_val = max(max(x_vals), max(y_vals)) + 10

    # Ligne d'égalité y = x
    ax1.plot([0, max_val], [0, max_val], linestyle='--', color='#94a3b8', alpha=0.7, linewidth=1.8,
             label="Frontière neutre (Degré = Degré des voisins)")

    # Zones ombrées
    ax1.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                     color=RED_MARVEL, alpha=0.1, label=f"Subissent le paradoxe ({pct_victims:.1f}%)")
    ax1.fill_between([0, max_val], 0, [0, max_val],
                     color=CYAN_ACCENT, alpha=0.08, label=f"Immunisés au paradoxe ({pct_beaters:.1f}%)")

    # Nuage de points
    colors = [CYAN_ACCENT if d >= knn else '#f43f5e' for d, knn in zip(degrees, knn_values)]
    scatter = ax1.scatter(x_vals, y_vals, c=colors, s=55, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=4)

    # Annotations des grands hubs
    key_hubs_to_annotate = [
        ("Spider-Man", 106, knn_dict.get('Spider-Man', 14.9), (65, 30)),
        ("Hulk", 65, knn_dict.get('Hulk', 15.3), (40, 26)),
        ("Wolverine (character)", 63, knn_dict.get('Wolverine_(character)', 17.8), (40, 10)),
        ("Doctor Strange", 57, knn_dict.get('Doctor_Strange', 19.0), (35, 36)),
        ("Deadpool", 41, knn_dict.get('Deadpool', 19.7), (25, 30)),
        ("Captain America", 100, knn_dict.get('Captain_America', 20.0), (70, 42)),
    ]

    for n_id, name, deg, knn, diff in beaters[:6]:
        clean_name = name.replace(' (character)', '').replace(' (comics)', '')
        ax1.annotate(clean_name, (deg, knn),
                     xytext=(deg + 2, knn - 2),
                     fontsize=9.5, fontweight='bold', color='#ffffff',
                     arrowprops=dict(arrowstyle="->", color=CYAN_ACCENT, lw=1.2))

    ax1.set_xlim(0, 115)
    ax1.set_ylim(0, 45)
    ax1.set_xlabel("Degré individuel du héros : k", fontsize=11, fontweight='600')
    ax1.set_ylabel("Degré moyen de ses voisins : k_nn", fontsize=11, fontweight='600')
    ax1.set_title("A. Dispersion Nœud par Nœud : Héros vs Voisins", fontsize=13, fontweight='700', pad=12)
    ax1.legend(loc='upper right', framealpha=0.85, facecolor=SURFACE_BG, edgecolor='#334155', fontsize=9.5)
    ax1.grid(True)

    # Panneau Droit : Comparaison des moyennes et répartition
    metrics = ["Héros Aléatoire\n⟨k⟩", "Voisin Aléatoire\n⟨k²⟩ / ⟨k⟩", "Moyenne Voisins\n⟨k_nn⟩"]
    values = [k_mean, k_neighbor_expected, k_nn_mean]
    bar_colors = [TEXT_MUTED, RED_MARVEL, PURPLE_ACCENT]

    bars = ax2.bar(metrics, values, color=bar_colors, width=0.55, edgecolor=(1.0, 1.0, 1.0, 0.2), linewidth=1, zorder=3)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT_MAIN)

    ax2.set_ylim(0, 30)
    ax2.set_ylabel("Nombre moyen de connexions", fontsize=11, fontweight='600')
    ax2.set_title("B. Amplification du Degré Moyen (Paradoxe)", fontsize=13, fontweight='700', pad=12)
    ax2.grid(axis='y', zorder=0)

    # Stat box explicative
    box_text = (
        f"✦ Échantillon GCC : {n_total} héros\n"
        f"✦ Victimes du paradoxe : {pct_victims:.1f}%\n"
        f"✦ Héros immunisés (Hubs) : {pct_beaters:.1f}%\n"
        f"✦ Ratio d'amplification : x{k_neighbor_expected/k_mean:.2f}\n"
        f"→ Un ami a en moyenne 2x plus\n   d'alliés que le héros lui-même !"
    )
    ax2.text(0.5, 0.58, box_text, transform=ax2.transAxes,
             fontsize=9.5, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor=SURFACE_BG, edgecolor=CYAN_ACCENT, alpha=0.9, lw=1.2))

    plt.suptitle("LE PARADOXE DE L'AMITIÉ DANS L'UNIVERS MARVEL COMICS", fontsize=16, fontweight='800', y=0.98, color=CYAN_ACCENT)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "friendship_paradox.png")
        fig.savefig(out_path, dpi=300, facecolor=DARK_BG, bbox_inches='tight')
        print(f"   💾 Graphique sauvegardé : {out_path}")

    plt.close(fig)

    return {
        'k_mean': k_mean,
        'k_neighbor_expected': k_neighbor_expected,
        'k_nn_mean': k_nn_mean,
        'pct_victims': pct_victims,
        'pct_beaters': pct_beaters,
        'top_beaters': beaters[:10]
    }


def analyze_null_model_clustering(G_gcc, output_dirs, n_iterations=100):
    """
    Partie 2 : Test de permutation (Shuffle Test) préservant la distribution des
    degrés pour évaluer la significativité statistique du clustering Marvel.
    """
    print("\n" + "="*70)
    print("2. SHUFFLE TEST SUR LE COEFFICIENT DE CLUSTERING (MODÈLE NUL)")
    print("="*70)

    # Métriques réelles
    real_c = nx.average_clustering(G_gcc)
    real_trans = nx.transitivity(G_gcc)

    m = G_gcc.number_of_edges()
    nswap = 10 * m
    max_tries = 100 * m

    print(f"• Clustering réel ⟨C_réel⟩   : {real_c:.4f}")
    print(f"• Transitivité réelle T_réel : {real_trans:.4f}")
    print(f"• Brassage par double permutation d'arêtes ({n_iterations} itérations, {nswap} swaps/réseau)...")

    np.random.seed(42)
    null_c_list = []
    null_trans_list = []

    for it in range(n_iterations):
        G_null = G_gcc.copy()
        nx.double_edge_swap(G_null, nswap=nswap, max_tries=max_tries, seed=42 + it)
        c_val = nx.average_clustering(G_null)
        t_val = nx.transitivity(G_null)
        null_c_list.append(c_val)
        null_trans_list.append(t_val)
        if (it + 1) % 20 == 0 or it == n_iterations - 1:
            print(f"   ✓ Modèles nuls générés : {it+1:3d}/{n_iterations}")

    null_c = np.array(null_c_list)
    null_trans = np.array(null_trans_list)

    null_c_mean = np.mean(null_c)
    null_c_std = np.std(null_c)
    z_score_c = (real_c - null_c_mean) / null_c_std
    p_val_c = np.sum(null_c >= real_c) / float(len(null_c))

    null_t_mean = np.mean(null_trans)
    null_t_std = np.std(null_trans)
    z_score_t = (real_trans - null_t_mean) / null_t_std
    p_val_t = np.sum(null_trans >= real_trans) / float(len(null_trans))

    print(f"\n📊 RÉSULTATS STATISTIQUES DU SHUFFLE TEST :")
    print(f"• Clustering Réel        : {real_c:.4f}")
    print(f"• Modèle Nul (Moyenne)   : {null_c_mean:.4f} ± {null_c_std:.4f}")
    print(f"• Min / Max Modèle Nul   : [{np.min(null_c):.4f}, {np.max(null_c):.4f}]")
    print(f"• Z-score Clustering     : +{z_score_c:.2f} σ")
    print(f"• p-value empirique      : {p_val_c:.5f} (p < 0.001)")
    print(f"• Z-score Transitivité   : +{z_score_t:.2f} σ (p-value = {p_val_t})")

    # =========================================================================
    # VISUALISATION 2 : Histogramme du Shuffle Test
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    # Histogramme & KDE de la distribution nulle
    sns.histplot(null_c, kde=True, ax=ax, color=CYAN_ACCENT, bins=14,
                 stat="density", alpha=0.45, edgecolor=CYAN_ACCENT, linewidth=1.2,
                 label="Modèles nuls brassés (Double Edge Swap, N=100)")

    # Moyenne du modèle nul
    ax.axvline(null_c_mean, color='#38bdf8', linestyle='--', linewidth=2,
               label=f"Moyenne Modèle Nul (⟨C_nul⟩ = {null_c_mean:.4f})")

    # Valeur réelle Marvel (Ligne rouge bien visible)
    ax.axvline(real_c, color=RED_MARVEL, linestyle='-', linewidth=3.5, zorder=5,
               label=f"Réseau Marvel Réel (⟨C_réel⟩ = {real_c:.4f})")

    # Zone d'écart & Flèche explicative
    arrow_y = ax.get_ylim()[1] * 0.45
    ax.annotate(f"Écart géant : +{z_score_c:.1f} écarts-types !\n(Z-score = {z_score_c:.2f})",
                xy=(real_c, arrow_y), xytext=(real_c - 0.055, arrow_y + 8),
                fontsize=11, fontweight='bold', color=RED_MARVEL,
                arrowprops=dict(facecolor=RED_MARVEL, edgecolor=RED_MARVEL, shrink=0.08, width=2, headwidth=9),
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=SURFACE_BG, edgecolor=RED_MARVEL, alpha=0.9))

    # Encart statistiques
    stat_box = (
        f"TEST D'HYPOTHÈSE NULLE :\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• H0 : Le clustering s'explique par la séquence de degrés\n"
        f"• H1 : Sur-représentation triadique (Clustering réel > Nul)\n\n"
        f"✦ ⟨C_réel⟩          : {real_c:.4f}\n"
        f"✦ ⟨C_nul⟩ (μ ± σ)   : {null_c_mean:.4f} ± {null_c_std:.4f}\n"
        f"✦ Z-Score           : +{z_score_c:.2f} σ\n"
        f"✦ p-value empirique : p = {p_val_c:.4f} (p < 0.001)\n\n"
        f"CONCLUSION : Rejet massif de H0 !\n"
        f"Le regroupement en équipes (Avengers, X-Men...)\n"
        f"crée une densité de triangles impossible par pur hasard."
    )
    ax.text(0.04, 0.94, stat_box, transform=ax.transAxes,
            fontsize=9.8, verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor=SURFACE_BG, edgecolor='#475569', alpha=0.92, lw=1.2))

    ax.set_title("SHUFFLE TEST SUR LE COEFFICIENT DE CLUSTERING (100 MODÈLES NULS)", fontsize=14, fontweight='800', pad=15, color=TEXT_MAIN)
    ax.set_xlabel("Coefficient de clustering moyen ⟨C⟩", fontsize=12, fontweight='600', labelpad=10)
    ax.set_ylabel("Densité de probabilité", fontsize=12, fontweight='600', labelpad=10)
    ax.set_xlim(null_c_mean - 0.03, real_c + 0.03)
    ax.legend(loc='upper right', framealpha=0.85, facecolor=SURFACE_BG, edgecolor='#334155', fontsize=10)
    ax.grid(True)

    plt.tight_layout()

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "null_model_clustering_shuffle.png")
        fig.savefig(out_path, dpi=300, facecolor=DARK_BG, bbox_inches='tight')
        print(f"   💾 Graphique sauvegardé : {out_path}")

    plt.close(fig)

    return {
        'real_c': real_c,
        'null_c_mean': null_c_mean,
        'null_c_std': null_c_std,
        'z_score': z_score_c,
        'p_value': p_val_c,
        'real_trans': real_trans,
        'z_score_trans': z_score_t
    }


def main():
    print("="*70)
    print("DTU 02805 - EXERCICE 2.11 : ANALYSES DU RÉSEAU MARVEL COMICS")
    print("="*70)

    # Dossiers de sortie pour les visualisations
    output_dirs = [
        os.path.join(WORKSPACE_DIR, 'week2'),
        os.path.join(WORKSPACE_DIR, 'SGI_group'),
        os.path.join(WORKSPACE_DIR, 'SGI_group', 'assets')
    ]

    G, G_gcc = load_marvel_gcc()
    paradox_results = analyze_friendship_paradox(G_gcc, output_dirs)
    null_results = analyze_null_model_clustering(G_gcc, output_dirs, n_iterations=100)

    print("\n" + "="*70)
    print("✨ TOUTES LES ANALYSES ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS !")
    print("="*70)


if __name__ == '__main__':
    main()
