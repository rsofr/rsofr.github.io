#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercice 2.6 : Analyse de la CCDF (Complementary Cumulative Distribution Function)
----------------------------------------------------------------------------------
Ce script :
1. Charge le réseau réel Marvel (nodes.tsv et edges.tsv) en préservant tous les 303 nœuds
   (y compris les 58 nœuds isolés) et calcule la distribution in-degree (degré entrant).
2. Calcule la distribution de probabilité classique P(k) avec le log-binning de la Semaine 1
   (largeur 1 pour k+1 in [1, 7], intervalles doublants au-delà, moyenne géométrique et normalisation).
3. Calcule la CCDF P(K >= k) du réseau Marvel sans binning (tri décroissant des degrés et rang / N).
4. Génère un réseau sans échelle de Barabási-Albert (N = 5000, m = 3).
5. Génère un réseau aléatoire d'Erdős-Rényi G(n, p) (N = 5000, même degré moyen <k> ~ 6).
6. Trace une figure à deux subplots sur axes log-log :
   - Subplot 1 : Distribution P(k) log-binnée de Marvel.
   - Subplot 2 : Superposition des 3 courbes CCDF (Marvel, Barabási-Albert, Erdős-Rényi).
7. Affiche dans le terminal les 20 premiers points (k, P(K >= k)) sous forme de texte brut
   prêt à copier-coller pour tester/piéger un LLM, accompagné d'une analyse théorique du piège.
"""

import os
import sys
import csv
import types

# Configuration cache matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

# Shims de compatibilité environnementale
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
except ImportError:
    d = types.ModuleType('dateutil')
    d.__version__ = '2.8.2'
    dp = types.ModuleType('dateutil.parser')
    dp.parse = lambda s, **kw: None
    d.parser = dp
    dr = types.ModuleType('dateutil.rrule')
    for c in ['rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU', 'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY', 'SECONDLY']:
        setattr(dr, c, 0)
    d.rrule = dr
    drm = types.ModuleType('dateutil.relativedelta')
    class relativedelta: pass
    drm.relativedelta = relativedelta
    d.relativedelta = drm
    dtz = types.ModuleType('dateutil.tz')
    dtz.tzlocal = lambda: None
    d.tz = dtz
    sys.modules['dateutil'] = d
    sys.modules['dateutil.parser'] = dp
    sys.modules['dateutil.rrule'] = dr
    sys.modules['dateutil.relativedelta'] = drm
    sys.modules['dateutil.tz'] = dtz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter


def load_marvel_network(nodes_path='nodes.tsv', edges_path='edges.tsv'):
    """
    Charge le réseau Marvel orienté en garantissant l'inclusion des 303 nœuds
    (y compris les isolats de degré entrant 0).
    """
    # Résolution des chemins relatifs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    
    if not os.path.exists(nodes_path):
        nodes_path = os.path.join(project_dir, 'nodes.tsv')
    if not os.path.exists(edges_path):
        edges_path = os.path.join(project_dir, 'edges.tsv')

    all_nodes = set()
    with open(nodes_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row and not row[0].startswith('#') and row[0].lower() not in ('node_id', 'id'):
                all_nodes.add(row[0].strip())

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)

    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row and not row[0].startswith('#') and row[0].lower() not in ('source', 'from'):
                if len(row) >= 2:
                    G.add_edge(row[0].strip(), row[1].strip())

    return G


def compute_log_binning(degrees_plus_1, total_nodes):
    """
    Applique le binning logarithmique exact de la Semaine 1 :
    - largeur 1 pour k+1 in [1, 7]
    - doublement au-delà de 8 : [8, 16), [16, 32), etc.
    - x = moyenne géométrique sqrt(low * high)
    - y = densité normalisée count / (total_nodes * width)
    """
    max_val = max(degrees_plus_1) + 2
    bin_edges = [1, 2, 3, 4, 5, 6, 7, 8]
    curr = 8
    while curr < max_val:
        curr *= 2
        bin_edges.append(curr)

    binned_x = []
    binned_y = []
    bin_info = []

    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i+1]
        width = high - low
        count = np.sum((degrees_plus_1 >= low) & (degrees_plus_1 < high))

        if count > 0:
            geom_mean = np.sqrt(low * high)
            density = count / (total_nodes * width)
            binned_x.append(geom_mean)
            binned_y.append(density)
            bin_info.append((f"[{low}, {high})", width, count, geom_mean, density))

    return np.array(binned_x), np.array(binned_y), bin_info


def compute_empirical_ccdf(degrees):
    """
    Calcule la CCDF empirique P(K >= k) pour un ensemble de degrés.
    Retourne :
    1. sorted_k, ccdf_rank : tri décroissant nœud par nœud (rang / N)
    2. unique_k, ccdf_unique : valeurs uniques de k > 0 et P(K >= k) exact
    """
    n = len(degrees)
    # Tri décroissant nœud par nœud
    sorted_d = np.sort(degrees)[::-1]
    ranks = np.arange(1, n + 1)
    ccdf_rank = ranks / n

    # Niveaux de degrés uniques k > 0
    unique_k = np.sort(np.unique(degrees))
    unique_k = unique_k[unique_k > 0]
    ccdf_unique = np.array([np.sum(degrees >= k) / n for k in unique_k])

    return (sorted_d, ccdf_rank), (unique_k, ccdf_unique)


def main():
    print("=" * 80)
    print("EXERCICE 2.6 : ANALYSE ET COMPARAISON DE LA CCDF DU RÉSEAU MARVEL")
    print("=" * 80)

    # 1. Chargement du réseau Marvel
    G_marvel = load_marvel_network()
    in_degrees = np.array([d for _, d in G_marvel.in_degree()])
    N_marvel = len(in_degrees)
    M_marvel = G_marvel.number_of_edges()
    mean_in_deg = np.mean(in_degrees)
    max_in_deg = np.max(in_degrees)
    num_isolates = np.sum(in_degrees == 0)

    print(f"\n[1] Réseau Réel Marvel (Orienté) :")
    print(f"    • Nœuds totaux (N)    : {N_marvel}")
    print(f"    • Arêtes (M)          : {M_marvel}")
    print(f"    • Degré entrant moyen : {mean_in_deg:.3f}")
    print(f"    • Degré entrant max   : {max_in_deg}")
    print(f"    • Isolats (k_in = 0)  : {num_isolates} ({num_isolates / N_marvel * 100:.1f}%)")

    # 2. Log-binning Marvel (Semaine 1)
    in_deg_p1 = in_degrees + 1
    binned_x, binned_y, bin_info = compute_log_binning(in_deg_p1, N_marvel)

    # Distribution brute
    raw_counter = Counter(in_deg_p1)
    raw_x = np.array(sorted(raw_counter.keys()))
    raw_y = np.array([raw_counter[k] / N_marvel for k in raw_x])

    # 3. Génération des réseaux synthétiques (N = 5000)
    N_synth = 5000
    m_ba = 3  # Chaque nouveau nœud s'attache à 3 nœuds existants
    print(f"\n[2] Génération du réseau Barabási-Albert (N = {N_synth}, m = {m_ba})...")
    G_ba = nx.barabasi_albert_graph(N_synth, m_ba, seed=42)
    ba_degrees = np.array([d for _, d in G_ba.degree()])
    mean_k_ba = np.mean(ba_degrees)
    max_k_ba = np.max(ba_degrees)
    print(f"    • Barabási-Albert : <k> = {mean_k_ba:.2f}, k_max = {max_k_ba}")

    # Erdős-Rényi avec la même moyenne de liens
    p_er = (2.0 * G_ba.number_of_edges()) / (N_synth * (N_synth - 1))
    print(f"\n[3] Génération du réseau Erdős-Rényi G(n, p) (N = {N_synth}, p = {p_er:.6f})...")
    G_er = nx.erdos_renyi_graph(N_synth, p_er, seed=42)
    er_degrees = np.array([d for _, d in G_er.degree()])
    mean_k_er = np.mean(er_degrees)
    max_k_er = np.max(er_degrees)
    print(f"    • Erdős-Rényi     : <k> = {mean_k_er:.2f}, k_max = {max_k_er}")

    # 4. Calcul des CCDFs
    (sorted_m, ccdf_rank_m), (u_km, ccdf_u_m) = compute_empirical_ccdf(in_degrees)
    _, (u_kba, ccdf_u_ba) = compute_empirical_ccdf(ba_degrees)
    _, (u_ker, ccdf_u_er) = compute_empirical_ccdf(er_degrees)

    # 5. Création de la figure comparative
    print("\n[4] Génération de la figure comparative (week2/exercise_2.6_ccdf.png)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300, facecolor='white')

    # Subplot 1 : Distribution P(k) binnée de Marvel
    ax1.set_facecolor('#F8FAFC')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7)
    
    # Bande verte marquant la zone de largeur = 1
    ax1.axvspan(1, 8, color='#10B981', alpha=0.12, label=r'Zone largeur $\Delta k = 1$ ($[1, 8)$)')
    
    # Points bruts
    ax1.loglog(raw_x, raw_y, 'o', color='#64748B', alpha=0.6, markersize=7, 
               label=r'Points bruts $P(k)$ ($k_{in}+1$)')
    
    # Courbe binnée
    ax1.loglog(binned_x, binned_y, 's-', color='#D97706', linewidth=2.5, markersize=8, 
               label=r'Courbe binnée $\frac{N_{bin}}{N \cdot \Delta k}$ à $\sqrt{b_i b_{i+1}}$')
    
    ax1.set_xlabel(r'Degré entrant $k_{in} + 1$ (échelle log)', fontsize=12, fontweight='medium')
    ax1.set_ylabel(r'Densité de probabilité $P(k)$ (échelle log)', fontsize=12, fontweight='medium')
    ax1.set_title(r'Subplot 1 : Distribution $P(k)$ du Réseau Marvel' + '\n' + 
                  rf'(Log-Binning de la Semaine 1, $N={N_marvel}$)', 
                  fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    # Subplot 2 : Superposition des 3 courbes CCDF
    ax2.set_facecolor('#F8FAFC')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7)

    # Marvel CCDF
    ax2.loglog(u_km, ccdf_u_m, 'o-', color='#DC2626', linewidth=2.2, markersize=6, 
               label=rf'Marvel Réel In-Degree ($N={N_marvel}, \langle k \rangle \approx {mean_in_deg:.2f}$)')
    
    # Barabási-Albert CCDF
    ax2.loglog(u_kba, ccdf_u_ba, 's-', color='#2563EB', linewidth=2.0, markersize=4.5, 
               label=rf'Barabási-Albert ($N={N_synth}, m={m_ba}, \langle k \rangle \approx {mean_k_ba:.1f}$)')
    
    # Erdős-Rényi CCDF
    ax2.loglog(u_ker, ccdf_u_er, '^-', color='#059669', linewidth=2.0, markersize=4.5, 
               label=rf'Aléatoire Erdős-Rényi $G(n,p)$ ($N={N_synth}, \langle k \rangle \approx {mean_k_er:.1f}$)')

    ax2.set_xlabel(r'Degré $k$ (échelle log)', fontsize=12, fontweight='medium')
    ax2.set_ylabel(r'CCDF $P(K \geq k)$ (échelle log)', fontsize=12, fontweight='medium')
    ax2.set_title(r'Subplot 2 : Comparaison des CCDF (Log-Log)' + '\n' + 
                  r'Marvel Réel vs Barabási-Albert vs Erdős-Rényi', 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exercise_2.6_ccdf.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Figure enregistrée avec succès : {output_path}")

    # 6. Extraction des 20 premiers points pour tester / piéger le LLM (Point 3)
    print("\n" + "=" * 80)
    print("EXTRACTION DES DONNÉES CCDF DU RÉSEAU MARVEL (POINT 3 : PIÉGER LE LLM)")
    print("=" * 80)
    
    print("\n--- FORMAT A : Tri décroissant nœud par nœud (Rangs 1 à 20 : k, P(K >= k)) ---")
    print(f"{'Rang':<6}{'Degré k':<10}{'P(K >= k)':<16}{'Fraction brute':<14}")
    for i in range(20):
        print(f"{i+1:<6}{sorted_m[i]:<10}{ccdf_rank_m[i]:<16.6f}{i+1}/{N_marvel}")

    # Distinct degree levels (descending)
    unique_desc_k = np.sort(np.unique(in_degrees[in_degrees > 0]))[::-1]
    unique_desc_ccdf = np.array([np.sum(in_degrees >= k) / N_marvel for k in unique_desc_k])

    print("\n--- FORMAT B : 20 premiers niveaux de degrés uniques (k décroissant) ---")
    print(f"{'Index':<6}{'Degré k':<10}{'P(K >= k)':<16}{'Nb nœuds >= k':<14}")
    for i in range(min(20, len(unique_desc_k))):
        n_ge = np.sum(in_degrees >= unique_desc_k[i])
        print(f"{i+1:<6}{unique_desc_k[i]:<10}{unique_desc_ccdf[i]:<16.6f}{n_ge}/{N_marvel}")

    print("\n--- FORMAT C : TEXTE BRUT PRÊT À COPIER-COLLER POUR L'INVITE DU LLM ---")
    print("Points CCDF Marvel (k, P(K >= k)) issus du tri décroissant nœud par nœud :")
    points_str = ", ".join([f"({sorted_m[i]}, {ccdf_rank_m[i]:.6f})" for i in range(20)])
    print(f"[{points_str}]\n")

    print("Listes séparées pour injection rapide dans un script Python :")
    print(f"k_values = {[int(sorted_m[i]) for i in range(20)]}")
    print(f"ccdf_values = {[round(float(ccdf_rank_m[i]), 6) for i in range(20)]}")

    print("\n" + "-" * 80)
    print("EXPLICATION DU PIÈGE THÉORIQUE POUR LE LLM :")
    print("-" * 80)
    print("1. ERREUR DE PENTE (CCDF vs PDF) :")
    print("   Si P(k) ~ k^(-gamma), alors la CCDF P(K >= k) ~ k^-(gamma - 1).")
    print("   Beaucoup de LLMs ajustent une droite log-log sur la CCDF, trouvent une pente")
    print("   alpha ~ -1.3 et concluent à tort que le réseau est sans échelle avec gamma = 1.3 !")
    print("   La vraie valeur de l'exposant de la loi de puissance est gamma = 1 + |alpha| ~ 2.3.")
    print("2. DISCRÉTISATION ET NŒUDS DE MÊME DEGRÉ :")
    print("   Aux rangs 11-13 (k=24) et 17-20 (k=21), la présence de plusieurs nœuds")
    print("   ayant le même degré crée des segments verticaux. Les régressions naïves")
    print("   sur données brutes (non filtrées) échouent ou sous-estiment la variance.")
    print("3. EFFET DE TAILLE FINIE (FINITE-SIZE CUTOFF) :")
    print("   Avec N = 303, le réseau Marvel présente une coupure exponentielle rapide.")
    print("   Un LLM hallucine souvent un comportement sans échelle 'pur' (scale-free)")
    print("   sans identifier la troncature due à la petite taille de l'échantillon.")
    print("=" * 80)


if __name__ == '__main__':
    main()
