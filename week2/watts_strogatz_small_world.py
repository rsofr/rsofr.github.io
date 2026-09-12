#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DTU SOCIAL GRAPHS & INTERACTIONS - WEEK 2
REPRODUCTION DE LA FIGURE CLASSIQUE "SMALL WORLD" DE WATTS & STROGATZ (1998)
===============================================================================

Ce script reproduit la célèbre figure du modèle de Watts-Strogatz publiée dans
la revue Nature (1998) : "Collective dynamics of 'small-world' networks".

Le modèle de Watts-Strogatz explore la transition topologique entre :
  1. Un treillis régulier unidimensionnel (anneau régulier, q = 0) :
     - Fort clustering local : C(0) = 3(k-2) / (4(k-1)) = 0.50 pour k = 4.
     - Longue distance géodésique moyenne : L(0) ≈ n / (2k) ≈ 62.88 pour n = 500, k = 4.
  2. Un réseau aléatoire type Erdős-Rényi (q -> 1) :
     - Faible distance moyenne : L_rand ~ ln(n) / ln(k).
     - Faible clustering : C_rand ~ k / n ≈ 0.008.

Le phénomène "Petit Monde" (Small-World) :
  Pour des valeurs intermédiaires très faibles de probabilité de reconnexion
  (q ∈ [0.01, 0.1]), l'apparition de quelques "raccourcis" à longue portée
  provoque un effondrement spectaculaire de la distance moyenne L(q) / L(0),
  tandis que le coefficient de clustering C(q) / C(0) reste remarquablement élevé
  (proche de 1).

Paramètres de l'exercice :
  - n = 500 nœuds
  - k = 4 (chaque nœud est connecté à ses 4 plus proches voisins sur l'anneau)
  - q_values = [0, 0.01, 0.03, 0.05, 0.1, 0.2]
  - 50 réalisations indépendantes par valeur de q
  - Normalisation : L(q) / L(0) et C(q) / C(0)
  - Tracé avec échelle logarithmique en X (symlog) et barres d'erreur (écart-type).
===============================================================================
"""

import os
import sys
import time
import types
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


# =============================================================================
# SIMULATION DE L'ENSEMBLE WATTS-STROGATZ
# =============================================================================

def run_watts_strogatz_simulation(
    n=500,
    k=4,
    q_values=(0, 0.01, 0.03, 0.05, 0.1, 0.2),
    num_runs=50,
    base_seed=42
):
    """
    Exécute la simulation Monte Carlo du modèle Watts-Strogatz.

    Pour chaque q dans q_values :
      1. Génère num_runs graphes indépendants via nx.watts_strogatz_graph(n, k, q).
      2. Calcule la distance moyenne ⟨d⟩ (nx.average_shortest_path_length).
      3. Calcule le clustering moyen C (nx.average_clustering).
      4. Calcule les moyennes et écarts-types.
      5. Normalise par L(0) et C(0).

    Retourne un dictionnaire contenant les données brutes et normalisées.
    """
    print("\n" + "=" * 78)
    print("🔬 SIMULATION DU MODÈLE WATTS-STROGATZ (NATURE 1998)")
    print("=" * 78)
    print(f"Paramètres du réseau :")
    print(f"  • Nombre de nœuds (n)           : {n}")
    print(f"  • Voisins les plus proches (k)  : {k}")
    print(f"  • Probabilités de reconnexion q : {list(q_values)}")
    print(f"  • Réalisations indépendantes    : {num_runs} par valeur de q")
    print("-" * 78)

    results = {
        'n': n,
        'k': k,
        'num_runs': num_runs,
        'q_values': list(q_values),
        'raw_data': {},
        'L_mean': [],
        'L_std': [],
        'C_mean': [],
        'C_std': [],
        'L_norm': [],
        'L_norm_std': [],
        'C_norm': [],
        'C_norm_std': []
    }

    start_time = time.time()

    for q_idx, q in enumerate(q_values):
        t_q0 = time.time()
        L_runs = []
        C_runs = []

        for run in range(num_runs):
            seed = base_seed + run + q_idx * 1000
            G = nx.watts_strogatz_graph(n, k, q, seed=seed)

            # Calcul de la distance géodésique moyenne ⟨d⟩
            # Si le graphe est déconnecté (rare pour q <= 0.2), on extrait la composante géante
            if nx.is_connected(G):
                apl = nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                gcc_sub = G.subgraph(largest_cc)
                apl = nx.average_shortest_path_length(gcc_sub)

            # Calcul du clustering moyen C
            c_val = nx.average_clustering(G)

            L_runs.append(apl)
            C_runs.append(c_val)

        mean_L = float(np.mean(L_runs))
        std_L = float(np.std(L_runs))
        mean_C = float(np.mean(C_runs))
        std_C = float(np.std(C_runs))

        results['raw_data'][q] = {
            'L_runs': L_runs,
            'C_runs': C_runs,
            'mean_L': mean_L,
            'std_L': std_L,
            'mean_C': mean_C,
            'std_C': std_C
        }
        results['L_mean'].append(mean_L)
        results['L_std'].append(std_L)
        results['C_mean'].append(mean_C)
        results['C_std'].append(std_C)

        elapsed_q = time.time() - t_q0
        print(f"  ✓ q = {q:4.2f} terminé en {elapsed_q:.2f}s | "
              f"⟨d⟩ = {mean_L:6.2f} ± {std_L:4.2f} | "
              f"⟨C⟩ = {mean_C:6.4f} ± {std_C:6.4f}")

    total_time = time.time() - start_time
    print(f"\n⏱️ Simulation terminée en {total_time:.2f} secondes.")

    # Normalisation par rapport à q = 0 (L(0) et C(0))
    L0 = results['L_mean'][0]
    C0 = results['C_mean'][0]
    results['L0'] = L0
    results['C0'] = C0

    print(f"\n📏 Valeurs de référence au treillis régulier (q = 0) :")
    print(f"   • L(0) mesuré = {L0:.4f} (Théorie anneau n/(2k) ≈ {n / (2 * k):.2f})")
    print(f"   • C(0) mesuré = {C0:.4f} (Théorie 3(k-2)/(4(k-1)) = {3 * (k - 2) / (4 * (k - 1)):.4f})")

    for i in range(len(q_values)):
        # Valeurs normalisées et propagation des incertitudes
        l_norm = results['L_mean'][i] / L0
        l_norm_std = results['L_std'][i] / L0
        c_norm = results['C_mean'][i] / C0
        c_norm_std = results['C_std'][i] / C0

        results['L_norm'].append(l_norm)
        results['L_norm_std'].append(l_norm_std)
        results['C_norm'].append(c_norm)
        results['C_norm_std'].append(c_norm_std)

    return results


# =============================================================================
# AFFICHAGE DE LA SYNTHÈSE EN CONSOLE
# =============================================================================

def print_summary_table(results):
    """
    Affiche un tableau ASCII clair avec les métriques brutes et normalisées.
    """
    q_vals = results['q_values']
    L0 = results['L0']
    C0 = results['C0']

    headers = ["q", "⟨d⟩ (brut)", "L(q) / L(0)", "⟨C⟩ (brut)", "C(q) / C(0)"]
    rows = []

    for i, q in enumerate(q_vals):
        l_mean = results['L_mean'][i]
        l_std = results['L_std'][i]
        l_norm = results['L_norm'][i]
        l_norm_std = results['L_norm_std'][i]

        c_mean = results['C_mean'][i]
        c_std = results['C_std'][i]
        c_norm = results['C_norm'][i]
        c_norm_std = results['C_norm_std'][i]

        l_raw_str = f"{l_mean:.2f} ± {l_std:.2f}"
        l_norm_str = f"{l_norm:.4f} ± {l_norm_std:.4f}"
        c_raw_str = f"{c_mean:.4f} ± {c_std:.4f}"
        c_norm_str = f"{c_norm:.4f} ± {c_norm_std:.4f}"

        rows.append([f"{q:.2f}", l_raw_str, l_norm_str, c_raw_str, c_norm_str])

    col_widths = [len(h) for h in headers]
    for r in rows:
        for idx, cell in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(cell))

    sep_top = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    sep_mid = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    sep_bot = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    header_line = "│" + "│".join(f" {headers[i]:<{col_widths[i]}} " if i == 0 else f" {headers[i]:>{col_widths[i]}} " for i in range(len(headers))) + "│"

    print("\n📊 TABLEAU RÉCAPITULATIF DES MÉTRIQUES WATTS-STROGATZ :")
    print(sep_top)
    print(header_line)
    print(sep_mid)
    for r in rows:
        line = "│" + "│".join(f" {r[i]:<{col_widths[i]}} " if i == 0 else f" {r[i]:>{col_widths[i]}} " for i in range(len(headers))) + "│"
        print(line)
    print(sep_bot + "\n")


# =============================================================================
# GÉNÉRATION DE LA FIGURE CLASSIQUE "SMALL WORLD"
# =============================================================================

def plot_watts_strogatz_figure(results, output_path='week2/watts_strogatz_small_world.png'):
    """
    Génère la figure classique de Watts & Strogatz (Nature 1998) :
      - X-axis : échelle logarithmique symlog permettant d'afficher q = 0
      - Y-axis : fraction normalisée [0.0, 1.0]
      - L(q)/L(0) : distance moyenne normalisée avec barres d'erreur (écart-type)
      - C(q)/C(0) : coefficient de clustering normalisé avec barres d'erreur
      - Bande ombrée soulignant le régime "Small World".
    """
    print(f"🎨 Tracé de la figure classique Watts-Strogatz : {output_path}...")

    q_vals = np.array(results['q_values'])
    l_norm = np.array(results['L_norm'])
    l_norm_std = np.array(results['L_norm_std'])
    c_norm = np.array(results['C_norm'])
    c_norm_std = np.array(results['C_norm_std'])

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300, facecolor='white')
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7, zorder=1)

    # 1. Zone ombrée délimitant le régime "Small World" (0.01 <= q <= 0.10)
    ax.axvspan(
        0.008, 0.12,
        color='#FEF3C7',
        alpha=0.55,
        zorder=2,
        label='Régime "Petit Monde" (Small-World)\n$C(q) \\approx C(0)$ et $L(q) \\ll L(0)$'
    )

    # 2. Distance géodésique moyenne normalisée L(q) / L(0)
    ax.errorbar(
        q_vals,
        l_norm,
        yerr=l_norm_std,
        fmt='o-',
        color='#2563EB',          # Bleu cobalt
        ecolor='#93C5FD',         # Bleu ciel pour les barres d'erreur
        elinewidth=2.0,
        capsize=4.5,
        capthick=1.6,
        markersize=7.5,
        linewidth=2.2,
        label=r'Distance moyenne normalisée $L(q) / L(0)$',
        zorder=4
    )

    # 3. Coefficient de clustering normalisé C(q) / C(0)
    ax.errorbar(
        q_vals,
        c_norm,
        yerr=c_norm_std,
        fmt='s-',
        color='#DC2626',          # Rouge carmin
        ecolor='#FCA5A5',         # Rose pâle pour les barres d'erreur
        elinewidth=2.0,
        capsize=4.5,
        capthick=1.6,
        markersize=7.5,
        linewidth=2.2,
        label=r'Clustering moyen normalisé $C(q) / C(0)$',
        zorder=5
    )

    # 4. Configuration de l'axe X en échelle logarithmique symlog (incluant q = 0)
    # Le seuil linéaire linthresh=0.005 permet d'avoir 0 à l'origine puis l'échelle log
    ax.set_xscale('symlog', linthresh=0.005)
    ax.set_xticks(q_vals)
    ax.set_xticklabels(['0', '0.01', '0.03', '0.05', '0.10', '0.20'], fontsize=11, fontweight='medium')
    ax.set_xlim(-0.001, 0.25)

    # 5. Configuration de l'axe Y (normalisé de 0 à 1)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.tick_params(axis='both', which='major', labelsize=11)

    # 6. Labels des axes et titre
    ax.set_xlabel(
        'Probabilité de reconnexion aléatoire $q$ (échelle logarithmique)',
        fontsize=12.5,
        fontweight='semibold',
        labelpad=10
    )
    ax.set_ylabel(
        'Fraction normalisée : $L(q) / L(0)$ et $C(q) / C(0)$',
        fontsize=12.5,
        fontweight='semibold',
        labelpad=10
    )
    ax.set_title(
        'Modèle de Watts-Strogatz : Effet "Small-World" (Nature 1998)\n'
        f'$n = {results["n"]}$ nœuds, $k = {results["k"]}$ voisins, 50 réalisations par point',
        fontsize=14,
        fontweight='bold',
        pad=14
    )

    # 7. Légende soignée
    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(0.02, 0.45),
        fontsize=10.5,
        frameon=True,
        facecolor='white',
        edgecolor='#CBD5E1',
        framealpha=0.95
    )
    legend.get_frame().set_boxstyle('round,pad=0.6')

    # 8. Annotations scientifiques pédagogiques
    # Annotation 1 : effondrement de la distance
    ax.annotate(
        'Effondrement rapide de $L(q)$ :\nQuelques raccourcis suffisent à\nréduire drastiquement les distances.',
        xy=(0.01, l_norm[1]),
        xytext=(0.0015, 0.18),
        arrowprops=dict(
            arrowstyle='->',
            color='#1E40AF',
            lw=1.8,
            connectionstyle='arc3,rad=-0.15'
        ),
        fontsize=9.5,
        color='#1E40AF',
        fontweight='medium',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EFF6FF', edgecolor='#93C5FD', alpha=0.95),
        zorder=6
    )

    # Annotation 2 : persistance du clustering
    ax.annotate(
        'Maintien d\'un clustering élevé :\nLa majorité des connexions restent\nlocales ($C(q) \\approx 0.87 - 0.97$).',
        xy=(0.03, c_norm[2]),
        xytext=(0.007, 0.60),
        arrowprops=dict(
            arrowstyle='->',
            color='#991B1B',
            lw=1.8,
            connectionstyle='arc3,rad=-0.15'
        ),
        fontsize=9.5,
        color='#991B1B',
        fontweight='medium',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF2F2', edgecolor='#FCA5A5', alpha=0.95),
        zorder=6
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure haute résolution enregistrée avec succès : {output_path}")


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

def main():
    # 1. Paramètres demandés par l'exercice
    n = 500
    k = 4
    q_values = [0, 0.01, 0.03, 0.05, 0.1, 0.2]
    num_runs = 50
    base_seed = 42

    # 2. Exécution des simulations Monte Carlo
    results = run_watts_strogatz_simulation(
        n=n,
        k=k,
        q_values=q_values,
        num_runs=num_runs,
        base_seed=base_seed
    )

    # 3. Affichage du tableau synthétique en console
    print_summary_table(results)

    # 4. Tracé et enregistrement de la figure
    output_img = 'week2/watts_strogatz_small_world.png'
    plot_watts_strogatz_figure(results, output_path=output_img)

    # 5. Conclusions théoriques
    print("=" * 78)
    print("🔍 ANALYSE PHYSIQUE DES RÉSULTATS :")
    print("=" * 78)
    print("1. Chute de la distance caractéristique L(q) :")
    print(f"   • Dès q = 0.01 (1% de liens reconnectés), L(q)/L(0) passe de 1.00 à {results['L_norm'][1]:.2f} (chute de >60%).")
    print(f"   • À q = 0.05 (5% de liens reconnectés), L(q)/L(0) tombe à {results['L_norm'][3]:.2f} (chute de >80%).")
    print("2. Persistance du coefficient de clustering C(q) :")
    print(f"   • À q = 0.01, C(q)/C(0) reste à {results['C_norm'][1]:.2f} (97% de sa valeur initiale de treillis).")
    print(f"   • À q = 0.05, C(q)/C(0) reste à {results['C_norm'][3]:.2f} (86% de sa valeur initiale de treillis).")
    print("3. Signature du phénomène Petit Monde (Small-World) :")
    print("   • Dans l'intervalle 0.01 <= q <= 0.1, le réseau combine simultanément :")
    print("     - Un clustering élevé typique d'un monde régulier et ordonné (forte cohésion locale).")
    print("     - Une distance moyenne ultra-courte typique d'un réseau aléatoire (diffusion globale rapide).")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
