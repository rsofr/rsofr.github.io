#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DTU SOCIAL GRAPHS & INTERACTIONS - WEEK 2
REPRODUCTION DE LA TRANSITION DE PHASE DANS UN RÉSEAU ALÉATOIRE G(n, p)
===============================================================================

Ce script modélise et visualise l'émergence de la composante géante (Giant
Connected Component, GCC) dans les réseaux aléatoires d'Erdős-Rényi G(n, p).

Phénomène de percolation :
  - Régime sous-critique (⟨k⟩ < 1) : Le graphe est fragmenté en petits arbres
    et composantes isolées de taille O(ln n). La fraction S tend vers 0.
  - Seuil critique (⟨k⟩ = 1) : Transition de phase continue (percolation).
  - Régime sur-critique (⟨k⟩ > 1) : Une composante géante unique émerge et
    englobe une fraction macroscopique S des nœuds telle que S = 1 - e^(-⟨k⟩*S).

Paramètres de simulation :
  - Nombre de nœuds : n = 1000
  - Plage de degré moyen : ⟨k⟩ ∈ [0.2, 4.0] (30 pas réguliers)
  - 20 réalisations indépendantes de G(n, p) pour chaque valeur de ⟨k⟩
  - Tracé avec barres d'erreur (écart-type) et comparaison avec la courbe théorique.
===============================================================================
"""

import os
import sys
import math
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
# RÉSOLUTION DE L'ÉQUATION D'AUTO-COHÉRENCE THÉORIQUE
# =============================================================================

def solve_theoretical_giant_component(k_avg, max_iter=200, tol=1e-8):
    """
    Résout numériquement l'équation théorique d'auto-cohérence pour la fraction S
    de la composante géante dans un réseau aléatoire G(n, p) (Barabási Ch. 3.6) :
        S = 1 - exp(-⟨k⟩ * S)

    Pour ⟨k⟩ <= 1 : S = 0 (pas de composante géante).
    Pour ⟨k⟩ > 1 : il existe une unique solution non nulle S in (0, 1).
    """
    if k_avg <= 1.0:
        return 0.0

    # Itération de point fixe à partir de S = 0.5
    s = 0.5
    for _ in range(max_iter):
        s_next = 1.0 - math.exp(-k_avg * s)
        if abs(s_next - s) < tol:
            return s_next
        s = s_next
    return s


# =============================================================================
# SIMULATION DE LA TRANSITION DE PHASE
# =============================================================================

def simulate_phase_transition(n=1000, k_min=0.2, k_max=4.0, num_steps=30, n_realizations=20, seed=42):
    """
    Simule l'émergence de la composante géante pour différentes valeurs de ⟨k⟩ :
      - Calcule p = ⟨k⟩ / (n - 1)
      - Génère n_realizations graphes G(n, p) indépendants
      - Extrait la fraction S = taille(plus grande composante) / n
      - Calcule la moyenne et l'écart-type de S

    Retourne :
      - k_values : tableau des valeurs de ⟨k⟩ testées
      - s_means  : fractions moyennes observées
      - s_stds   : écarts-types observés
      - s_theo   : fractions théoriques attendues
    """
    if seed is not None:
        np.random.seed(seed)

    k_values = np.linspace(k_min, k_max, num_steps)
    s_means = []
    s_stds = []
    s_theo = []

    print("\n" + "=" * 78)
    print(f"🔄 SIMULATION DE LA TRANSITION DE PHASE DANS G(n={n}, p)")
    print("=" * 78)
    print(f"  • Taille du réseau (n)        : {n}")
    print(f"  • Plage de degré moyen ⟨k⟩    : [{k_min:.1f}, {k_max:.1f}] ({num_steps} pas)")
    print(f"  • Réalisations par valeur     : {n_realizations} graphes indépendants")
    print(f"  • Nombre total de graphes     : {num_steps * n_realizations}")
    print("-" * 78)
    print(f"{'⟨k⟩':>6} | {'p = ⟨k⟩/(n-1)':>15} | {'S moyen (obs)':>15} | {'Écart-type (σ)':>16} | {'S théorique':>13}")
    print("-" * 78)

    for step_idx, k_val in enumerate(k_values, 1):
        p = k_val / float(n - 1)
        fractions = []

        for _ in range(n_realizations):
            # Génération d'un graphe aléatoire G(n, p)
            # En omettant le seed local dans erdos_renyi_graph, on utilise le générateur aléatoire global
            G = nx.erdos_renyi_graph(n, p)

            # Identification de la plus grande composante connexe
            # len(max(nx.connected_components(G), key=len))
            components = list(nx.connected_components(G))
            gcc_size = len(max(components, key=len)) if components else 0
            s_fraction = gcc_size / float(n)
            fractions.append(s_fraction)

        mean_s = float(np.mean(fractions))
        std_s = float(np.std(fractions))
        theo_s = solve_theoretical_giant_component(k_val)

        s_means.append(mean_s)
        s_stds.append(std_s)
        s_theo.append(theo_s)

        # Affichage régulier (tous les 3 pas ou premier/dernier)
        if step_idx == 1 or step_idx % 3 == 0 or step_idx == num_steps:
            print(f"{k_val:6.2f} | {p:15.6f} | {mean_s:15.4f} | {std_s:16.4f} | {theo_s:13.4f}")

    print("-" * 78)
    print("✅ Simulation terminée avec succès.")

    return {
        'k_values': k_values,
        's_means': np.array(s_means),
        's_stds': np.array(s_stds),
        's_theo': np.array(s_theo),
        'n': n,
        'n_realizations': n_realizations
    }


# =============================================================================
# TRACÉ DU GRAPHIQUE DE TRANSITION DE PHASE
# =============================================================================

def plot_phase_transition(results, output_img='week2/phase_transition_giant_component.png'):
    """
    Génère le graphique de transition de phase :
      - Courbe de S en fonction de ⟨k⟩ avec barres d'erreur (plt.errorbar)
      - Ligne verticale rouge en pointillés à ⟨k⟩ = 1 (seuil critique)
      - Superposition de la solution théorique S = 1 - exp(-⟨k⟩*S)
      - Annotations des régimes sous-critique et sur-critique
      - Axes clairs, grille et titre explicite.
    """
    print(f"\n🎨 Génération du graphique de transition de phase : {output_img}...")

    k_vals = results['k_values']
    s_means = results['s_means']
    s_stds = results['s_stds']
    s_theo = results['s_theo']
    n = results['n']
    n_real = results['n_realizations']

    # Configuration de la figure haute résolution
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300, facecolor='white')
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7, zorder=1)

    # 1. Tracé empirique avec barres d'erreur (moyenne ± écart-type sur les 20 réalisations)
    ax.errorbar(
        k_vals,
        s_means,
        yerr=s_stds,
        fmt='o-',
        color='#2563EB',          # Bleu cobalt
        ecolor='#93C5FD',         # Bleu ciel pour les barres d'erreur
        elinewidth=1.8,
        capsize=3.5,
        capthick=1.4,
        markersize=5.5,
        linewidth=1.8,
        label=f'Simulation empirique ($n={n}$, {n_real} réalisations)\nMoyenne $S = \\frac{{\\mathrm{{taille}}}}{{n}} \\pm \\sigma$',
        zorder=4
    )

    # 2. Superposition de la courbe théorique d'auto-cohérence
    # Tracé d'une courbe continue fine pour une résolution optimale
    k_dense = np.linspace(k_vals[0], k_vals[-1], 300)
    s_dense_theo = [solve_theoretical_giant_component(k) for k in k_dense]
    ax.plot(
        k_dense,
        s_dense_theo,
        color='#059669',          # Vert émeraude
        linestyle='-',
        linewidth=2.4,
        label='Théorie analytique ($S = 1 - e^{-\\langle k \\rangle S}$)',
        zorder=3
    )

    # 3. Ligne verticale rouge pointillée au seuil critique ⟨k⟩ = 1
    ax.axvline(
        x=1.0,
        color='#DC2626',          # Rouge vif
        linestyle=':',
        linewidth=2.2,
        label='Seuil critique $\\langle k \\rangle_c = 1.0$ (Percolation)',
        zorder=5
    )

    # 4. Coloration subtile des deux régimes (sous-critique et sur-critique)
    ax.axvspan(0.15, 1.0, color='#FEE2E2', alpha=0.35, zorder=1, label='Régime sous-critique (Pas de composante géante)')
    ax.axvspan(1.0, 4.05, color='#DCFCE7', alpha=0.35, zorder=1, label='Régime sur-critique (Composante géante $S > 0$)')

    # 5. Titre et labels soignés
    ax.set_title(
        f"Transition de phase dans les réseaux aléatoires d'Erdős-Rényi $G(n={n}, p)$\n"
        f"Émergence de la composante géante en fonction du degré moyen $\\langle k \\rangle$",
        fontsize=13,
        fontweight='bold',
        pad=16,
        color='#0F172A'
    )
    ax.set_xlabel("Degré moyen $\\langle k \\rangle = p \\cdot (n - 1)$", fontsize=11.5, fontweight='medium', color='#0F172A')
    ax.set_ylabel("Fraction relative de la composante géante $S = \\frac{N_{GCC}}{n}$", fontsize=11.5, fontweight='medium', color='#0F172A')

    # Limites d'axes et graduations
    ax.set_xlim(0.15, 4.05)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(np.arange(0.5, 4.5, 0.5))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    pct_val = s_dense_theo[-1] * 100.0
    notes_text = (
        "Propriétés clés :\n"
        " • $\\langle k \\rangle < 1$ : $S \\approx 0$, petites composantes $O(\\ln n)$\n"
        " • $\\langle k \\rangle = 1$ : seuil critique (percolation)\n"
        " • $\\langle k \\rangle > 1$ : composante géante unique $O(n)$\n"
        f" • À $\\langle k \\rangle = 4.0$ : $S \\approx {s_dense_theo[-1]:.3f}$ (env. {pct_val:.1f}% du réseau)"
    )
    ax.text(
        0.97, 0.28,
        notes_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='white', edgecolor='#CBD5E1', alpha=0.95),
        zorder=6
    )

    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5)
    plt.tight_layout()

    # Création du dossier cible si besoin et sauvegarde
    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
    fig.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"  💾 Graphique exporté avec succès : {output_img}")


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def main():
    """Point d'entrée pour la simulation et le tracé de la transition de phase."""
    print("🚀 Lancement de l'analyse de la transition de phase...")

    # Paramètres conformes aux consignes
    n = 1000
    k_min = 0.2
    k_max = 4.0
    num_steps = 30
    n_realizations = 20
    output_image = 'week2/phase_transition_giant_component.png'

    # Simulation
    results = simulate_phase_transition(
        n=n,
        k_min=k_min,
        k_max=k_max,
        num_steps=num_steps,
        n_realizations=n_realizations,
        seed=42
    )

    # Tracé et export
    plot_phase_transition(results, output_img=output_image)

    print("\n" + "=" * 78)
    print("🎯 ANALYSE DE LA TRANSITION DE PHASE ACHEVÉE AVEC SUCCÈS")
    print(f"Fichier image produit : {output_image}")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
