#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DTU SOCIAL GRAPHS & INTERACTIONS - WEEK 2
MODÉLISATION ET ANALYSE DES DISTRIBUTIONS DE DEGRÉS DANS LES RÉSEAUX G(n, p)
===============================================================================

Ce script modélise et analyse la distribution de degrés de réseaux aléatoires
d'Erdős-Rényi G(n, p) selon deux régimes :
  - Partie 1 : Réseau clairsemé (⟨k⟩ = 10, p = 10/4999, n = 5000)
               Comparaison avec la loi théorique de Poisson(λ = 10)
               Calcul et comparaison du degré max observé vs théorique (k_max).
  - Partie 2 : Réseau plus dense (⟨k⟩ = 100, p = 100/4999, n = 5000)
               Comparaison avec l'approximation Normale N(μ = 100, σ² = 100).

Attention particulière portée aux nœuds isolés :
  - Les nœuds sans liens (k = 0) sont rigoureusement préservés et comptabilisés
    grâce à nx.erdos_renyi_graph() et l'extraction exhaustive sur tous les n nœuds.
  - Calculs statistiques basés sur NumPy et le module standard math pour une
    compatibilité totale sans dépendance externe obligatoire.
===============================================================================
"""

import os
import sys
import math
import types
import numpy as np

# Configuration du cache Matplotlib dans un dossier accessible
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')

# Compatibilité et shims légers pour les environnements minimaux
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
# Utiliser Agg pour l'export sans interface graphique requise
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# FONCTIONS MATHÉMATIQUES & THÉORIQUES (NumPy + math)
# =============================================================================

def compute_poisson_pmf(k_values, lam):
    """
    Calcule la fonction de masse de probabilité (PMF) d'une loi de Poisson(λ)
    pour un tableau d'entiers k >= 0 :
        P(X = k) = (λ^k * e^(-λ)) / k!
    Utilise le log-espace avec math.lgamma pour éviter tout risque de débordement.
    """
    pmf = []
    log_lam = math.log(lam)
    for k in k_values:
        if k < 0:
            pmf.append(0.0)
        else:
            # log(P) = k * log(λ) - λ - log(k!)
            log_p = k * log_lam - lam - math.lgamma(k + 1)
            pmf.append(math.exp(log_p))
    return np.array(pmf)


def compute_normal_pdf(x_values, mu, sigma):
    """
    Calcule la densité de probabilité (PDF) d'une loi normale N(μ, σ²) :
        f(x) = (1 / (σ * sqrt(2π))) * exp(-0.5 * ((x - μ) / σ)²)
    """
    denom = sigma * math.sqrt(2.0 * math.pi)
    z = (np.asarray(x_values) - mu) / sigma
    return (1.0 / denom) * np.exp(-0.5 * (z ** 2))


def calculate_expected_kmax_poisson(n, lam, search_max=120):
    """
    Calcule le degré maximum théorique attendu k_max dans un réseau aléatoire
    selon la théorie de la distribution de Poisson (Barabási, Network Science Ch. 3).

    Le cutoff naturel k_max est défini par la condition où l'on s'attend à trouver
    au plus un nœud de degré supérieur ou égal à k_max dans le réseau :
        P(X >= k_max) = sum_{k = k_max}^{infinity} p_k ≈ 1 / n
        soit  n * P(X >= k_max) ≈ 1

    Retourne:
      - k_closest: l'entier k pour lequel n * P(X >= k) est le plus proche de 1.
      - k_floor: le plus grand entier k tel que n * P(X >= k) >= 1.
      - k_ceil: le plus petit entier k tel que n * P(X >= k) <= 1.
      - details: dictionnaire contenant les probabilités cumulées correspondantes.
    """
    target = 1.0 / n
    cdf = 0.0
    tail_probs = {}

    for k in range(0, search_max):
        # P(X >= k) = 1 - P(X <= k - 1) = 1 - (cdf avant k)
        tail = max(0.0, 1.0 - cdf)
        tail_probs[k] = tail
        pmf_k = math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
        cdf += pmf_k

    # Recherche du k minimisant l'écart absolu |P(X >= k) - 1/n|
    k_closest = min(tail_probs.keys(), key=lambda k: abs(tail_probs[k] - target))

    # Seuils inférieur et supérieur
    candidates_ge_1 = [k for k, p in tail_probs.items() if p * n >= 1.0]
    candidates_le_1 = [k for k, p in tail_probs.items() if p * n <= 1.0]

    k_floor = max(candidates_ge_1) if candidates_ge_1 else k_closest
    k_ceil = min(candidates_le_1) if candidates_le_1 else k_closest

    return {
        'k_closest': k_closest,
        'k_floor': k_floor,
        'k_ceil': k_ceil,
        'tail_closest': tail_probs[k_closest],
        'expected_nodes_closest': n * tail_probs[k_closest],
        'tail_floor': tail_probs[k_floor],
        'expected_nodes_floor': n * tail_probs[k_floor],
        'tail_ceil': tail_probs[k_ceil],
        'expected_nodes_ceil': n * tail_probs[k_ceil],
    }


# =============================================================================
# EXTRACTION DE LA DISTRIBUTION DE DEGRÉS (AVEC GESTION DES NŒUDS ISOLÉS)
# =============================================================================

def extract_degree_distribution(G, n):
    """
    Calcule rigoureusement la distribution de degrés pour TOUS les nœuds du graphe G,
    en garantissant que les nœuds isolés (degré 0) sont bien inclus.

    Paramètres :
      - G : graphe NetworkX (nx.Graph)
      - n : nombre attendu de nœuds

    Retourne :
      - k_vals : tableau NumPy des degrés de 0 à k_max_obs
      - pk_obs : tableau NumPy des fractions de nœuds pour chaque degré k
      - counts : comptage absolu des nœuds par degré
      - isolated_count : nombre de nœuds isolés (k = 0)
      - degrees : liste de tous les degrés extraits
    """
    # Extraction exhaustive des degrés pour l'ensemble des n nœuds
    degrees = [d for _, d in G.degree()]

    # Vérification d'intégrité : aucun nœud ne doit manquer
    if len(degrees) != n:
        raise ValueError(
            f"Erreur d'intégrité : le nombre de degrés ({len(degrees)}) "
            f"ne correspond pas au nombre de nœuds attendu ({n})."
        )

    k_max_obs = max(degrees) if len(degrees) > 0 else 0
    k_vals = np.arange(0, k_max_obs + 1)

    # np.bincount compte explicitement à partir de k = 0 jusqu'à k_max_obs
    counts = np.bincount(degrees, minlength=k_max_obs + 1)
    pk_obs = counts / float(n)

    isolated_count = counts[0]

    return {
        'k_vals': k_vals,
        'pk_obs': pk_obs,
        'counts': counts,
        'isolated_count': isolated_count,
        'degrees': degrees,
        'k_max_obs': k_max_obs,
        'k_min_obs': min(degrees),
        'avg_degree_obs': float(np.mean(degrees)),
        'var_degree_obs': float(np.var(degrees)),
    }


# =============================================================================
# PARTIE 1 : RÉSEAU CLAIRSEMÉ (⟨k⟩ = 10) ET COMPARAISON POISSON
# =============================================================================

def run_part1(n=5000, target_k=10.0, seed=42, output_img='week2/part1_poisson_degree_dist.png'):
    """
    Exécute la Partie 1 :
      - Génère un graphe aléatoire G(n, p) avec n=5000 et p=10/4999 (⟨k⟩ = 10).
      - Extrait la fraction de nœuds pour chaque k de 0 à k_max (inclus isolats).
      - Trace un diagramme en barres avec superposition de la loi théorique de Poisson(λ=10).
      - Affiche le degré maximum observé vs le degré max prédit par Poisson.
    """
    print("\n" + "=" * 78)
    print("📌 PARTIE 1 : Réseau Aléatoire G(n, p) avec ⟨k⟩ = 10 (Distribution de Poisson)")
    print("=" * 78)

    p = target_k / float(n - 1)
    print(f"  • Nombre de nœuds (n)         : {n}")
    print(f"  • Probabilité de lien (p)     : {p:.8f} (soit {target_k}/{n-1})")
    print(f"  • Degré moyen théorique ⟨k⟩   : {p * (n - 1):.4f}")
    print(f"  • Graine aléatoire (seed)     : {seed}")

    # 1. Génération du graphe d'Erdős-Rényi
    # nx.erdos_renyi_graph garantit que tous les nœuds de 0 à n-1 existent
    G1 = nx.erdos_renyi_graph(n, p, seed=seed)
    num_edges = G1.number_of_edges()
    print(f"  • Nombre d'arêtes générées    : {num_edges}")

    # 2. Distribution empirique des degrés (incluant k = 0)
    dist = extract_degree_distribution(G1, n)
    k_vals = dist['k_vals']
    pk_obs = dist['pk_obs']
    k_max_obs = dist['k_max_obs']
    k_min_obs = dist['k_min_obs']
    avg_k_obs = dist['avg_degree_obs']
    isolated = dist['isolated_count']

    # Probabilité théorique d'avoir k=0 dans Poisson(10)
    p0_theo = math.exp(-target_k)
    expected_isolated = n * p0_theo

    print(f"\n📊 Statistiques observées sur le réseau G1 :")
    print(f"   - Degré moyen observé ⟨k⟩     : {avg_k_obs:.4f} (théorique: {target_k})")
    print(f"   - Variance observée σ²        : {dist['var_degree_obs']:.4f} (théorique: {target_k})")
    print(f"   - Degré minimum observé       : {k_min_obs}")
    print(f"   - Nœuds isolés (k = 0)        : {isolated} nœud(s) (attendu théorique: {expected_isolated:.3f})")
    print(f"   - Fraction observée k=0       : {pk_obs[0]:.6e} (Poisson: {p0_theo:.6e})")

    # 3. Calcul du degré max attendu / prédit par la loi de Poisson
    kmax_theo_info = calculate_expected_kmax_poisson(n, target_k)
    k_max_theo = kmax_theo_info['k_closest']

    print(f"\n🎯 Analyse du degré maximum (k_max) :")
    print(f"   - Degré maximum OBSERVE       : {k_max_obs}")
    print(f"   - Degré maximum THEORIQUE     : {k_max_theo} (selon n · P(X ≥ k_max) ≈ 1)")
    print(f"     Plage théorique [k_floor, k_ceil] : [{kmax_theo_info['k_floor']}, {kmax_theo_info['k_ceil']}]")
    print(f"     N · P(X ≥ {k_max_theo}) = {kmax_theo_info['expected_nodes_closest']:.3f} nœud(s) attendu(s)")
    print(f"     Écart (observé - théorique) : {k_max_obs - k_max_theo:+d}")

    # 4. Calcul de la distribution de Poisson théorique
    # On évalue Poisson sur tout le domaine observé k = 0 .. k_max_obs
    k_extended = np.arange(0, max(k_max_obs, k_max_theo) + 3)
    poisson_pmf = compute_poisson_pmf(k_extended, target_k)

    # 5. Visualisation
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7, zorder=1)

    # Histogramme en barres de la distribution observée (fraction de nœuds)
    bars = ax.bar(
        k_vals,
        pk_obs,
        width=0.75,
        color='#3B82F6',
        edgecolor='#1E3A8A',
        linewidth=0.8,
        alpha=0.75,
        label=f'Réseau G(n={n}, p={target_k}/{n-1})\n(Observed degree distribution)',
        zorder=2
    )

    # Superposition de la distribution de Poisson théorique
    ax.plot(
        k_extended,
        poisson_pmf,
        color='#DC2626',
        linestyle='-',
        linewidth=2.2,
        marker='o',
        markersize=4.5,
        label=f'Poisson théorique (λ = {int(target_k)})\n$P(k) = \\frac{{{int(target_k)}^k e^{{-{int(target_k)}}}}}{{k!}}$',
        zorder=3
    )

    # Lignes verticales repères pour la moyenne et les k_max
    ax.axvline(
        target_k,
        color='#059669',
        linestyle=':',
        linewidth=1.8,
        label=f'Moyenne $\\langle k \\rangle = {int(target_k)}$',
        zorder=2.5
    )
    ax.axvline(
        k_max_obs,
        color='#1E40AF',
        linestyle='--',
        linewidth=1.5,
        label=f'$k_{{max}}^{{obs}} = {k_max_obs}$',
        zorder=2.5
    )
    ax.axvline(
        k_max_theo,
        color='#B91C1C',
        linestyle='-.',
        linewidth=1.5,
        label=f'$k_{{max}}^{{theo}} \\approx {k_max_theo}$',
        zorder=2.5
    )

    # Titre et libellés explicites
    ax.set_title(
        f"Partie 1 : Distribution de degrés d'un réseau aléatoire G(n={n}, p={target_k}/{n-1})\n"
        f"Comparaison avec la loi de Poisson théorique (λ = {int(target_k)})",
        fontsize=13,
        fontweight='bold',
        pad=14,
        color='#0F172A'
    )
    ax.set_xlabel("Degré $k$ (nombre de voisins, incluant $k=0$)", fontsize=11, fontweight='medium', color='#0F172A')
    ax.set_ylabel("Fraction de nœuds $P(k)$", fontsize=11, fontweight='medium', color='#0F172A')

    # Ajustement des limites pour afficher nettement k = 0
    ax.set_xlim(-0.8, max(k_max_obs, k_max_theo) + 2)
    ax.set_xticks(np.arange(0, max(k_max_obs, k_max_theo) + 3, 2))

    # Encadré récapitulatif en haut à droite
    summary_text = (
        f"Statistiques réseau :\n"
        f" • $n = {n}$\n"
        f" • $\\langle k \\rangle_{{theo}} = {target_k:.1f}$\n"
        f" • $\\langle k \\rangle_{{obs}} = {avg_k_obs:.2f}$\n"
        f" • Nœuds isolés ($k=0$) : {isolated}\n"
        f" • $k_{{max}}^{{obs}} = {k_max_obs}$\n"
        f" • $k_{{max}}^{{theo}} \\approx {k_max_theo}$"
    )
    ax.text(
        0.98, 0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.92),
        zorder=4
    )

    # Légende en haut à gauche (zone dégagée)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5)
    plt.tight_layout()

    # Création du dossier cible si nécessaire et sauvegarde
    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
    fig.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"  💾 Graphique exporté avec succès : {output_img}")

    return {
        'G': G1,
        'dist': dist,
        'kmax_theo': kmax_theo_info,
        'p': p,
        'output_img': output_img
    }


# =============================================================================
# PARTIE 2 : RÉSEAU PLUS DENSE (⟨k⟩ = 100) ET COMPARAISON NORMALE
# =============================================================================

def run_part2(n=5000, target_k=100.0, seed=42, output_img='week2/part2_normal_degree_dist.png'):
    """
    Exécute la Partie 2 :
      - Génère un second graphe aléatoire G(n, p) avec n=5000 et p=100/4999 (⟨k⟩ = 100).
      - Extrait la distribution de degrés (en barres, de k=0 à k_max).
      - Superpose l'approximation de la loi normale N(μ = 100, σ² = 100) -> σ = 10.
      - Affiche les graphiques avec axes clairs, légendes et titres soignés.
    """
    print("\n" + "=" * 78)
    print("📌 PARTIE 2 : Réseau Aléatoire G(n, p) avec ⟨k⟩ = 100 (Distribution Normale)")
    print("=" * 78)

    p = target_k / float(n - 1)
    mu = target_k
    sigma2 = target_k  # Pour Poisson(λ), variance = λ = 100 ; pour Binomial(n-1, p), variance = (n-1)*p*(1-p) ≈ 98.0
    sigma = math.sqrt(sigma2)  # σ = 10.0

    print(f"  • Nombre de nœuds (n)         : {n}")
    print(f"  • Probabilité de lien (p)     : {p:.8f} (soit {target_k}/{n-1})")
    print(f"  • Degré moyen théorique ⟨k⟩   : {p * (n - 1):.4f}")
    print(f"  • Paramètres Loi Normale      : μ = {mu:.1f}, σ² = {sigma2:.1f} (σ = {sigma:.2f})")
    print(f"  • Graine aléatoire (seed)     : {seed}")

    # 1. Génération du graphe d'Erdős-Rényi
    G2 = nx.erdos_renyi_graph(n, p, seed=seed)
    num_edges = G2.number_of_edges()
    print(f"  • Nombre d'arêtes générées    : {num_edges}")

    # 2. Distribution empirique des degrés (incluant k = 0)
    dist = extract_degree_distribution(G2, n)
    k_vals = dist['k_vals']
    pk_obs = dist['pk_obs']
    k_max_obs = dist['k_max_obs']
    k_min_obs = dist['k_min_obs']
    avg_k_obs = dist['avg_degree_obs']
    var_k_obs = dist['var_degree_obs']
    isolated = dist['isolated_count']

    print(f"\n📊 Statistiques observées sur le réseau G2 :")
    print(f"   - Degré moyen observé ⟨k⟩     : {avg_k_obs:.4f} (théorique: {mu:.1f})")
    print(f"   - Variance observée σ²        : {var_k_obs:.4f} (théorique Poisson: {sigma2:.1f}, Binomiale: {target_k * (1 - p):.4f})")
    print(f"   - Écart-type observé σ        : {math.sqrt(var_k_obs):.4f} (théorique: {sigma:.4f})")
    print(f"   - Degré minimum observé       : {k_min_obs}")
    print(f"   - Degré maximum observé       : {k_max_obs}")
    print(f"   - Nœuds isolés (k = 0)        : {isolated} nœud(s)")

    # 3. Évaluation de la courbe normale théorique
    # On trace une courbe continue lisse centrée sur la zone des degrés actifs
    x_min = max(0, int(k_min_obs - 5))
    x_max = int(k_max_obs + 5)
    x_smooth = np.linspace(x_min, x_max, 500)
    normal_curve = compute_normal_pdf(x_smooth, mu, sigma)

    # Également calcul discret sur les entiers pour comparaison directe
    k_active = np.arange(x_min, x_max + 1)
    normal_discrete = compute_normal_pdf(k_active, mu, sigma)

    # 4. Visualisation
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7, zorder=1)

    # Histogramme en barres
    # Note : on affiche la zone active avec xlim tout en ayant k=0 rigoureusement
    # dans le tableau pk_obs depuis l'indice 0.
    ax.bar(
        k_vals,
        pk_obs,
        width=0.75,
        color='#10B981',
        edgecolor='#065F46',
        linewidth=0.8,
        alpha=0.75,
        label=f'Réseau G(n={n}, p={target_k}/{n-1})\n(Observed degree distribution)',
        zorder=2
    )

    # Superposition de la courbe Normale théorique
    ax.plot(
        x_smooth,
        normal_curve,
        color='#E11D48',
        linestyle='-',
        linewidth=2.5,
        label=f'Distribution Normale théorique\n$\\mathcal{{N}}(\\mu = {int(mu)}, \\sigma^2 = {int(sigma2)})$',
        zorder=3
    )

    # Lignes verticales repères pour μ et μ ± σ
    ax.axvline(
        mu,
        color='#1E293B',
        linestyle='--',
        linewidth=1.6,
        label=f'Moyenne $\\mu = {int(mu)}$',
        zorder=2.5
    )
    ax.axvline(
        mu - sigma,
        color='#64748B',
        linestyle=':',
        linewidth=1.3,
        label=f'Intervalles $\\mu \\pm \\sigma$ ({int(mu - sigma)}, {int(mu + sigma)})',
        zorder=2.5
    )
    ax.axvline(
        mu + sigma,
        color='#64748B',
        linestyle=':',
        linewidth=1.3,
        zorder=2.5
    )

    # Titre et libellés explicites
    ax.set_title(
        f"Partie 2 : Distribution de degrés d'un réseau aléatoire G(n={n}, p={target_k}/{n-1})\n"
        f"Comparaison avec la loi Normale $\\mathcal{{N}}(\\mu = {int(mu)}, \\sigma^2 = {int(sigma2)})$",
        fontsize=13,
        fontweight='bold',
        pad=14,
        color='#0F172A'
    )
    ax.set_xlabel("Degré $k$ (nombre de voisins)", fontsize=11, fontweight='medium', color='#0F172A')
    ax.set_ylabel("Fraction de nœuds $P(k)$ / Densité $f(k)$", fontsize=11, fontweight='medium', color='#0F172A')

    # Zoom sur la zone où se concentre la distribution pour une lisibilité parfaite
    ax.set_xlim(x_min - 1, x_max + 1)

    # Encadré récapitulatif en haut à droite
    summary_text = (
        f"Statistiques réseau :\n"
        f" • $n = {n}$\n"
        f" • $\\langle k \\rangle_{{theo}} = {target_k:.1f}$\n"
        f" • $\\langle k \\rangle_{{obs}} = {avg_k_obs:.2f}$\n"
        f" • $\\sigma_{{theo}} = \\sqrt{{100}} = {sigma:.1f}$\n"
        f" • $\\sigma_{{obs}} = {math.sqrt(var_k_obs):.2f}$\n"
        f" • Min: {k_min_obs}, Max: {k_max_obs}"
    )
    ax.text(
        0.98, 0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.92),
        zorder=4
    )

    # Légende en haut à gauche (zone dégagée)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
    fig.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"  💾 Graphique exporté avec succès : {output_img}")

    return {
        'G': G2,
        'dist': dist,
        'mu': mu,
        'sigma': sigma,
        'output_img': output_img
    }


# =============================================================================
# FIGURE COMBINÉE : COMPARAISON CÔTE À CÔTE (PARTIE 1 & PARTIE 2)
# =============================================================================

def plot_combined_comparison(res1, res2, output_img='week2/random_networks_degree_distributions.png'):
    """
    Génère une figure comparative double panneau (côte à côte) regroupant
    la Partie 1 (Poisson, λ=10) et la Partie 2 (Normale, μ=100, σ=10).
    Idéal pour les rapports et devoirs DTU.
    """
    print(f"\n🎨 Génération de la figure combinée comparative : {output_img}...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5), dpi=300, facecolor='white')

    for ax in (ax1, ax2):
        ax.set_facecolor('#F8FAFC')
        ax.grid(True, linestyle='--', linewidth=0.7, color='#CBD5E1', alpha=0.7, zorder=1)

    # ------------------ Panneau Gauche : Partie 1 (Poisson) ------------------
    dist1 = res1['dist']
    k_vals1 = dist1['k_vals']
    pk_obs1 = dist1['pk_obs']
    kmax_info = res1['kmax_theo']
    kmax_theo = kmax_info['k_closest']
    kmax_obs1 = dist1['k_max_obs']

    k_ext1 = np.arange(0, max(kmax_obs1, kmax_theo) + 3)
    poisson_pmf = compute_poisson_pmf(k_ext1, 10.0)

    ax1.bar(
        k_vals1, pk_obs1,
        width=0.75, color='#3B82F6', edgecolor='#1E3A8A',
        linewidth=0.8, alpha=0.75,
        label='Degrés observés $G(5000, 10/4999)$',
        zorder=2
    )
    ax1.plot(
        k_ext1, poisson_pmf,
        color='#DC2626', linestyle='-', linewidth=2.2, marker='o', markersize=4,
        label='Poisson théorique ($\\lambda = 10$)',
        zorder=3
    )
    ax1.axvline(10, color='#059669', linestyle=':', linewidth=1.8, label='Moyenne $\\langle k \\rangle = 10$', zorder=2.5)
    ax1.axvline(kmax_obs1, color='#1E40AF', linestyle='--', linewidth=1.5, label=f'$k_{{max}}^{{obs}} = {kmax_obs1}$', zorder=2.5)
    ax1.axvline(kmax_theo, color='#B91C1C', linestyle='-.', linewidth=1.5, label=f'$k_{{max}}^{{theo}} \\approx {kmax_theo}$', zorder=2.5)

    ax1.set_title("Partie 1 : Régime clairsemé ($\\langle k \\rangle = 10$)\nLoi empirique vs Loi de Poisson", fontsize=12, fontweight='bold', color='#0F172A')
    ax1.set_xlabel("Degré $k$ (incluant $k=0$)", fontsize=11, fontweight='medium', color='#0F172A')
    ax1.set_ylabel("Fraction de nœuds $P(k)$", fontsize=11, fontweight='medium', color='#0F172A')
    ax1.set_xlim(-0.8, max(kmax_obs1, kmax_theo) + 2)
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9)

    # ----------------- Panneau Droit : Partie 2 (Normale) --------------------
    dist2 = res2['dist']
    k_vals2 = dist2['k_vals']
    pk_obs2 = dist2['pk_obs']
    kmin2 = dist2['k_min_obs']
    kmax2 = dist2['k_max_obs']

    x_min2 = max(0, int(kmin2 - 5))
    x_max2 = int(kmax2 + 5)
    x_smooth2 = np.linspace(x_min2, x_max2, 400)
    normal_curve2 = compute_normal_pdf(x_smooth2, 100.0, 10.0)

    ax2.bar(
        k_vals2, pk_obs2,
        width=0.75, color='#10B981', edgecolor='#065F46',
        linewidth=0.8, alpha=0.75,
        label='Degrés observés $G(5000, 100/4999)$',
        zorder=2
    )
    ax2.plot(
        x_smooth2, normal_curve2,
        color='#E11D48', linestyle='-', linewidth=2.4,
        label='Normale $\\mathcal{N}(\\mu = 100, \\sigma^2 = 100)$',
        zorder=3
    )
    ax2.axvline(100, color='#1E293B', linestyle='--', linewidth=1.6, label='Moyenne $\\mu = 100$', zorder=2.5)
    ax2.axvline(90, color='#64748B', linestyle=':', linewidth=1.3, label='Écart $\\mu \\pm \\sigma$ (90, 110)', zorder=2.5)
    ax2.axvline(110, color='#64748B', linestyle=':', linewidth=1.3, zorder=2.5)

    ax2.set_title("Partie 2 : Régime plus dense ($\\langle k \\rangle = 100$)\nLoi empirique vs Loi Normale", fontsize=12, fontweight='bold', color='#0F172A')
    ax2.set_xlabel("Degré $k$", fontsize=11, fontweight='medium', color='#0F172A')
    ax2.set_ylabel("Fraction de nœuds $P(k)$", fontsize=11, fontweight='medium', color='#0F172A')
    ax2.set_xlim(x_min2 - 1, x_max2 + 1)
    ax2.legend(loc='upper right', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9)

    plt.suptitle("Analyse Comparative des Distributions de Degrés dans les Réseaux Aléatoires $G(n, p)$", fontsize=14, fontweight='bold', y=0.98, color='#0F172A')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
    fig.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"  💾 Figure combinée exportée : {output_img}")


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def main():
    """Fonction principale d'exécution des deux parties."""
    print("🚀 Démarrage de l'analyse des réseaux aléatoires G(n, p)...")

    # Exécution Partie 1
    res1 = run_part1(
        n=5000,
        target_k=10.0,
        seed=42,
        output_img='week2/part1_poisson_degree_dist.png'
    )

    # Exécution Partie 2
    res2 = run_part2(
        n=5000,
        target_k=100.0,
        seed=42,
        output_img='week2/part2_normal_degree_dist.png'
    )

    # Figure combinée comparative
    plot_combined_comparison(
        res1,
        res2,
        output_img='week2/random_networks_degree_distributions.png'
    )

    print("\n" + "=" * 78)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
    print("=" * 78)
    print("Fichiers graphiques générés dans le dossier 'week2/' :")
    print("  1. week2/part1_poisson_degree_dist.png")
    print("  2. week2/part2_normal_degree_dist.png")
    print("  3. week2/random_networks_degree_distributions.png")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
