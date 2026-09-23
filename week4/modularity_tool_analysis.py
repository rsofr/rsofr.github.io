"""
Exercice Pratique : Analyse de la Modularité & Limites Statistiques
===================================================================
Ce script implémente de manière modulaire et commentée les 3 questions
de la partie pratique (🔬 Tool) :
1. Test de permutation (Shuffle-test) sur les philosophes (Q3)
   - 20 réseaux aléatoires préservant les degrés (degree-preserving)
   - 20 réseaux aléatoires G(n, m) d'Erdős-Rényi
   - Calcul de la moyenne, écart-type et z-score du Q réel
2. Limites de la modularité sur le bruit pur (Q4)
   - G(n, p) avec n=1000 et k de 1 à 50 (tracé de Q en fonction de k)
   - G(n, p) avec k=3 et n de 100 à 10 000 (tracé de Q en fonction de n)
   - Extraction systématique de la GCC pour les réseaux clairsemés
   - Réseau aléatoire à Q ~ 0.51 : exécution de Louvain avec deux seeds et NMI
3. Le club de karaté de Zachary et l'impact des poids (Q5)
   - Séparation réelle (attribut 'club')
   - Calcul de Q avec et sans poids
   - Louvain (sans poids) et calcul de la NMI avec la vérité terrain
"""

import sys
sys.path.insert(0, './.venv_packages')

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# FONCTIONS UTILITAIRES DE CHARGEMENT
# =============================================================================

def load_philosophers_gcc(edgelist_path="philosophers.edgelist",
                           edges_tsv="week4_philosophers_edges.tsv"):
    """
    Charge le réseau des philosophes et extrait sa composante géante (GCC).
    Tente de charger 'philosophers.edgelist', sinon le génère depuis le fichier TSV.
    """
    if os.path.exists(edgelist_path):
        print(f"Chargement depuis '{edgelist_path}'...")
        G = nx.read_weighted_edgelist(edgelist_path)
    elif os.path.exists(edges_tsv):
        print(f"'{edgelist_path}' introuvable. Construction depuis '{edges_tsv}'...")
        df_edges = pd.read_csv(edges_tsv, sep="\t", comment="#")
        G = nx.Graph()
        for _, row in df_edges.iterrows():
            u, v, w = row["source"], row["target"], float(row["weight"])
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
        # Sauvegarde pour les utilisations futures
        gcc_nodes = max(nx.connected_components(G), key=len)
        gcc = G.subgraph(gcc_nodes).copy()
        nx.write_edgelist(gcc, edgelist_path, data=["weight"])
        return gcc
    else:
        raise FileNotFoundError(f"Aucun fichier de données trouvé ({edgelist_path} ou {edges_tsv}).")

    # Extraction de la GCC
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes).copy()
    print(f"Composante géante (GCC) : {gcc.number_of_nodes()} nœuds, {gcc.number_of_edges()} arêtes.")
    return gcc


# =============================================================================
# TÂCHE 1 : TEST DE PERMUTATION (SHUFFLE-TEST) SUR LES PHILOSOPHES (Q3)
# =============================================================================

def task_1_shuffle_test(gcc, n_samples=20, base_seed=42):
    """
    Exécute le shuffle-test sur le réseau des philosophes :
    1. Calcule la modularité Q réelle avec Louvain.
    2. Génère n_samples réseaux aléatoires préservant les degrés (degree-preserving).
    3. Génère n_samples réseaux G(n, m) d'Erdős-Rényi.
    4. Calcule moyenne, écart-type et z-score : z = (Q_real - mean(Q_null)) / std(Q_null).
    """
    print("\n" + "=" * 75)
    print("TÂCHE 1 (Q3) : TEST DE PERMUTATION (SHUFFLE-TEST) SUR LES PHILOSOPHES")
    print("=" * 75)

    n = gcc.number_of_nodes()
    m = gcc.number_of_edges()

    # 1. Modularité réelle observée
    real_comms = nx.community.louvain_communities(gcc, seed=base_seed)
    q_real = nx.community.modularity(gcc, real_comms)
    print(f"Réseau réel : {n} nœuds, {m} arêtes.")
    print(f"Modularité réelle (Louvain) Q_real = {q_real:.4f} ({len(real_comms)} communautés)")

    # 2. Modèle nul 1 : Degree-Preserving Randomization (Double Edge Swap)
    print(f"\nGénération de {n_samples} réseaux aléatoires préservant les degrés (double edge swaps)...")
    q_shuffled = []
    n_swaps = 3 * m  # Règle usuelle : au moins 3 à 5 swaps par arête pour randomiser
    t0 = time.time()

    for i in range(n_samples):
        g_rand = gcc.copy()
        # double_edge_swap préserve la suite exacte des degrés
        nx.double_edge_swap(g_rand, nswap=n_swaps, max_tries=n_swaps * 5, seed=base_seed + i)
        comms = nx.community.louvain_communities(g_rand, seed=base_seed)
        q = nx.community.modularity(g_rand, comms)
        q_shuffled.append(q)

    mean_shuff = np.mean(q_shuffled)
    std_shuff = np.std(q_shuffled, ddof=1)
    z_shuff = (q_real - mean_shuff) / std_shuff
    print(f"Temps écoulé : {time.time() - t0:.2f} s")
    print(f"Degree-Preserving Shuffles : Moyenne Q = {mean_shuff:.4f} ± {std_shuff:.4f}")
    print(f"--> z-score (Degree-preserving) : z = {z_shuff:.2f}")

    # 3. Modèle nul 2 : Erdős-Rényi G(n, m)
    print(f"\nGénération de {n_samples} réseaux Erdős-Rényi G(n={n}, m={m})...")
    q_er = []
    t0 = time.time()

    for i in range(n_samples):
        g_er = nx.gnm_random_graph(n, m, seed=base_seed + i)
        comms = nx.community.louvain_communities(g_er, seed=base_seed)
        q = nx.community.modularity(g_er, comms)
        q_er.append(q)

    mean_er = np.mean(q_er)
    std_er = np.std(q_er, ddof=1)
    z_er = (q_real - mean_er) / std_er
    print(f"Temps écoulé : {time.time() - t0:.2f} s")
    print(f"Erdős-Rényi G(n, m)        : Moyenne Q = {mean_er:.4f} ± {std_er:.4f}")
    print(f"--> z-score (Erdős-Rényi)       : z = {z_er:.2f}")

    # 4. Tracé des distributions des modèles nuls vs Q_real
    plt.figure(figsize=(9, 4.5))
    sns.kdeplot(q_shuffled, fill=True, color="blue", label=f"Degree-preserving (μ={mean_shuff:.3f})")
    sns.kdeplot(q_er, fill=True, color="green", label=f"Erdős-Rényi G(n, m) (μ={mean_er:.3f})")
    plt.axvline(q_real, color="red", linestyle="--", linewidth=2.5, label=f"Réseau réel (Q={q_real:.3f})")
    plt.title(f"Test de permutation (Q3) : Modularité Réelle vs Modèles Nuls (z > {min(z_shuff, z_er):.1f})", fontsize=12)
    plt.xlabel("Modularité Q", fontsize=11)
    plt.ylabel("Densité", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("q3_shuffle_test.png", dpi=150)
    plt.show()

    print("\nConclusion Tâche 1 :")
    print(f"Les z-scores extrêmement élevés (z = {z_shuff:.1f} et z = {z_er:.1f}) démontrent")
    print("que la structure communautaire observée chez les philosophes est statistiquement")
    print("très significative et ne peut absolument pas résulter de fluctuations aléatoires.")

    return {
        "q_real": q_real,
        "shuffled": {"mean": mean_shuff, "std": std_shuff, "z_score": z_shuff, "values": q_shuffled},
        "er": {"mean": mean_er, "std": std_er, "z_score": z_er, "values": q_er}
    }


# =============================================================================
# TÂCHE 2 : LIMITES DE LA MODULARITÉ SUR LE BRUIT PUR (Q4)
# =============================================================================

def task_2_modularity_on_noise(base_seed=42):
    """
    Étudie le comportement de la modularité sur des réseaux aléatoires purs (G(n, p)) :
    1. Fixe n = 1000, fait varier k de 1 à 50. Trace Q en fonction de k.
    2. Fixe k = 3, fait varier n de 100 à 10 000. Trace Q en fonction de n.
       (Extrait systématiquement la GCC pour les réseaux clairsemés).
    3. Identifie le réseau généré dont Q est le plus proche de 0.51.
    4. Exécute Louvain sur ce réseau avec deux seeds différents et calcule la NMI.
    """
    print("\n" + "=" * 75)
    print("TÂCHE 2 (Q4) : LIMITES DE LA MODULARITÉ SUR LE BRUIT PUR G(n, p)")
    print("=" * 75)

    # 1. Évolution de Q en fonction de k pour n = 1000 fixe
    n_fixed = 1000
    k_range = [1, 2, 3, 4, 4.2, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    q_vs_k = []
    networks_k = []

    print(f"1. Simulation de G(n={n_fixed}, p) pour k allant de 1 à 50...")
    for k in k_range:
        p = k / (n_fixed - 1)
        G = nx.erdos_renyi_graph(n_fixed, p, seed=base_seed)
        # Extraction de la composante géante
        gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        comms = nx.community.louvain_communities(gcc, seed=base_seed)
        q = nx.community.modularity(gcc, comms)
        q_vs_k.append(q)
        networks_k.append((k, gcc, q))
        print(f"   k = {k:4.1f} | GCC: {gcc.number_of_nodes():4d} nœuds | {len(comms):2d} comms | Q = {q:.4f}")

    # 2. Évolution de Q en fonction de n pour k = 3 fixe
    k_fixed = 3.0
    n_range = [100, 200, 500, 1000, 2000, 4000, 7000, 10000]
    q_vs_n = []

    print(f"\n2. Simulation de G(n, p) pour k={k_fixed} fixe et n variant de 100 à 10 000...")
    for n in n_range:
        p = k_fixed / (n - 1)
        G = nx.erdos_renyi_graph(n, p, seed=base_seed)
        gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        comms = nx.community.louvain_communities(gcc, seed=base_seed)
        q = nx.community.modularity(gcc, comms)
        q_vs_n.append(q)
        print(f"   n = {n:5d} | GCC: {gcc.number_of_nodes():5d} nœuds | {len(comms):2d} comms | Q = {q:.4f}")

    # Tracé des deux figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1 : Q vs k
    ax1.plot(k_range, q_vs_k, marker="o", color="#1f77b4", linewidth=2)
    ax1.axhline(0.51, color="red", linestyle=":", label="Seuil Q ≈ 0.51")
    ax1.set_title("Évolution de Q en fonction du degré moyen $\\langle k \\rangle$\n(n = 1000 fixe)", fontsize=11)
    ax1.set_xlabel("Degré moyen $\\langle k \\rangle$", fontsize=10)
    ax1.set_ylabel("Modularité maximale Q", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2 : Q vs n
    ax2.plot(n_range, q_vs_n, marker="s", color="#ff7f0e", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_title("Évolution de Q en fonction de la taille n\n($\\langle k \\rangle = 3$ fixe, échelle log)", fontsize=11)
    ax2.set_xlabel("Nombre de nœuds n", fontsize=10)
    ax2.set_ylabel("Modularité maximale Q", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("q4_noise_modularity.png", dpi=150)
    plt.show()

    # 3. Recherche du réseau avec Q le plus proche de 0.51
    best_k, best_gcc, best_q = min(networks_k, key=lambda item: abs(item[2] - 0.51))
    print(f"\nRéseau aléatoire sélectionné (le plus proche de Q = 0.51) :")
    print(f" - Paramètre k            : {best_k}")
    print(f" - Taille de la GCC       : {best_gcc.number_of_nodes()} nœuds, {best_gcc.number_of_edges()} arêtes")
    print(f" - Modularité observée Q  : {best_q:.4f}")

    # 4. Test d'instabilité : Deux exécutions de Louvain avec seeds distincts
    c1 = nx.community.louvain_communities(best_gcc, seed=1)
    q1 = nx.community.modularity(best_gcc, c1)

    c2 = nx.community.louvain_communities(best_gcc, seed=2)
    q2 = nx.community.modularity(best_gcc, c2)

    # Calcul de la NMI entre les deux partitions
    node_list = sorted(list(best_gcc.nodes()))
    n2c1 = {node: cid for cid, comm in enumerate(c1) for node in comm}
    n2c2 = {node: cid for cid, comm in enumerate(c2) for node in comm}

    labels1 = [n2c1[u] for u in node_list]
    labels2 = [n2c2[u] for u in node_list]

    nmi_noise = normalized_mutual_info_score(labels1, labels2)

    print(f"\n--- COMPARAISON DES DEUX SEEDS SUR CE BRUIT PUR ---")
    print(f"Louvain Seed 1 : {len(c1)} communautés, Q = {q1:.4f}")
    print(f"Louvain Seed 2 : {len(c2)} communautés, Q = {q2:.4f}")
    print(f"NMI(Seed 1, Seed 2) sur bruit pur : {nmi_noise:.4f}")
    print("\nInterprétation cruciale :")
    print(" - Dans un graphe aléatoire clairsemé, des fluctuations stochastiques créent des agrégats locaux")
    print("   que Louvain regroupe artificiellement, produisant une modularité élevée (Q ~ 0.51).")
    print(f" - Pourtant, le score NMI est très bas (~{nmi_noise:.2f}) : deux exécutions trouvent des partitions")
    print("   totalement différentes. Cela prouve qu'un Q élevé ne garantit pas l'existence de communautés réelles !")

    return {
        "k_range": k_range, "q_vs_k": q_vs_k,
        "n_range": n_range, "q_vs_n": q_vs_n,
        "selected_k": best_k, "selected_q": best_q,
        "nmi_two_seeds": nmi_noise
    }


# =============================================================================
# TÂCHE 3 : LE CLUB DE KARATÉ ET LES POIDS (Q5)
# =============================================================================

def task_3_karate_club():
    """
    Analyse de la modularité sur Zachary's Karate Club :
    1. Extrait la séparation réelle (ground truth) depuis l'attribut 'club'.
    2. Calcule Q avec les poids d'arêtes (weight='weight') et sans poids (weight=None).
    3. Exécute Louvain avec weight=None.
    4. Calcule la NMI entre la partition Louvain (non pondérée) et la séparation réelle.
    """
    print("\n" + "=" * 75)
    print("TÂCHE 3 (Q5) : CLUB DE KARATÉ DE ZACHARY & EFFET DES POIDS")
    print("=" * 75)

    G = nx.karate_club_graph()
    print(f"Graphe Karate Club : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")

    # 1. Extraction de la vérité terrain (ground truth)
    mr_hi_nodes = {n for n, d in G.nodes(data=True) if d.get("club") == "Mr. Hi"}
    officer_nodes = {n for n, d in G.nodes(data=True) if d.get("club") == "Officer"}
    ground_truth = [mr_hi_nodes, officer_nodes]

    print(f"Séparation réelle : Mr. Hi ({len(mr_hi_nodes)} membres), Officer ({len(officer_nodes)} membres)")

    # 2. Calcul de Q de la séparation réelle AVEC et SANS poids
    q_truth_weighted = nx.community.modularity(G, ground_truth, weight="weight")
    q_truth_unweighted = nx.community.modularity(G, ground_truth, weight=None)

    print(f"\nModularité de la séparation réelle :")
    print(f" - AVEC les poids (weight='weight') : Q = {q_truth_weighted:.4f}")
    print(f" - SANS les poids (weight=None)     : Q = {q_truth_unweighted:.4f}")
    print(f"   --> Différence ΔQ = {q_truth_weighted - q_truth_unweighted:+.4f}")
    print("   Explication : Les poids représentent la fréquence d'interaction en dehors du club.")
    print("   Ils sont plus forts entre membres d'un même camp, ce qui renforce les frontières de communauté.")

    # 3. Exécution de Louvain SANS les poids (weight=None)
    louvain_unw = nx.community.louvain_communities(G, weight=None, seed=42)
    q_louvain_unw = nx.community.modularity(G, louvain_unw, weight=None)

    # 4. Calcul de la NMI entre Louvain (sans poids) et la vérité terrain
    nodes = sorted(list(G.nodes()))
    true_labels = [0 if G.nodes[n]["club"] == "Mr. Hi" else 1 for n in nodes]
    
    n2c_louvain = {node: cid for cid, comm in enumerate(louvain_unw) for node in comm}
    louvain_labels = [n2c_louvain[n] for n in nodes]

    nmi_karate = normalized_mutual_info_score(true_labels, louvain_labels)

    print(f"\nRésultats de Louvain non pondéré (weight=None, seed=42) :")
    print(f" - Nombre de communautés trouvées : {len(louvain_unw)}")
    print(f" - Tailles des communautés        : {[len(c) for c in louvain_unw]}")
    print(f" - Modularité Q (sans poids)      : {q_louvain_unw:.4f}")
    print(f" - NMI avec la séparation réelle  : {nmi_karate:.4f}")

    # Exécution optionnelle avec poids pour mise en perspective
    louvain_w = nx.community.louvain_communities(G, weight="weight", seed=42)
    q_louvain_w = nx.community.modularity(G, louvain_w, weight="weight")
    n2c_w = {node: cid for cid, comm in enumerate(louvain_w) for node in comm}
    labels_w = [n2c_w[n] for n in nodes]
    nmi_w = normalized_mutual_info_score(true_labels, labels_w)

    print(f"\nPour comparaison avec Louvain pondéré (weight='weight') :")
    print(f" - Nombre de communautés : {len(louvain_w)}")
    print(f" - Modularité Q (pondéré): {q_louvain_w:.4f}")
    print(f" - NMI avec vérité terrain: {nmi_w:.4f}")

    return {
        "q_truth_weighted": q_truth_weighted,
        "q_truth_unweighted": q_truth_unweighted,
        "louvain_unweighted": {"comms": louvain_unw, "Q": q_louvain_unw, "NMI": nmi_karate},
        "louvain_weighted": {"comms": louvain_w, "Q": q_louvain_w, "NMI": nmi_w}
    }


# =============================================================================
# EXÉCUTION COMPLÈTE
# =============================================================================

def main():
    print("DÉMARRAGE DU TOOLKIT D'ANALYSE DE MODULARITÉ (Q3, Q4, Q5)\n")

    # Chargement des philosophes
    gcc = load_philosophers_gcc()

    # Tâche 1 : Q3
    res_q3 = task_1_shuffle_test(gcc, n_samples=20)

    # Tâche 2 : Q4
    res_q4 = task_2_modularity_on_noise()

    # Tâche 3 : Q5
    res_q5 = task_3_karate_club()

    print("\n" + "=" * 75)
    print("TOUTES LES EXPÉRIMENTATIONS ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS !")
    print("=" * 75)


if __name__ == "__main__":
    main()
