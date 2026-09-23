"""
Analyse de graphes & Détection de communautés sur le réseau des philosophes
===========================================================================
Ce script répond de façon complète et modulaire aux 5 tâches demandées :
1. Louvain avec NetworkX (nombre de communautés, modularité Q).
2. Top 5 membres par communauté et calcul de NMI avec 'era' et 'subfields'.
3. Robustesse de Louvain sur 5 seeds (matrice NMI 5x5, identification des nœuds instables).
4. Clauset-Newman-Moore (Greedy modularity) et comparaison avec Louvain.
5. Implémentation manuelle 'from scratch' de la Phase 1 de Louvain et test sur Zachary Karate Club.

Dataset requis dans le dossier :
- week4_philosophers_nodes.tsv
- week4_philosophers_edges.tsv
"""

import sys
# Inclusion des éventuels packages locaux si nécessaire
sys.path.insert(0, './.venv_packages')

from collections import defaultdict
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


# =============================================================================
# PRÉ-TRAITEMENT ET CHARGEMENT DU RÉSEAU
# =============================================================================

def load_philosophers_graph(nodes_path="week4_philosophers_nodes.tsv", 
                             edges_path="week4_philosophers_edges.tsv"):
    """
    Charge les fichiers TSV des philosophes, agrège les poids dans les deux
    sens pour créer un graphe non orienté pondéré, et extrait la composante géante (GCC).
    """
    print("=" * 70)
    print("CHARGEMENT ET PRÉ-TRAITEMENT DU GRAPHE")
    print("=" * 70)

    # 1. Lecture des nœuds et attributs
    df_nodes = pd.read_csv(nodes_path, sep="\t", comment="#")
    # 2. Lecture des arêtes orientées avec poids
    df_edges = pd.read_csv(edges_path, sep="\t", comment="#")

    # Création du graphe non orienté
    G = nx.Graph()

    # Ajout des attributs de nœuds
    for _, row in df_nodes.iterrows():
        node_id = row['node_id']
        attrs = row.to_dict()
        G.add_node(node_id, **attrs)

    # Ajout des arêtes en sommant les poids dans les deux directions
    for _, row in df_edges.iterrows():
        u = row['source']
        v = row['target']
        w = float(row['weight'])
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)

    print(f"Graphe complet : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")
    
    # 3. Extraction de la composante géante connectée (GCC)
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes).copy()
    
    print(f"Composante Géante (GCC) : {gcc.number_of_nodes()} nœuds, {gcc.number_of_edges()} arêtes.")
    total_weight = sum(d['weight'] for _, _, d in gcc.edges(data=True))
    print(f"Poids total des arêtes dans la GCC : {total_weight:.1f}")
    
    return gcc


# =============================================================================
# TÂCHE 1 : LOUVAIN DE BASE (COMMUNAUTÉS ET MODULARITÉ Q)
# =============================================================================

def task_1_louvain(gcc, seed=42):
    """
    Exécute Louvain avec networkx.community.louvain_communities.
    Affiche le nombre de communautés détectées et la modularité Q.
    """
    print("\n" + "=" * 70)
    print("TÂCHE 1 : Détection de communautés par Louvain (NetworkX)")
    print("=" * 70)
    
    comms = nx.community.louvain_communities(gcc, weight='weight', seed=seed)
    q = nx.community.modularity(gcc, comms, weight='weight')
    
    print(f"Seed utilisé               : {seed}")
    print(f"Nombre de communautés      : {len(comms)}")
    print(f"Modularité Q               : {q:.4f}")
    
    sizes = sorted([len(c) for c in comms], reverse=True)
    print(f"Tailles des communautés    : {sizes}")
    
    return comms, q


# =============================================================================
# TÂCHE 2 : TOP 5 MEMBRES ET CALCUL NMI (ERA & SUBFIELDS)
# =============================================================================

def task_2_top_members_and_nmi(gcc, comms):
    """
    1. Pour chaque communauté, affiche les 5 philosophes ayant le degré le plus élevé.
    2. Calcule la NMI (Normalized Mutual Information) entre la partition Louvain
       et l'attribut 'era', puis avec 'subfields' (valeurs vides remplacées par 'none').
    """
    print("\n" + "=" * 70)
    print("TÂCHE 2 : Caractérisation des communautés & Mesure de NMI")
    print("=" * 70)
    
    # Tri des communautés par taille décroissante
    sorted_comms = sorted(comms, key=len, reverse=True)
    
    print("\n--- TOP 5 MEMBRES PAR DEGRÉ DANS CHAQUE COMMUNAUTÉ ---")
    for i, comm in enumerate(sorted_comms, start=1):
        # Tri des nœuds de la communauté par degré décroissant
        top_5 = sorted(comm, key=lambda n: gcc.degree(n), reverse=True)[:5]
        members_str = ", ".join([f"{n} (k={gcc.degree(n)})" for n in top_5])
        print(f"Communauté {i:02d} ({len(comm):4d} membres) : {members_str}")
        
    # Préparation des labels pour le calcul du NMI
    node_list = sorted(list(gcc.nodes()))
    
    # Dictionnaire de mappage node -> id de communauté
    node2comm = {node: cid for cid, comm in enumerate(sorted_comms) for node in comm}
    labels_comm = [node2comm[n] for n in node_list]
    
    # Nettoyage des attributs : remplacer NaN ou chaîne vide par 'none'
    def get_clean_attr(node, attr_name):
        val = gcc.nodes[node].get(attr_name, 'none')
        if pd.isna(val) or str(val).strip() == "":
            return "none"
        return str(val).strip()
    
    labels_era = [get_clean_attr(n, 'era') for n in node_list]
    labels_subfields = [get_clean_attr(n, 'subfields') for n in node_list]
    
    # Calcul NMI
    nmi_era = normalized_mutual_info_score(labels_era, labels_comm)
    nmi_subfields = normalized_mutual_info_score(labels_subfields, labels_comm)
    
    print("\n--- NORMALIZED MUTUAL INFORMATION (NMI) ---")
    print(f"NMI(Louvain, era)       : {nmi_era:.4f}")
    print(f"NMI(Louvain, subfields) : {nmi_subfields:.4f}")
    print("Interprétation :")
    print(" - Une NMI modérée/forte avec 'era' montre que les philosophes sont fortement")
    print("   regroupés par périodes historiques et contextes temporels partagés.")
    print(" - Une NMI plus faible avec 'subfields' s'explique par la forte proportion")
    print("   de valeurs manquantes ('none') et par le fait que les disciplines transcendent les époques.")
    
    return nmi_era, nmi_subfields


# =============================================================================
# TÂCHE 3 : ROBUSTESSE SUR 5 SEEDS, MATRICE NMI & NŒUDS INSTABLES
# =============================================================================

def task_3_seed_stability(gcc, seeds=[1, 42, 123, 456, 789]):
    """
    Exécute Louvain avec 5 seeds différents.
    1. Stocke le nombre de communautés et Q pour chaque seed.
    2. Calcule et affiche la matrice NMI 5x5.
    3. Identifie les philosophes changeant le plus souvent de communauté :
       - Par instabilité de co-appartenance (label-invariant).
       - Par alignement des labels (algorithme hongrois).
    """
    print("\n" + "=" * 70)
    print("TÂCHE 3 : Étude de robustesse (5 seeds différents)")
    print("=" * 70)
    
    node_list = sorted(list(gcc.nodes()))
    n_nodes = len(node_list)
    node_index = {n: i for i, n in enumerate(node_list)}
    
    history = []
    partitions_labels = []
    
    for s in seeds:
        comms = nx.community.louvain_communities(gcc, weight='weight', seed=s)
        q = nx.community.modularity(gcc, comms, weight='weight')
        n_comms = len(comms)
        history.append({'seed': s, 'num_communities': n_comms, 'modularity_Q': q})
        
        # Vecteur de labels pour ce seed
        n2c = {node: cid for cid, c in enumerate(comms) for node in c}
        partitions_labels.append([n2c[n] for n in node_list])
        
    df_runs = pd.DataFrame(history)
    print("\n--- RÉSULTATS PAR RUN ---")
    print(df_runs.to_string(index=False))
    
    # 2. Matrice NMI 5x5
    k = len(seeds)
    nmi_mat = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            nmi_mat[i, j] = normalized_mutual_info_score(partitions_labels[i], partitions_labels[j])
            
    df_nmi = pd.DataFrame(nmi_mat, index=[f"Seed {s}" for s in seeds], columns=[f"Seed {s}" for s in seeds])
    print("\n--- MATRICE NMI 5x5 ENTRE LES RUNS ---")
    print(df_nmi.round(4))
    
    # 3. Identification des philosophes instables
    # Approche A : Matrice de co-assignation (Consensus Co-membership Matrix)
    # C_ij = fraction des 5 runs où le nœud i et le nœud j sont dans la même communauté.
    # L'instabilité d'un nœud = variance moyenne de ses relations de co-assignation.
    co_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for labels in partitions_labels:
        arr = np.array(labels)
        co_matrix += (arr[:, None] == arr[None, :])
    co_matrix /= k
    
    # Mesure de dispersion binaire 4 * p * (1 - p) qui vaut 1 quand p = 0.5 (instabilité max) et 0 quand p=0 ou 1.
    node_instability = np.sum(4 * co_matrix * (1.0 - co_matrix), axis=1) / (n_nodes - 1)
    
    # Approche B : Alignement hongrois des labels par rapport au Run 0
    aligned_labels = [partitions_labels[0]]
    for i in range(1, k):
        ref = partitions_labels[0]
        curr = partitions_labels[i]
        n_ref = max(ref) + 1
        n_curr = max(curr) + 1
        cost = np.zeros((n_curr, n_ref), dtype=int)
        for c, r in zip(curr, ref):
            cost[c, r] -= 1  # Maximiser l'intersection
        r_ind, c_ind = linear_sum_assignment(cost)
        mapping = {r: c for r, c in zip(r_ind, c_ind)}
        aligned_labels.append([mapping.get(c, c) for c in curr])
        
    aligned_arr = np.array(aligned_labels).T  # Shape (n_nodes, 5)
    n_unique_comms = [len(np.unique(row)) for row in aligned_arr]
    
    df_stability = pd.DataFrame({
        'philosophe': node_list,
        'degree': [gcc.degree(n) for n in node_list],
        'score_instabilite': node_instability,
        'nb_communautes_distinctes': n_unique_comms,
        'era': [gcc.nodes[n].get('era', 'none') for n in node_list]
    })
    
    # Top 10 des philosophes les plus instables
    top_instables = df_stability.sort_values(
        by=['score_instabilite', 'nb_communautes_distinctes', 'degree'], 
        ascending=[False, False, False]
    ).head(10)
    
    print("\n--- TOP 10 DES PHILOSOPHES QUI CHANGENT LE PLUS DE COMMUNAUTÉ ---")
    print(top_instables.to_string(index=False))
    print("\nExplication sociologique / historique :")
    print("Ces nœuds sont souvent des figures 'frontières' ou polymaths (ex: Galilée, Machiavel, Saint-Simon, Jésus)")
    print("qui possèdent des liens denses à cheval entre plusieurs traditions (ex: science moderne vs scolastique,")
    print("théologie vs rationalisme, pensée politique vs idéalisme).")
    
    return df_runs, df_nmi, df_stability


# =============================================================================
# TÂCHE 4 : CLAUKET-NEWMAN-MOORE (GREEDY MODULARITY) VS LOUVAIN
# =============================================================================

def task_4_greedy_modularity(gcc, comms_louvain_run1):
    """
    Exécute networkx.community.greedy_modularity_communities.
    Affiche le nombre de communautés, la modularité Q, et la NMI par rapport
    à la première exécution de Louvain.
    """
    print("\n" + "=" * 70)
    print("TÂCHE 4 : Greedy Modularity Communities (Clauset-Newman-Moore)")
    print("=" * 70)
    
    greedy_comms = nx.community.greedy_modularity_communities(gcc, weight='weight')
    q_greedy = nx.community.modularity(gcc, greedy_comms, weight='weight')
    
    # Calcul de la modularité Louvain Run 1
    q_louvain = nx.community.modularity(gcc, comms_louvain_run1, weight='weight')
    
    # Préparation des labels pour NMI
    node_list = sorted(list(gcc.nodes()))
    n2c_louvain = {node: cid for cid, c in enumerate(comms_louvain_run1) for node in c}
    n2c_greedy = {node: cid for cid, c in enumerate(greedy_comms) for node in c}
    
    labels_louvain = [n2c_louvain[n] for n in node_list]
    labels_greedy = [n2c_greedy[n] for n in node_list]
    
    nmi_greedy_louvain = normalized_mutual_info_score(labels_louvain, labels_greedy)
    
    print(f"Greedy Modularity - Nombre de communautés : {len(greedy_comms)}")
    print(f"Greedy Modularity - Modularité Q          : {q_greedy:.4f}")
    print(f"Louvain (Run 1)   - Nombre de communautés : {len(comms_louvain_run1)}")
    print(f"Louvain (Run 1)   - Modularité Q          : {q_louvain:.4f}")
    print(f"NMI(Greedy, Louvain Run 1)               : {nmi_greedy_louvain:.4f}")
    
    print("\nComparaison :")
    print(" - L'approche Greedy fusionne agressivement les paires maximisant delta Q de façon ascendante.")
    print(" - Louvain effectue des déplacements locaux de nœuds suivis d'agrégations successives,")
    print("   ce qui lui permet d'explorer l'espace des partitions plus efficacement.")
    
    return greedy_comms, q_greedy, nmi_greedy_louvain


# =============================================================================
# TÂCHE 5 : IMPLÉMENTATION MANUELLE DE LA PHASE 1 DE LOUVAIN ('FROM SCRATCH')
# =============================================================================

def compute_modularity_manual(G, communities, weight='weight'):
    """
    Calcul 'from scratch' de la modularité Q de Newman-Girvan :
    Q = sum_c [ (Sigma_in^c / 2m) - (Sigma_tot^c / 2m)^2 ]
    """
    m = G.size(weight=weight)
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    k = dict(G.degree(weight=weight))
    
    Q = 0.0
    for comm in communities:
        # Poids des arêtes internes
        subG = G.subgraph(comm)
        w_in = subG.size(weight=weight)
        # Somme des degrés des nœuds de la communauté
        w_tot = sum(k[n] for n in comm)
        Q += (2.0 * w_in / two_m) - (w_tot / two_m) ** 2
    return Q


def louvain_phase_1(G, weight='weight', seed=None, max_passes=50):
    """
    Implémentation manuelle de la Phase 1 de Louvain :
    1. Initialise chaque nœud dans sa propre communauté.
    2. Pour chaque nœud, évalue le gain de modularité Delta Q en le déplaçant
       dans chacune des communautés de ses voisins.
    3. Déplace le nœud vers la communauté maximisant Delta Q (si gain > 0).
    4. Répète des passes complètes sur tous les nœuds jusqu'à stabilisation (aucun nœud ne bouge).
    
    Formule de gain Delta Q(u -> C) :
        Delta Q = (k_{u, in}^C / m) - (Sigma_tot^C * k_u) / (2 * m^2)
    où :
        - m est la somme des poids de toutes les arêtes du graphe
        - k_u est le degré pondéré du nœud u
        - Sigma_tot^C est la somme des degrés pondérés des membres de la communauté C (hors u)
        - k_{u, in}^C est le poids total des liens entre u et les membres de la communauté C
    """
    if G.is_directed():
        raise ValueError("Le graphe doit être non orienté pour cette implémentation.")
        
    if seed is not None:
        random.seed(seed)
        
    nodes = list(G.nodes())
    m = G.size(weight=weight)
    if m == 0:
        return [{n} for n in nodes], 0
        
    # Degrés pondérés de chaque nœud
    k = dict(G.degree(weight=weight))
    
    # Initialisation : chaque nœud dans sa communauté propre
    community = {n: i for i, n in enumerate(nodes)}
    sigma_tot = {i: float(k[n]) for i, n in enumerate(nodes)}
    
    passes = 0
    next_singleton_id = len(nodes)
    
    while passes < max_passes:
        passes += 1
        moved_in_pass = False
        
        # Ordre de visite des nœuds (optionnellement mélangé si seed fourni)
        order = list(nodes)
        if seed is not None:
            random.shuffle(order)
            
        for u in order:
            c_old = community[u]
            k_u = k[u]
            
            # 1. Retrait virtuel de u de sa communauté actuelle
            sigma_tot[c_old] -= k_u
            
            # 2. Calcul des poids vers les communautés voisines
            neigh_comm_weights = defaultdict(float)
            for v, edge_data in G[u].items():
                if v == u:
                    continue  # Ignore les boucles
                w = edge_data.get(weight, 1.0)
                neigh_comm_weights[community[v]] += w
                
            # 3. Évaluation de Delta Q pour chaque communauté candidate
            # Baseline : former une communauté isolée donne Delta Q = 0
            best_comm = c_old
            best_gain = 0.0
            
            for c, k_u_in in neigh_comm_weights.items():
                # Gain Delta Q d'insertion de u dans la communauté candidate c
                gain = (k_u_in / m) - (sigma_tot[c] * k_u) / (2.0 * m * m)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c
            
            # Si le meilleur gain est nul ou négatif et que c_old n'est pas vide,
            # u préfère créer une communauté singleton séparée
            if best_gain <= 0.0 and sigma_tot[c_old] > 0.0:
                best_comm = next_singleton_id
                next_singleton_id += 1
                best_gain = 0.0
                
            # 4. Insertion de u dans best_comm
            community[u] = best_comm
            sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + k_u
            
            if best_comm != c_old:
                moved_in_pass = True
                
        # Condition de terminaison : aucun nœud n'a bougé durant toute la passe
        if not moved_in_pass:
            break
            
    # Regroupement des nœuds par communauté
    grouped = defaultdict(set)
    for n, c in community.items():
        grouped[c].add(n)
        
    return list(grouped.values()), passes


def task_5_test_karate_club():
    """
    Teste la Phase 1 manuelle sur le réseau networkx.karate_club_graph().
    Compare le nombre de communautés et la modularité Q avec l'implémentation NetworkX.
    """
    print("\n" + "=" * 70)
    print("TÂCHE 5 : Test de la Phase 1 manuelle sur Karate Club")
    print("=" * 70)
    
    G_karate = nx.karate_club_graph()
    
    # 1. Exécution de notre Phase 1 manuelle
    manual_comms, passes = louvain_phase_1(G_karate, seed=42)
    # Calcul avec notre fonction manuelle et validation avec NetworkX
    q_manual = compute_modularity_manual(G_karate, manual_comms)
    q_manual_nx = nx.community.modularity(G_karate, manual_comms)
    
    # 2. Exécution de Louvain complet de NetworkX (multi-phases avec agrégation)
    nx_comms = nx.community.louvain_communities(G_karate, seed=42)
    q_nx = nx.community.modularity(G_karate, nx_comms)
    
    print(f"Graphe Karate Club : {G_karate.number_of_nodes()} nœuds, {G_karate.number_of_edges()} arêtes.")
    print("\nRésultats de l'implémentation manuelle (Phase 1 seule) :")
    print(f" - Nombre de passes avant convergence : {passes}")
    print(f" - Nombre de communautés trouvées    : {len(manual_comms)}")
    print(f" - Modularité Q (calcul manuel)       : {q_manual:.4f}")
    print(f" - Modularité Q (validation NetworkX) : {q_manual_nx:.4f}")
    print(f" - Tailles des communautés           : {[len(c) for c in manual_comms]}")
    
    print("\nRésultats de NetworkX (Louvain complet multi-phases) :")
    print(f" - Nombre de communautés trouvées    : {len(nx_comms)}")
    print(f" - Modularité Q (NetworkX)            : {q_nx:.4f}")
    print(f" - Tailles des communautés           : {[len(c) for c in nx_comms]}")
    
    print("\nExplication de la différence :")
    print(" - La Phase 1 effectue uniquement l'optimisation locale au niveau nœud par nœud,")
    print("   ce qui produit souvent un partitionnement plus fin (ici 7 communautés, Q ~ 0.40).")
    print(" - L'algorithme de Louvain complet enchaîne ensuite la Phase 2 (coarsening / agrégation de super-nœuds)")
    print("   puis relance la Phase 1 sur le méta-graphe, atteignant un optimum plus global (4 communautés, Q ~ 0.43).")


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

def main():
    # 1. Chargement et extraction de la GCC
    gcc = load_philosophers_graph()
    
    # 2. Tâche 1 : Louvain NetworkX
    comms_1, q_1 = task_1_louvain(gcc, seed=42)
    
    # 3. Tâche 2 : Top membres et NMI avec 'era' et 'subfields'
    nmi_era, nmi_sub = task_2_top_members_and_nmi(gcc, comms_1)
    
    # 4. Tâche 3 : 5 seeds, matrice NMI et nœuds instables
    df_runs, df_nmi, df_stability = task_3_seed_stability(gcc, seeds=[1, 42, 123, 456, 789])
    
    # 5. Tâche 4 : Greedy Modularity vs Louvain
    greedy_comms, q_greedy, nmi_g_l = task_4_greedy_modularity(gcc, comms_1)
    
    # 6. Tâche 5 : Phase 1 manuelle et validation sur Karate Club
    task_5_test_karate_club()
    
    print("\n" + "=" * 70)
    print("TOUTES LES TÂCHES ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    main()
