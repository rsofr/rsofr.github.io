#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MARVEL SOCIAL NETWORK ANALYSIS & VISUALIZATION
===============================================================================
Ce script charge les données du réseau Marvel (nœuds et arêtes dirigées),
construit un DiGraph (en préservant les personnages isolés), calcule les degrés
entrants/sortants, et génère un scatter plot haute résolution 'marvel_in_out.png'.
===============================================================================
"""

import os
import sys
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# CONFIGURATION DES FICHIERS
# =============================================================================
# Vous pouvez laisser None pour une détection automatique des fichiers
# dans le dossier courant, ou spécifier directement vos noms de fichiers :
# Ex: NODES_FILE = "marvel_characters.csv"
#     EDGES_FILE = "marvel_edges.tsv"
NODES_FILE = None
EDGES_FILE = None


def detect_file_delimiter(filepath):
    """
    Détecte automatiquement le délimiteur (virgule ou tabulation)
    en inspectant les premières lignes du fichier.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    sample = "".join(first_lines)
    tab_count = sample.count('\t')
    comma_count = sample.count(',')
    
    return '\t' if tab_count > comma_count else ','


def auto_detect_data_files():
    """
    Recherche automatiquement dans le dossier courant les deux fichiers :
    le fichier des nœuds (personnages) et le fichier des arêtes (liens).
    """
    all_files = [
        f for f in os.listdir('.')
        if f.endswith(('.csv', '.tsv', '.txt', '.dat')) and not f.startswith('.')
    ]
    
    node_candidates = []
    edge_candidates = []
    
    for fname in all_files:
        lower_name = fname.lower()
        if any(w in lower_name for w in ['node', 'perso', 'character', 'hero', 'vert']):
            node_candidates.append(fname)
        elif any(w in lower_name for w in ['edge', 'link', 'tie', 'arête', 'arete', 'reseau', 'network']):
            edge_candidates.append(fname)
            
    nodes_f = node_candidates[0] if node_candidates else None
    edges_f = edge_candidates[0] if edge_candidates else None
    
    # Si la détection par mots-clés échoue mais qu'il y a exactement 2 fichiers :
    if (nodes_f is None or edges_f is None) and len(all_files) == 2:
        # On inspecte le contenu des deux fichiers
        f1, f2 = all_files[0], all_files[1]
        sep1 = detect_file_delimiter(f1)
        sep2 = detect_file_delimiter(f2)
        df1 = pd.read_csv(f1, sep=sep1, nrows=5, engine='python')
        df2 = pd.read_csv(f2, sep=sep2, nrows=5, engine='python')
        
        # Le fichier d'arêtes a généralement au moins 2 colonnes avec des paires
        if df1.shape[1] > df2.shape[1]:
            edges_f, nodes_f = f1, f2
        else:
            edges_f, nodes_f = f2, f1
            
    return nodes_f, edges_f


def load_data(nodes_path, edges_path):
    """
    Charge les nœuds et arêtes avec Pandas, en détectant automatiquement
    les séparateurs (virgules ou tabulations).
    """
    print(f"\n📂 Chargement des données...")
    print(f"   • Fichier des nœuds  : {nodes_path}")
    print(f"   • Fichier des arêtes : {edges_path}")

    # Détection automatique du séparateur
    sep_nodes = detect_file_delimiter(nodes_path)
    sep_edges = detect_file_delimiter(edges_path)
    print(f"   ℹ️ Séparateur détecté pour les nœuds  : '{'\\t' if sep_nodes == '\\t' else ','}'")
    print(f"   ℹ️ Séparateur détecté pour les arêtes : '{'\\t' if sep_edges == '\\t' else ','}'")

    # Lecture sécurisée avec Pandas
    df_nodes = pd.read_csv(nodes_path, sep=sep_nodes, engine='python')
    df_edges = pd.read_csv(edges_path, sep=sep_edges, engine='python')

    # Identification de la colonne des nœuds
    node_col = None
    for col in df_nodes.columns:
        if str(col).strip().lower() in ['name', 'id', 'node', 'character', 'label', 'hero']:
            node_col = col
            break
    if node_col is None:
        node_col = df_nodes.columns[0]  # Première colonne par défaut
    
    nodes_list = df_nodes[node_col].astype(str).str.strip().unique().tolist()

    # Identification des colonnes source / target
    cols_edges = list(df_edges.columns)
    src_col, dst_col = cols_edges[0], cols_edges[1]
    for col in cols_edges:
        lower = str(col).strip().lower()
        if lower in ['source', 'from', 'src', 'origin']:
            src_col = col
        elif lower in ['target', 'to', 'dst', 'destination']:
            dst_col = col

    edges_df = df_edges[[src_col, dst_col]].dropna()
    edges_list = [
        (str(src).strip(), str(dst).strip())
        for src, dst in zip(edges_df[src_col], edges_df[dst_col])
    ]

    return nodes_list, edges_list


def build_marvel_graph(nodes_list, edges_list):
    """
    Construit le DiGraph NetworkX en respectant la consigne stricte :
    Tous les nœuds sont ajoutés EN PREMIER pour ne perdre aucun personnage isolé.
    """
    print("\n🕸️  Construction du réseau Marvel (DiGraph)...")
    G = nx.DiGraph()

    # 1. Ajout de TOUS les nœuds en premier (crucial pour les isolates)
    G.add_nodes_from(nodes_list)
    print(f"   ✓ {G.number_of_nodes()} nœuds ajoutés.")

    # 2. Ajout des arêtes dirigées
    G.add_edges_from(edges_list)
    print(f"   ✓ {G.number_of_edges()} arêtes dirigées ajoutées.")

    num_isolates = nx.number_of_isolates(G)
    print(f"   ✓ Personnages isolés (isolates) préservés : {num_isolates}")

    return G


def compute_degrees(G):
    """
    Calcule le in-degree et out-degree pour chaque personnage du réseau.
    Retourne un DataFrame Pandas structuré.
    """
    print("\n📊 Calcul des degrés (In-degree & Out-degree)...")
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    df = pd.DataFrame({
        'character': list(G.nodes()),
        'in_degree': [in_degrees[n] for n in G.nodes()],
        'out_degree': [out_degrees[n] for n in G.nodes()]
    })

    # Top 3 In-degree
    top_in = df.sort_values(by=['in_degree', 'out_degree'], ascending=[False, False]).head(3)
    # Top 3 Out-degree
    top_out = df.sort_values(by=['out_degree', 'in_degree'], ascending=[False, False]).head(3)

    print("\n🏆 Top 3 In-Degree (Popularité / Cités par les autres) :")
    for _, row in top_in.iterrows():
        print(f"   ⭐ {row['character']:<30} | In-degree : {row['in_degree']:<4} | Out-degree : {row['out_degree']}")

    print("\n⚡ Top 3 Out-Degree (Effort / Citent les autres) :")
    for _, row in top_out.iterrows():
        print(f"   ⭐ {row['character']:<30} | Out-degree : {row['out_degree']:<4} | In-degree : {row['in_degree']}")

    return df, top_in, top_out


def plot_and_export(df, top_in, top_out, output_filename='marvel_in_out.png'):
    """
    Génère un scatter plot aux couleurs Marvel (rouge/bleu),
    annote les top 3 in-degree et top 3 out-degree sans doublons,
    et exporte la figure en 300 DPI.
    """
    print(f"\n🎨 Génération du graphique '{output_filename}'...")

    # Palette Marvel
    MARVEL_RED = '#ED1D24'      # Rouge iconique Marvel
    MARVEL_BLUE = '#0047AB'     # Bleu Cobalt / Captain America
    BG_COLOR = '#F8FAFC'        # Arrière-plan doux et moderne
    GRID_COLOR = '#CBD5E1'      # Grille subtile
    TEXT_COLOR = '#0F172A'      # Texte sombre lisible

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300, facecolor='white')
    ax.set_facecolor(BG_COLOR)

    # Grille pour la lisibilité
    ax.grid(True, linestyle='--', linewidth=0.7, color=GRID_COLOR, alpha=0.7, zorder=1)

    # Calcul de la couleur/taille des points pour donner du relief visuel
    # Degré total = In + Out
    total_degree = df['in_degree'] + df['out_degree']

    # Scatter Plot :
    # Axe X : Out-degree (effort de citer les autres)
    # Axe Y : In-degree (popularité conférée par les autres)
    scatter = ax.scatter(
        df['out_degree'],
        df['in_degree'],
        c=df['in_degree'],                   # Gradient lié à la popularité
        cmap='coolwarm',                     # Dégradé bleu -> rouge Marvel
        s=35 + (total_degree ** 0.5) * 6,    # Taille proportionnelle
        alpha=0.55,                          # Opacité pour visualiser la densité
        edgecolors='white',
        linewidth=0.5,
        zorder=3
    )

    # Colorbar discrète
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label('In-Degree (Popularity Intensity)', fontsize=10, fontweight='medium', color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=9)

    # -------------------------------------------------------------------------
    # GESTION DES ANNOTATIONS (Top 3 In + Top 3 Out sans doublons)
    # -------------------------------------------------------------------------
    annotated_characters = set()
    
    # 1. Annotation Top 3 In-Degree (Popularité)
    for idx, row in top_in.iterrows():
        name = row['character']
        x, y = row['out_degree'], row['in_degree']
        annotated_characters.add(name)

        ax.annotate(
            f"👑 {name}\n(In: {y}, Out: {x})",
            xy=(x, y),
            xytext=(-40, 25 + idx * 8),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor=MARVEL_RED,
                edgecolor='darkred',
                alpha=0.92,
                lw=1.2
            ),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0.15',
                color=MARVEL_RED,
                lw=1.5
            ),
            zorder=5
        )

    # 2. Annotation Top 3 Out-Degree (Effort), sans doublons
    out_idx = 0
    for _, row in top_out.iterrows():
        name = row['character']
        x, y = row['out_degree'], row['in_degree']
        
        # Éviter les doublons si un personnage est déjà annoté
        if name in annotated_characters:
            continue
            
        annotated_characters.add(name)
        
        ax.annotate(
            f"⚡ {name}\n(Out: {x}, In: {y})",
            xy=(x, y),
            xytext=(25, -20 - out_idx * 12),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor=MARVEL_BLUE,
                edgecolor='navy',
                alpha=0.92,
                lw=1.2
            ),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-0.15',
                color=MARVEL_BLUE,
                lw=1.5
            ),
            zorder=5
        )
        out_idx += 1

    # -------------------------------------------------------------------------
    # TITRES, LABELS ET MISE EN FORME
    # -------------------------------------------------------------------------
    ax.set_title(
        "Marvel Universe Social Network: Popularity vs. Citation Effort",
        fontsize=15,
        fontweight='bold',
        color=TEXT_COLOR,
        pad=18
    )
    
    ax.set_xlabel(
        "Out-degree (Effort to cite others)",
        fontsize=12,
        fontweight='semibold',
        color=TEXT_COLOR,
        labelpad=10
    )
    
    ax.set_ylabel(
        "In-degree (Popularity conferred by others)",
        fontsize=12,
        fontweight='semibold',
        color=TEXT_COLOR,
        labelpad=10
    )

    # Légende explicative
    custom_legend = [
        plt.Line2D([0], [0], marker='s', color='w', label='Top In-Degree (Popularity)',
                   markerfacecolor=MARVEL_RED, markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Top Out-Degree (Effort)',
                   markerfacecolor=MARVEL_BLUE, markersize=10)
    ]
    ax.legend(handles=custom_legend, loc='upper left', framealpha=0.9, facecolor='white')

    # Ajustement des bordures
    for spine in ax.spines.values():
        spine.set_color('#94A3B8')
        spine.set_linewidth(1.0)

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # EXPORT HAUTE RÉSOLUTION
    # -------------------------------------------------------------------------
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegarde réussie : '{output_filename}' (DPI=300)")
    plt.show()


# =============================================================================
# EXÉCUTION DU SCRIPT
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("           MARVEL NETWORK ANALYSIS & SCATTER PLOT")
    print("=" * 70)

    # 1. Résolution des chemins de fichiers
    # Vérification des arguments en ligne de commande (ex: python script.py nodes.csv edges.csv)
    if len(sys.argv) >= 3:
        nodes_path = sys.argv[1]
        edges_path = sys.argv[2]
    else:
        nodes_path = NODES_FILE
        edges_path = EDGES_FILE

    # Détection automatique si les chemins ne sont pas explicitement définis
    if not nodes_path or not edges_path:
        detected_nodes, detected_edges = auto_detect_data_files()
        nodes_path = nodes_path or detected_nodes
        edges_path = edges_path or detected_edges

    # Vérification finale de l'existence des fichiers
    if not nodes_path or not edges_path or not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print("\n❌ ERREUR : Impossible d'identifier automatiquement les fichiers.")
        print("Veuillez soit :")
        print("  1. Renseigner NODES_FILE et EDGES_FILE en haut du script.")
        print("  2. Lancer le script avec les fichiers en arguments :")
        print("     python marvel_network_analysis.py <fichier_nodes> <fichier_edges>")
        print("\nFichiers trouvés dans le dossier courant :")
        for f in os.listdir('.'):
            if f.endswith(('.csv', '.tsv', '.txt')):
                print(f"  - {f}")
        sys.exit(1)

    # 2. Chargement des données
    nodes_list, edges_list = load_data(nodes_path, edges_path)

    # 3. Construction du graphe orienté (DiGraph) en préservant les isolés
    G = build_marvel_graph(nodes_list, edges_list)

    # 4. Calculs des In-degree et Out-degree
    df_degrees, top_in, top_out = compute_degrees(G)

    # 5. Visualisation et export haute résolution
    plot_and_export(df_degrees, top_in, top_out, output_filename='marvel_in_out.png')
    print("\n🎉 Terminé avec succès !")
