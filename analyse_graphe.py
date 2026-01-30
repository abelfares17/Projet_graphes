"""
Projet Graphes 2025-2026 - Analyse d'un essaim de nanosatellites
INPT N7 - 2SN-LMB
"""

import matplotlib
matplotlib.use('macosx')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import os
import subprocess

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fichiers de données (3 densités)
DATA_FILES = {
    'low': 'data/topology_low.csv',
    'avg': 'data/topology_avg.csv',
    'high': 'data/topology_high.csv'
}

# Portées de communication (en mètres)
PORTEES = {
    'courte': 20_000,    # 20 km
    'moyenne': 40_000,   # 40 km
    'longue': 60_000     # 60 km
}

# =============================================================================
# CACHE GLOBAL - Évite de recharger/reconstruire les données
# =============================================================================

_cache_donnees = {}      # Cache des DataFrames satellites
_cache_graphes = {}      # Cache des graphes non pondérés
_cache_graphes_pond = {} # Cache des graphes pondérés

# =============================================================================
# FONCTIONS DE BASE (avec cache)
# =============================================================================

def charger_donnees(densite):
    """
    Charge les positions des satellites depuis un fichier CSV.
    Utilise un cache pour éviter les rechargements.

    Args:
        densite: 'low', 'avg' ou 'high'
    Returns:
        DataFrame avec les colonnes sat_id, x, y, z
    """
    if densite not in _cache_donnees:
        fichier = DATA_FILES[densite]
        df = pd.read_csv(fichier)
        _cache_donnees[densite] = df
        print(f"✓ {len(df)} satellites chargés depuis {fichier}")
    return _cache_donnees[densite]


def calculer_distance(p1, p2):
    """Calcule la distance euclidienne entre deux points 3D."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def obtenir_graphe(densite, portee_nom, pondere=False):
    """
    Obtient un graphe depuis le cache ou le construit si nécessaire.

    Args:
        densite: 'low', 'avg' ou 'high'
        portee_nom: 'courte', 'moyenne' ou 'longue'
        pondere: si True, retourne un graphe avec poids = distance²
    Returns:
        Graphe NetworkX
    """
    cache_key = (densite, portee_nom)
    cache = _cache_graphes_pond if pondere else _cache_graphes

    if cache_key not in cache:
        satellites = charger_donnees(densite)
        portee = PORTEES[portee_nom]

        if pondere:
            G = construire_graphe_pondere(satellites, portee)
        else:
            G = construire_graphe(satellites, portee)

        cache[cache_key] = G

    return cache[cache_key]


def construire_graphe(satellites, portee):
    """
    Construit un graphe non pondéré où une arête existe
    si deux satellites sont à portée de communication.
    (Fonction interne - utiliser obtenir_graphe())
    """
    G = nx.Graph()

    # Ajouter tous les satellites comme nœuds
    for _, sat in satellites.iterrows():
        G.add_node(sat['sat_id'], pos=(sat['x'], sat['y'], sat['z']))

    # Ajouter les arêtes si distance <= portée
    positions = satellites[['sat_id', 'x', 'y', 'z']].values

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            sat_i = positions[i]
            sat_j = positions[j]

            distance = calculer_distance(
                (sat_i[1], sat_i[2], sat_i[3]),
                (sat_j[1], sat_j[2], sat_j[3])
            )

            if distance <= portee:
                G.add_edge(int(sat_i[0]), int(sat_j[0]))

    return G


def construire_graphe_pondere(satellites, portee):
    """
    Construit un graphe pondéré où le poids = distance².
    (Fonction interne - utiliser obtenir_graphe(pondere=True))
    """
    G = nx.Graph()

    # Ajouter tous les satellites comme nœuds
    for _, sat in satellites.iterrows():
        G.add_node(sat['sat_id'], pos=(sat['x'], sat['y'], sat['z']))

    positions = satellites[['sat_id', 'x', 'y', 'z']].values

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            sat_i = positions[i]
            sat_j = positions[j]

            distance = calculer_distance(
                (sat_i[1], sat_i[2], sat_i[3]),
                (sat_j[1], sat_j[2], sat_j[3])
            )

            if distance <= portee:
                G.add_edge(int(sat_i[0]), int(sat_j[0]), weight=distance**2)

    return G


# =============================================================================
# FONCTIONS D'ANALYSE RÉUTILISABLES
# =============================================================================

def analyser_degres(G):
    """
    Analyse les degrés du graphe.
    Returns: dict avec moyenne, min, max, distribution
    """
    degres = [d for n, d in G.degree()]
    return {
        'liste': degres,
        'moyenne': np.mean(degres),
        'min': min(degres),
        'max': max(degres),
        'distribution': Counter(degres)
    }


def analyser_clustering(G):
    """
    Analyse le coefficient de clustering du graphe.
    Returns: dict avec moyenne, min, max, liste
    """
    clustering = list(nx.clustering(G).values())
    return {
        'liste': clustering,
        'moyenne': np.mean(clustering),
        'min': min(clustering),
        'max': max(clustering)
    }


def analyser_cliques(G):
    """
    Analyse les cliques du graphe.
    Returns: dict avec total, par_ordre, max_ordre
    """
    cliques = list(nx.find_cliques(G))
    ordres = Counter([len(c) for c in cliques])
    return {
        'total': len(cliques),
        'par_ordre': dict(sorted(ordres.items())),
        'max_ordre': max(ordres.keys())
    }


def analyser_composantes(G):
    """
    Analyse les composantes connexes du graphe.
    Returns: dict avec nombre, par_ordre, liste
    """
    composantes = list(nx.connected_components(G))
    ordres = Counter([len(c) for c in composantes])
    return {
        'liste': composantes,
        'nombre': len(composantes),
        'par_ordre': dict(sorted(ordres.items())),
        'est_connexe': len(composantes) == 1
    }


def analyser_chemins(G):
    """
    Analyse les plus courts chemins (en nombre de sauts).
    Returns: dict avec longueurs, moyenne, diametre, distribution
    """
    composantes = analyser_composantes(G)
    longueurs = []

    if composantes['est_connexe']:
        all_paths = dict(nx.all_pairs_shortest_path_length(G))
        longueurs = [l for s in all_paths for t, l in all_paths[s].items() if s < t]
    else:
        # Calculer pour chaque composante
        for comp in composantes['liste']:
            if len(comp) > 1:
                subG = G.subgraph(comp)
                paths = dict(nx.all_pairs_shortest_path_length(subG))
                longueurs.extend([l for s in paths for t, l in paths[s].items() if s < t])

    if longueurs:
        return {
            'longueurs': longueurs,
            'paires': len(longueurs),
            'moyenne': np.mean(longueurs),
            'diametre': max(longueurs),
            'distribution': Counter(longueurs)
        }
    return None


def analyser_poids(G):
    """
    Analyse les poids des arêtes (graphe pondéré uniquement).
    Returns: dict avec total, moyenne, min, max, std, liste
    """
    poids = [d['weight'] for u, v, d in G.edges(data=True)]
    if not poids:
        return None
    return {
        'liste': poids,
        'total': sum(poids),
        'moyenne': np.mean(poids),
        'min': min(poids),
        'max': max(poids),
        'std': np.std(poids)
    }


def analyser_chemins_ponderes(G):
    """
    Analyse les plus courts chemins pondérés (Dijkstra).
    Returns: dict avec couts, moyenne, diametre, etc.
    """
    composantes = analyser_composantes(G)
    couts = []

    if composantes['est_connexe']:
        all_paths = dict(nx.all_pairs_dijkstra_path_length(G))
        couts = [c for s in all_paths for t, c in all_paths[s].items() if s < t]
    else:
        for comp in composantes['liste']:
            if len(comp) > 1:
                subG = G.subgraph(comp)
                paths = dict(nx.all_pairs_dijkstra_path_length(subG))
                couts.extend([c for s in paths for t, c in paths[s].items() if s < t])

    if couts:
        return {
            'couts': couts,
            'paires': len(couts),
            'moyenne': np.mean(couts),
            'min': min(couts),
            'max': max(couts),
            'diametre_pondere': max(couts),
            'std': np.std(couts)
        }
    return None


def analyser_mst(G):
    """
    Analyse l'arbre couvrant minimum (ou forêt si non connexe).
    Returns: dict avec poids_total, nb_aretes, stats arêtes
    """
    composantes = analyser_composantes(G)

    if composantes['est_connexe']:
        mst = nx.minimum_spanning_tree(G)
        poids_aretes = [d['weight'] for u, v, d in mst.edges(data=True)]
        return {
            'poids_total': sum(poids_aretes),
            'nb_aretes': mst.number_of_edges(),
            'arete_min': min(poids_aretes),
            'arete_max': max(poids_aretes),
            'arete_moyenne': np.mean(poids_aretes),
            'est_arbre': True
        }
    else:
        forest = nx.minimum_spanning_tree(G)
        poids_aretes = [d['weight'] for u, v, d in forest.edges(data=True)]
        return {
            'poids_total': sum(poids_aretes) if poids_aretes else 0,
            'nb_aretes': forest.number_of_edges(),
            'est_arbre': False
        }


def analyser_centralite(G):
    """
    Analyse la centralité de proximité pondérée.
    Returns: dict avec top_nodes ou None si non connexe
    """
    if not analyser_composantes(G)['est_connexe']:
        return None

    closeness = nx.closeness_centrality(G, distance='weight')
    top_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        'top_5': [(n, round(c, 4)) for n, c in top_nodes]
    }


# =============================================================================
# GÉNÉRATION D'HISTOGRAMMES
# =============================================================================

def generer_histogrammes_partie2(degres, clustering, cliques, chemins, config_name, titre):
    """
    Génère les histogrammes pour la Partie 2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(titre, fontsize=14, fontweight='bold')

    # Histogramme des degrés
    axes[0, 0].hist(degres['liste'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(degres['moyenne'], color='red', linestyle='--',
                       label=f"Moyenne: {degres['moyenne']:.1f}")
    axes[0, 0].set_xlabel('Degré')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des degrés')
    axes[0, 0].legend()

    # Histogramme du clustering
    axes[0, 1].hist(clustering['liste'], bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(clustering['moyenne'], color='red', linestyle='--',
                       label=f"Moyenne: {clustering['moyenne']:.3f}")
    axes[0, 1].set_xlabel('Coefficient de clustering')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution du clustering')
    axes[0, 1].legend()

    # Histogramme des cliques
    ordres = list(cliques['par_ordre'].keys())
    counts = list(cliques['par_ordre'].values())
    axes[1, 0].bar(ordres, counts, color='darkorange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Ordre de la clique')
    axes[1, 0].set_ylabel('Nombre de cliques')
    axes[1, 0].set_title('Distribution des cliques par ordre')

    # Histogramme des plus courts chemins
    if chemins:
        axes[1, 1].hist(chemins['longueurs'], bins=range(1, chemins['diametre']+2),
                        color='purple', edgecolor='black', alpha=0.7, align='left')
        axes[1, 1].axvline(chemins['moyenne'], color='red', linestyle='--',
                           label=f"Moyenne: {chemins['moyenne']:.2f}")
        axes[1, 1].set_xlabel('Longueur (en sauts)')
        axes[1, 1].set_ylabel('Nombre de paires')
        axes[1, 1].set_title('Distribution des plus courts chemins')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Pas de données', ha='center', va='center')
        axes[1, 1].set_title('Distribution des plus courts chemins')

    plt.tight_layout()
    plt.savefig(f"histogrammes/hist_{config_name}.png", dpi=150)
    plt.close()


def generer_histogrammes_partie3(degres, clustering, cliques, chemins, poids, chemins_pond, densite):
    """
    Génère les histogrammes pour la Partie 3.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Partie 3 - Distributions - {densite} / portée 60 km (coût = distance²)",
                 fontsize=14, fontweight='bold')

    # Histogramme des degrés
    axes[0, 0].hist(degres['liste'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(degres['moyenne'], color='red', linestyle='--',
                       label=f"Moyenne: {degres['moyenne']:.1f}")
    axes[0, 0].set_xlabel('Degré')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des degrés')
    axes[0, 0].legend()

    # Histogramme du clustering
    axes[0, 1].hist(clustering['liste'], bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(clustering['moyenne'], color='red', linestyle='--',
                       label=f"Moyenne: {clustering['moyenne']:.3f}")
    axes[0, 1].set_xlabel('Coefficient de clustering')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution du clustering')
    axes[0, 1].legend()

    # Histogramme des cliques
    ordres = list(cliques['par_ordre'].keys())
    counts = list(cliques['par_ordre'].values())
    axes[0, 2].bar(ordres, counts, color='darkorange', edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Ordre de la clique')
    axes[0, 2].set_ylabel('Nombre de cliques')
    axes[0, 2].set_title('Distribution des cliques par ordre')

    # Histogramme des plus courts chemins (sauts)
    if chemins:
        axes[1, 0].hist(chemins['longueurs'], bins=range(1, chemins['diametre']+2),
                        color='purple', edgecolor='black', alpha=0.7, align='left')
        axes[1, 0].axvline(chemins['moyenne'], color='red', linestyle='--',
                           label=f"Moyenne: {chemins['moyenne']:.2f}")
        axes[1, 0].set_xlabel('Longueur (en sauts)')
        axes[1, 0].set_ylabel('Nombre de paires')
        axes[1, 0].set_title('Distribution des plus courts chemins (sauts)')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Pas de données', ha='center', va='center')
        axes[1, 0].set_title('Distribution des plus courts chemins (sauts)')

    # Histogramme des poids des arêtes
    if poids:
        axes[1, 1].hist(poids['liste'], bins=30, color='crimson', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(poids['moyenne'], color='blue', linestyle='--',
                           label=f"Moyenne: {poids['moyenne']:.2e}")
        axes[1, 1].set_xlabel('Poids (distance²)')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].set_title('Distribution des poids des arêtes')
        axes[1, 1].legend()
        axes[1, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # Histogramme des coûts pondérés
    if chemins_pond:
        axes[1, 2].hist(chemins_pond['couts'], bins=30, color='teal', edgecolor='black', alpha=0.7)
        axes[1, 2].axvline(chemins_pond['moyenne'], color='red', linestyle='--',
                           label=f"Moyenne: {chemins_pond['moyenne']:.2e}")
        axes[1, 2].set_xlabel('Coût pondéré (distance²)')
        axes[1, 2].set_ylabel('Nombre de paires')
        axes[1, 2].set_title('Distribution des coûts (Dijkstra)')
        axes[1, 2].legend()
        axes[1, 2].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    else:
        axes[1, 2].text(0.5, 0.5, 'Pas de données', ha='center', va='center')
        axes[1, 2].set_title('Distribution des coûts (Dijkstra)')

    plt.tight_layout()
    plt.savefig(f"histogrammes_partie3/hist_pondere_{densite}.png", dpi=150)
    plt.close()


# =============================================================================
# PARTIE 2 : ANALYSE DES GRAPHES NON VALUÉS
# =============================================================================

def analyser_neuf_configurations():
    """
    PARTIE 2 : Analyse complète des 9 configurations (3 densités × 3 portées).
    Graphes non valués - statistiques topologiques.
    """
    fichier_sortie = "resultats_partie2.txt"
    os.makedirs("histogrammes", exist_ok=True)

    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   PARTIE 2 : ANALYSE DES GRAPHES NON VALUÉS (9 configurations)\n")
        f.write("=" * 80 + "\n")

        for densite in ['low', 'avg', 'high']:
            for portee_nom in ['courte', 'moyenne', 'longue']:
                portee = PORTEES[portee_nom]
                config_name = f"{densite}_{portee_nom}"

                f.write(f"\n{'─'*80}\n")
                f.write(f"  DENSITÉ: {densite} | PORTÉE: {portee_nom} ({portee/1000:.0f} km)\n")
                f.write(f"{'─'*80}\n")

                # Obtenir le graphe (avec cache)
                G = obtenir_graphe(densite, portee_nom)

                # Analyses
                degres = analyser_degres(G)
                clustering = analyser_clustering(G)
                cliques = analyser_cliques(G)
                composantes = analyser_composantes(G)
                chemins = analyser_chemins(G)

                # Écriture des résultats
                f.write(f"\n   DEGRÉS:\n")
                f.write(f"     • Degré moyen: {degres['moyenne']:.2f}\n")
                f.write(f"     • Degré min/max: {degres['min']} / {degres['max']}\n")
                f.write(f"     • Distribution: {dict(sorted(degres['distribution'].items()))}\n")

                f.write(f"\n   CLUSTERING:\n")
                f.write(f"     • Clustering moyen: {clustering['moyenne']:.4f}\n")
                f.write(f"     • Clustering min/max: {clustering['min']:.4f} / {clustering['max']:.4f}\n")

                f.write(f"\n   CLIQUES:\n")
                f.write(f"     • Nombre total: {cliques['total']}\n")
                f.write(f"     • Par ordre: {cliques['par_ordre']}\n")
                f.write(f"     • Clique max: {cliques['max_ordre']} sommets\n")

                f.write(f"\n   COMPOSANTES CONNEXES:\n")
                f.write(f"     • Nombre: {composantes['nombre']}\n")
                f.write(f"     • Par ordre: {composantes['par_ordre']}\n")

                f.write(f"\n   PLUS COURTS CHEMINS:\n")
                if chemins:
                    f.write(f"     • Paires connectées: {chemins['paires']}\n")
                    f.write(f"     • Longueur moyenne: {chemins['moyenne']:.2f}\n")
                    f.write(f"     • Diamètre: {chemins['diametre']}\n")
                    f.write(f"     • Distribution: {dict(sorted(chemins['distribution'].items()))}\n")
                else:
                    f.write(f"     • Aucun chemin (composantes isolées)\n")

                # Histogrammes
                titre = f"Distributions - {densite} / {portee_nom} ({portee/1000:.0f} km)"
                generer_histogrammes_partie2(degres, clustering, cliques, chemins, config_name, titre)

        f.write("\n" + "=" * 80 + "\n")
        f.write("   PARTIE 2 TERMINÉE\n")
        f.write("=" * 80 + "\n")

    print(f"Résultats écrits dans: {fichier_sortie}")
    print(f"Histogrammes sauvegardés dans: histogrammes/")

    subprocess.run(['open', fichier_sortie])
    subprocess.run(['open', 'histogrammes'])


# =============================================================================
# PARTIE 3 : ANALYSE DES GRAPHES VALUÉS (coût = distance²)
# =============================================================================

def analyser_graphes_ponderes():
    """
    PARTIE 3 : Analyse des graphes valués pour portée 60km.
    Coût de chaque arête = distance² entre les deux satellites.
    Reprend toutes les analyses de la Partie 2 en version pondérée.
    """
    fichier_sortie = "resultats_partie3.txt"
    os.makedirs("histogrammes_partie3", exist_ok=True)

    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   PARTIE 3 : ANALYSE DES GRAPHES VALUÉS (portée 60km, coût = distance²)\n")
        f.write("=" * 80 + "\n")

        for densite in ['low', 'avg', 'high']:
            f.write(f"\n{'─'*80}\n")
            f.write(f"  DENSITÉ: {densite} | PORTÉE: 60 km | COÛT = distance²\n")
            f.write(f"{'─'*80}\n")

            # Obtenir le graphe pondéré (avec cache)
            G = obtenir_graphe(densite, 'longue', pondere=True)

            # Analyses communes (identiques Partie 2)
            degres = analyser_degres(G)
            clustering = analyser_clustering(G)
            cliques = analyser_cliques(G)
            composantes = analyser_composantes(G)
            chemins = analyser_chemins(G)

            # Analyses spécifiques graphes pondérés
            poids = analyser_poids(G)
            chemins_pond = analyser_chemins_ponderes(G)
            mst = analyser_mst(G)
            centralite = analyser_centralite(G)

            # Écriture des résultats
            f.write(f"\n   STATISTIQUES DE BASE:\n")
            f.write(f"     • Nœuds: {G.number_of_nodes()}\n")
            f.write(f"     • Arêtes: {G.number_of_edges()}\n")
            f.write(f"     • Densité: {nx.density(G):.4f}\n")

            if poids:
                f.write(f"\n   POIDS DES ARÊTES (distance²):\n")
                f.write(f"     • Poids total: {poids['total']:.2e}\n")
                f.write(f"     • Poids moyen: {poids['moyenne']:.2e}\n")
                f.write(f"     • Poids min/max: {poids['min']:.2e} / {poids['max']:.2e}\n")
                f.write(f"     • Écart-type: {poids['std']:.2e}\n")

            f.write(f"\n   DEGRÉS:\n")
            f.write(f"     • Degré moyen: {degres['moyenne']:.2f}\n")
            f.write(f"     • Degré min/max: {degres['min']} / {degres['max']}\n")
            f.write(f"     • Distribution: {dict(sorted(degres['distribution'].items()))}\n")

            f.write(f"\n   CLUSTERING:\n")
            f.write(f"     • Clustering moyen: {clustering['moyenne']:.4f}\n")
            f.write(f"     • Clustering min/max: {clustering['min']:.4f} / {clustering['max']:.4f}\n")

            f.write(f"\n   CLIQUES:\n")
            f.write(f"     • Nombre total: {cliques['total']}\n")
            f.write(f"     • Par ordre: {cliques['par_ordre']}\n")
            f.write(f"     • Clique max: {cliques['max_ordre']} sommets\n")

            f.write(f"\n   COMPOSANTES CONNEXES:\n")
            f.write(f"     • Nombre: {composantes['nombre']}\n")
            f.write(f"     • Par ordre: {composantes['par_ordre']}\n")

            f.write(f"\n   PLUS COURTS CHEMINS (en nombre de sauts):\n")
            if chemins:
                f.write(f"     • Paires connectées: {chemins['paires']}\n")
                f.write(f"     • Longueur moyenne: {chemins['moyenne']:.2f}\n")
                f.write(f"     • Diamètre (sauts): {chemins['diametre']}\n")
                f.write(f"     • Distribution: {dict(sorted(chemins['distribution'].items()))}\n")
            else:
                f.write(f"     • Graphe non connexe\n")

            f.write(f"\n   PLUS COURTS CHEMINS PONDÉRÉS (Dijkstra, coût = distance²):\n")
            if chemins_pond:
                f.write(f"     • Paires connectées: {chemins_pond['paires']}\n")
                f.write(f"     • Coût moyen: {chemins_pond['moyenne']:.2e}\n")
                f.write(f"     • Coût min/max: {chemins_pond['min']:.2e} / {chemins_pond['max']:.2e}\n")
                f.write(f"     • Diamètre pondéré: {chemins_pond['diametre_pondere']:.2e}\n")
                f.write(f"     • Écart-type: {chemins_pond['std']:.2e}\n")
            else:
                f.write(f"     • Graphe non connexe\n")

            f.write(f"\n   ARBRE COUVRANT MINIMUM (Kruskal):\n")
            if mst['est_arbre']:
                f.write(f"     • Poids total MST: {mst['poids_total']:.2e}\n")
                f.write(f"     • Nb arêtes MST: {mst['nb_aretes']}\n")
                f.write(f"     • Arête min/max: {mst['arete_min']:.2e} / {mst['arete_max']:.2e}\n")
                f.write(f"     • Arête moyenne: {mst['arete_moyenne']:.2e}\n")
            else:
                f.write(f"     • Poids forêt couvrante: {mst['poids_total']:.2e}\n")
                f.write(f"     • Nb arêtes: {mst['nb_aretes']}\n")

            f.write(f"\n   CENTRALITÉ DE PROXIMITÉ PONDÉRÉE:\n")
            if centralite:
                f.write(f"     • Top 5 nœuds centraux: {centralite['top_5']}\n")
            else:
                f.write(f"     • Non calculable (graphe non connexe)\n")

            # Histogrammes
            generer_histogrammes_partie3(degres, clustering, cliques, chemins, poids, chemins_pond, densite)

        f.write("\n" + "=" * 80 + "\n")
        f.write("   PARTIE 3 TERMINÉE\n")
        f.write("=" * 80 + "\n")

    print(f"Résultats écrits dans: {fichier_sortie}")
    print(f"Histogrammes sauvegardés dans: histogrammes_partie3/")

    subprocess.run(['open', fichier_sortie])
    subprocess.run(['open', 'histogrammes_partie3'])

# =============================================================================
# VISUALISATION INTERACTIVE
# =============================================================================

def visualiser_interactif_2d():
    """Interface interactive 2D pour modifier densité et portée en temps réel."""
    from matplotlib.widgets import RadioButtons

    etat = {'densite': 'avg', 'portee': 'moyenne'}

    # Pré-charger toutes les données
    for d in DATA_FILES:
        charger_donnees(d)

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(left=0.2)

    ax_densite = plt.axes([0.02, 0.7, 0.10, 0.15])
    ax_portee = plt.axes([0.02, 0.4, 0.10, 0.15])

    radio_densite = RadioButtons(ax_densite, ('low', 'avg', 'high'), active=1)
    radio_portee = RadioButtons(ax_portee, ('courte', 'moyenne', 'longue'), active=1)

    ax_densite.set_title('Densité', fontsize=10, fontweight='bold')
    ax_portee.set_title('Portée', fontsize=10, fontweight='bold')

    def update_graph():
        ax.clear()
        G = obtenir_graphe(etat['densite'], etat['portee'])
        portee = PORTEES[etat['portee']]

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        degres = dict(G.degree())
        couleurs = [degres[n] for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=couleurs, cmap=plt.cm.viridis,
                               node_size=150, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=6, font_color='white', ax=ax)

        titre = f"Densité: {etat['densite']} | Portée: {etat['portee']} ({portee/1000:.0f} km)\n"
        titre += f"Nœuds: {G.number_of_nodes()} | Arêtes: {G.number_of_edges()} | Clustering: {nx.average_clustering(G):.3f}"
        ax.set_title(titre, fontsize=12, fontweight='bold')
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_densite(label):
        etat['densite'] = label
        update_graph()

    def on_portee(label):
        etat['portee'] = label
        update_graph()

    radio_densite.on_clicked(on_densite)
    radio_portee.on_clicked(on_portee)

    update_graph()
    plt.show()


def visualiser_interactif_3d():
    """Interface interactive 3D pour modifier densité et portée en temps réel."""
    from matplotlib.widgets import RadioButtons

    etat = {'densite': 'avg', 'portee': 'moyenne'}

    # Pré-charger toutes les données
    for d in DATA_FILES:
        charger_donnees(d)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax_densite = plt.axes([0.02, 0.7, 0.10, 0.15])
    ax_portee = plt.axes([0.02, 0.4, 0.10, 0.15])

    radio_densite = RadioButtons(ax_densite, ('low', 'avg', 'high'), active=1)
    radio_portee = RadioButtons(ax_portee, ('courte', 'moyenne', 'longue'), active=1)

    ax_densite.set_title('Densité', fontsize=10, fontweight='bold')
    ax_portee.set_title('Portée', fontsize=10, fontweight='bold')

    def update_graph():
        ax.clear()
        G = obtenir_graphe(etat['densite'], etat['portee'])
        satellites = charger_donnees(etat['densite'])
        portee = PORTEES[etat['portee']]

        positions = {row['sat_id']: (row['x'], row['y'], row['z'])
                     for _, row in satellites.iterrows()}

        for u, v in G.edges():
            x = [positions[u][0], positions[v][0]]
            y = [positions[u][1], positions[v][1]]
            z = [positions[u][2], positions[v][2]]
            ax.plot(x, y, z, 'gray', alpha=0.3, linewidth=0.5)

        degres = dict(G.degree())
        xs = [positions[n][0] for n in G.nodes()]
        ys = [positions[n][1] for n in G.nodes()]
        zs = [positions[n][2] for n in G.nodes()]
        colors = [degres[n] for n in G.nodes()]

        ax.scatter(xs, ys, zs, c=colors, cmap='viridis',
                   s=50, alpha=0.8, edgecolors='black', linewidths=0.5)

        titre = f"Densité: {etat['densite']} | Portée: {etat['portee']} ({portee/1000:.0f} km)\n"
        titre += f"Nœuds: {G.number_of_nodes()} | Arêtes: {G.number_of_edges()} | Clustering: {nx.average_clustering(G):.3f}"
        ax.set_title(titre, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        fig.canvas.draw_idle()

    def on_densite(label):
        etat['densite'] = label
        update_graph()

    def on_portee(label):
        etat['portee'] = label
        update_graph()

    radio_densite.on_clicked(on_densite)
    radio_portee.on_clicked(on_portee)

    update_graph()
    plt.show()


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse d'un essaim de nanosatellites en orbite lunaire",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i2d', '--interactif2d',
        action='store_true',
        help="Mode interactif 2D: modifier densité/portée avec des boutons"
    )

    parser.add_argument(
        '-i3d', '--interactif3d',
        action='store_true',
        help="Mode interactif 3D: modifier densité/portée avec des boutons + rotation"
    )

    parser.add_argument(
        '-a', '--analyse',
        action='store_true',
        help="PARTIE 2: Analyse des 9 configurations (graphes non valués)"
    )

    parser.add_argument(
        '-p3', '--partie3',
        action='store_true',
        help="PARTIE 3: Analyse des graphes valués (portée 60km, coût = distance²)"
    )

    args = parser.parse_args()

    if args.analyse:
        analyser_neuf_configurations()
    elif args.partie3:
        analyser_graphes_ponderes()
    elif args.interactif2d:
        print("→ Lancement du mode interactif 2D...")
        visualiser_interactif_2d()
    elif args.interactif3d:
        print("→ Lancement du mode interactif 3D...")
        visualiser_interactif_3d()
    else:
        print("Utilisation: python3 analyse_graphe.py [OPTION]")
        print("Options disponibles:")
        print("  -a, --analyse      PARTIE 2: Analyse des 9 configurations")
        print("  -p3, --partie3     PARTIE 3: Graphes valués (coût = distance²)")
        print("  -i2d, --interactif2d  Mode interactif 2D")
        print("  -i3d, --interactif3d  Mode interactif 3D")
        print("\nUtilisez --help pour plus de détails.")
