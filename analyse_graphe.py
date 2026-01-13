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
from itertools import combinations

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
    'courte': 30_000,    # 30 km
    'moyenne': 60_000,   # 60 km
    'longue': 100_000    # 100 km
}

# =============================================================================
# FONCTIONS
# =============================================================================

def charger_donnees(fichier):
    """Charge les positions des satellites depuis un fichier CSV."""
    df = pd.read_csv(fichier)
    print(f"✓ {len(df)} satellites chargés depuis {fichier}")
    return df


def calculer_distance(p1, p2):
    """Calcule la distance euclidienne entre deux points 3D."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def construire_graphe(satellites, portee):
    """
    Construit un graphe non pondéré où une arête existe 
    si deux satellites sont à portée de communication.
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
    (Pour la partie 3 du projet)
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
                # Coût = distance²
                G.add_edge(int(sat_i[0]), int(sat_j[0]), weight=distance**2)
    
    return G


def afficher_statistiques(G, nom):
    """Affiche les statistiques principales du graphe."""
    print(f"\n{'='*60}")
    print(f"STATISTIQUES - {nom}")
    print(f"{'='*60}")
    print(f"  • Nombre de nœuds (satellites) : {G.number_of_nodes()}")
    print(f"  • Nombre d'arêtes (liens)      : {G.number_of_edges()}")
    print(f"  • Densité du graphe            : {nx.density(G):.4f}")
    
    if G.number_of_edges() > 0:
        degres = [d for n, d in G.degree()]
        print(f"  • Degré minimum                : {min(degres)}")
        print(f"  • Degré maximum                : {max(degres)}")
        print(f"  • Degré moyen                  : {np.mean(degres):.2f}")
        
        # Composantes connexes
        nb_composantes = nx.number_connected_components(G)
        print(f"  • Composantes connexes         : {nb_composantes}")
        
        if nb_composantes == 1:
            print(f"  • Diamètre du graphe           : {nx.diameter(G)}")
            print(f"  • Rayon du graphe              : {nx.radius(G)}")
        
        # Coefficient de clustering
        print(f"  • Clustering moyen             : {nx.average_clustering(G):.4f}")


def analyser_neuf_configurations():
    """Analyse complète des 9 configurations (3 densités × 3 portées)."""
    from collections import Counter
    
    print("\n" + "="*80)
    print("   ANALYSE COMPLÈTE DES 9 CONFIGURATIONS")
    print("="*80)
    
    for densite in ['low', 'avg', 'high']:
        satellites = pd.read_csv(DATA_FILES[densite])
        
        for portee_nom, portee in PORTEES.items():
            print(f"\n{'─'*80}")
            print(f"  DENSITÉ: {densite} | PORTÉE: {portee_nom} ({portee/1000:.0f} km)")
            print(f"{'─'*80}")
            
            G = construire_graphe(satellites, portee)
            
            # --- DEGRÉS ---
            degres = [d for n, d in G.degree()]
            print(f"\n   DEGRÉS:")
            print(f"     • Degré moyen: {np.mean(degres):.2f}")
            print(f"     • Degré min/max: {min(degres)} / {max(degres)}")
            deg_dist = Counter(degres)
            print(f"     • Distribution: {dict(sorted(deg_dist.items()))}")
            
            # --- CLUSTERING ---
            clustering = list(nx.clustering(G).values())
            print(f"\n  🔗 CLUSTERING:")
            print(f"     • Clustering moyen: {np.mean(clustering):.4f}")
            print(f"     • Clustering min/max: {min(clustering):.4f} / {max(clustering):.4f}")
            
            # --- CLIQUES ---
            cliques = list(nx.find_cliques(G))
            ordres_cliques = Counter([len(c) for c in cliques])
            print(f"\n   CLIQUES:")
            print(f"     • Nombre total: {len(cliques)}")
            print(f"     • Par ordre: {dict(sorted(ordres_cliques.items()))}")
            print(f"     • Clique max: {max(ordres_cliques.keys())} sommets")
            
            # --- COMPOSANTES CONNEXES ---
            composantes = list(nx.connected_components(G))
            ordres_comp = Counter([len(c) for c in composantes])
            print(f"\n   COMPOSANTES CONNEXES:")
            print(f"     • Nombre: {len(composantes)}")
            print(f"     • Par ordre: {dict(sorted(ordres_comp.items()))}")
            
            # --- PLUS COURTS CHEMINS ---
            print(f"\n   PLUS COURTS CHEMINS:")
            if nx.is_connected(G):
                all_paths = dict(nx.all_pairs_shortest_path_length(G))
                longueurs = [l for s in all_paths for t, l in all_paths[s].items() if s < t]
                dist_chemins = Counter(longueurs)
                print(f"     • Paires connectées: {len(longueurs)}")
                print(f"     • Longueur moyenne: {np.mean(longueurs):.2f}")
                print(f"     • Diamètre: {max(longueurs)}")
                print(f"     • Distribution: {dict(sorted(dist_chemins.items()))}")
            else:
                print(f"     • Graphe non connexe")
                for i, comp in enumerate(sorted(composantes, key=len, reverse=True)):
                    if len(comp) > 1:
                        subG = G.subgraph(comp)
                        paths = dict(nx.all_pairs_shortest_path_length(subG))
                        lens = [l for s in paths for t, l in paths[s].items() if s < t]
                        print(f"     • Comp. {i+1} ({len(comp)} nœuds): moy={np.mean(lens):.2f}, max={max(lens)}")
    
    print("\n" + "="*80)
    print("   ANALYSE TERMINÉE")
    print("="*80)


def visualiser_graphe_2d(G, titre):
    """Visualise le graphe en 2D (zoom/pan avec la toolbar en bas)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Couleurs selon le degré
    degres = dict(G.degree())
    couleurs = [degres[n] for n in G.nodes()]
    
    # Dessiner le graphe
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=couleurs, cmap=plt.cm.viridis,
                           node_size=150, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color='white', ax=ax)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=min(couleurs), vmax=max(couleurs)))
    plt.colorbar(sm, ax=ax, label='Degré du nœud', shrink=0.8)
    
    ax.set_title(titre, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualiser_graphe_3d(G, satellites, titre):
    """Visualise le graphe en 3D avec les vraies positions des satellites."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Récupérer les positions
    positions = {row['sat_id']: (row['x'], row['y'], row['z']) 
                 for _, row in satellites.iterrows()}
    
    # Dessiner les arêtes
    for u, v in G.edges():
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        z = [positions[u][2], positions[v][2]]
        ax.plot(x, y, z, 'gray', alpha=0.3, linewidth=0.5)
    
    # Dessiner les nœuds
    degres = dict(G.degree())
    xs = [positions[n][0] for n in G.nodes()]
    ys = [positions[n][1] for n in G.nodes()]
    zs = [positions[n][2] for n in G.nodes()]
    colors = [degres[n] for n in G.nodes()]
    
    scatter = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', 
                         s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    plt.colorbar(scatter, label='Degré du nœud', shrink=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(titre, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# VISUALISATION INTERACTIVE
# =============================================================================

def visualiser_interactif_2d():
    """Interface interactive 2D pour modifier densité et portée en temps réel."""
    from matplotlib.widgets import RadioButtons
    
    etat = {'densite': 'avg', 'portee': 'moyenne'}
    all_satellites = {d: pd.read_csv(DATA_FILES[d]) for d in DATA_FILES}
    
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
        satellites = all_satellites[etat['densite']]
        portee = PORTEES[etat['portee']]
        G = construire_graphe(satellites, portee)
        
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
    all_satellites = {d: pd.read_csv(DATA_FILES[d]) for d in DATA_FILES}
    
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
        satellites = all_satellites[etat['densite']]
        portee = PORTEES[etat['portee']]
        G = construire_graphe(satellites, portee)
        
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
        '-d', '--densite',
        choices=['low', 'avg', 'high'],
        default='avg',
        help="Densité de l'essaim:\n  low  = faible\n  avg  = moyenne (défaut)\n  high = forte"
    )
    
    parser.add_argument(
        '-p', '--portee',
        choices=['courte', 'moyenne', 'longue'],
        default='moyenne',
        help="Portée de communication:\n  courte  = 30 km\n  moyenne = 60 km (défaut)\n  longue  = 100 km"
    )
    
    parser.add_argument(
        '-v', '--visu',
        choices=['2d', '3d', 'both', 'none'],
        default='none',
        help="Type de visualisation:\n  2d   = graphe 2D\n  3d   = graphe 3D\n  both = les deux\n  none = stats seulement (défaut)"
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
        help="Analyse complète des 9 configurations (3 densités × 3 portées)"
    )
    
    args = parser.parse_args()
    
    # Modes spéciaux
    if args.analyse:
        analyser_neuf_configurations()
    elif args.interactif2d:
        print("→ Lancement du mode interactif 2D...")
        visualiser_interactif_2d()
    elif args.interactif3d:
        print("→ Lancement du mode interactif 3D...")
        visualiser_interactif_3d()
    else:
        print("=" * 60)
        print("   PROJET GRAPHES - Essaim de Nanosatellites Lunaires")
        print("=" * 60)
        
        # Charger les données
        satellites = charger_donnees(DATA_FILES[args.densite])
        
        # Construire le graphe
        portee = PORTEES[args.portee]
        print(f"\n→ Construction du graphe (portée = {portee/1000:.0f} km)...")
        G = construire_graphe(satellites, portee)
        
        # Afficher les statistiques
        nom_config = f"Densité {args.densite} - Portée {args.portee} ({portee/1000:.0f} km)"
        afficher_statistiques(G, nom_config)
        
        # Visualiser selon le choix
        if args.visu in ['2d', 'both']:
            print("\n→ Affichage du graphe 2D...")
            visualiser_graphe_2d(G, f"Graphe des satellites\n{nom_config}")
        
        if args.visu in ['3d', 'both']:
            print("\n→ Affichage du graphe 3D...")
            visualiser_graphe_3d(G, satellites, f"Graphe 3D des satellites\n{nom_config}")
        
        print("\n✓ Analyse terminée !")
