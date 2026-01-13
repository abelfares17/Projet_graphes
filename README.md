# Projet_graphes
Projet UE Théorie des Graphes - INPT N7 (2SN-LMB)

## Description
Analyse d'un essaim de nanosatellites en orbite lunaire pour une application d'interférométrie.

## Objectifs
- **Partie 1** : Modélisation sous forme de graphe (3 densités × 3 portées)
- **Partie 2** : Étude des graphes non valués (degrés, clustering, cliques, composantes connexes, plus courts chemins)
- **Partie 3** : Étude des graphes valués (portée 60km, coût = distance²)

## Données
- `topology_low.csv` - Densité faible
- `topology_avg.csv` - Densité moyenne
- `topology_high.csv` - Densité forte

## Technologies
- Python
- NetworkX
- Matplotlib

## Auteur
2025-2026

---

## Utilisation

### Afficher l'aide
```bash
python3 analyse_graphe.py --help
```

### Options disponibles

| Option | Description |
|--------|-------------|
| `-a`, `--analyse` | PARTIE 2: Graphes non valués (9 configurations) |
| `-p3`, `--partie3` | PARTIE 3: Graphes valués (coût = distance²) |
| `-i2d`, `--interactif2d` | Mode interactif 2D avec boutons |
| `-i3d`, `--interactif3d` | Mode interactif 3D avec boutons + rotation |

---

### PARTIE 2 : Graphes non valués (9 configurations)

Analyse topologique pour les 9 combinaisons (3 densités × 3 portées) :
```bash
python3 analyse_graphe.py -a
```

**Résultats :**
- Fichier texte `resultats_partie2.txt` (s'ouvre automatiquement)
- Histogrammes dans le dossier `histogrammes/` (s'ouvre automatiquement)

**Statistiques calculées :**
- Degrés : moyenne, min/max, distribution
- Clustering : moyen, min/max
- Cliques : nombre total, distribution par ordre, clique maximale
- Composantes connexes : nombre, distribution par ordre
- Plus courts chemins : longueur moyenne, diamètre, distribution

---

### PARTIE 3 : Graphes valués (coût = distance²)

Analyse des graphes pondérés pour portée 60km avec coût = distance² :
```bash
python3 analyse_graphe.py -p3
```

**Statistiques calculées :**
- Poids des arêtes : total, moyen, min/max
- Plus courts chemins pondérés (Dijkstra) : coût moyen, diamètre pondéré
- Arbre couvrant minimum (Kruskal) : poids total, nombre d'arêtes
- Centralité de proximité pondérée

---

### Modes interactifs

Interface avec boutons pour modifier la densité et la portée en temps réel :

**Mode interactif 2D :**
```bash
python3 analyse_graphe.py -i2d
```

**Mode interactif 3D (avec rotation à la souris) :**
```bash
python3 analyse_graphe.py -i3d
```

Le graphe se met à jour automatiquement quand tu cliques sur une option.

### Navigation dans les graphes
- **2D** : Utilisez la toolbar en bas (zoom, pan)
- **3D** : Cliquez et glissez pour tourner la vue