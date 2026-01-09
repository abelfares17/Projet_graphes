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
- Google Antigravity IDE

## Auteur
2025-2026

---

## Utilisation

### Afficher l'aide
```bash
python3 analyse_graphe.py --help
```

### Options disponibles

| Option | Valeurs | Description |
|--------|---------|-------------|
| `-d`, `--densite` | `low`, `avg`, `high` | Densité de l'essaim (défaut: `avg`) |
| `-p`, `--portee` | `courte`, `moyenne`, `longue` | Portée de communication (défaut: `moyenne`) |
| `-v`, `--visu` | `2d`, `3d`, `both`, `none` | Type de visualisation (défaut: `none`) |
| `-i2d`, `--interactif2d` | - | Mode interactif 2D avec boutons |
| `-i3d`, `--interactif3d` | - | Mode interactif 3D avec boutons + rotation |

### Portées de communication
- `courte` = 30 km
- `moyenne` = 60 km
- `longue` = 100 km

### Exemples

**Statistiques seulement (défaut) :**
```bash
python3 analyse_graphe.py
```

**Changer la densité et la portée :**
```bash
python3 analyse_graphe.py -d low -p courte    # Densité faible, 30 km
python3 analyse_graphe.py -d high -p longue   # Densité forte, 100 km
```

**Afficher le graphe 2D :**
```bash
python3 analyse_graphe.py -v 2d
```

**Afficher le graphe 3D (rotation avec la souris) :**
```bash
python3 analyse_graphe.py -v 3d
```

**Combiner toutes les options :**
```bash
python3 analyse_graphe.py -d high -p longue -v both
```

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