# GNN Fluid Simulation

> Implémentation de réseaux de neurones graphiques (GNN) pour l'approximation des simulations d'écoulements de fluides incompressibles — Dataset **The WELL** de Polymathic AI.

**Auteur :** AHNANI Ali — Master 1 Informatique & Mathématiques Appliquées — 2025/2026

---

## Description

Ce projet explore l'utilisation des **Graph Neural Networks (GNN)** comme modèles de substitution (*surrogate models*) pour la simulation d'écoulements de fluides incompressibles. Trois architectures sont implémentées, entraînées et comparées sur l'instabilité de **Kelvin-Helmholtz** (écoulement de cisaillement 2D périodique).

### Architectures comparées

| Modèle | Particularité | Paramètres | NMSE moyen |
|--------|--------------|-----------|------------|
| **MeshGraphNet** | Baseline Encode-Process-Decode | 252 674 | 0.0007 |
| **EGNN** | Équivariance E(3) intégrée | 32 362 | 0.0002 |
| **GMN** | Projection de Helmholtz (div=0) | 265 155 | 0.0009 |

---

## Structure du projet

```
gnn-fluid-simulation/
├── GNN_Fluid_Simulation.ipynb     # Notebook principal
├── gnn_fluides.pdf                # Rapport complet
├── requirements.txt               # Dépendances Python
│
├── src/
│   ├── models/
│   │   ├── meshgraphnet.py        # GNS + MeshGraphNet
│   │   ├── e3gnn.py               # EGNN équivariant E(n)
│   │   └── gmn.py                 # GMN + projection Helmholtz-Hodge + GRU
│   ├── data/
│   │   ├── dataset.py             # ShearFlowDataset, WELLShearFlowModule
│   │   └── preprocessing.py       # Normalizer, build_grid_edges, bruit
│   └── utils/
│       ├── metrics.py             # NMSE, t*, spectre E(k), SSIM
│       └── visualization.py       # Figures et animations
│
├── data/                          # Données HDF5 (The WELL)
├── experiments/checkpoints/       # Sauvegardes des modèles
└── results/figures/               # Figures exportées
```

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/Ahnani1Ali/gnn-fluid-simulation.git
cd gnn-fluid-simulation

# Installer les dépendances
pip install -r requirements.txt
```

> **Note :** PyTorch Geometric nécessite une installation spécifique selon votre version de CUDA. Consultez [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Données

Les données proviennent de **The WELL** (Polymathic AI) — dataset open access de simulations physiques haute qualité.

- Lien : [polymathic-ai.org/the_well](https://polymathic-ai.org/the_well/)
- Dataset utilisé : `shear_flow` — instabilité de Kelvin-Helmholtz 2D
- Format : HDF5, résolution 128×128, ~1000 trajectoires de longueur 200

Placez les fichiers HDF5 dans le dossier `data/`.

---

## Utilisation

### Lancer le notebook

```bash
jupyter notebook GNN_Fluid_Simulation.ipynb
```

### Entraîner un modèle

```python
from src.models.meshgraphnet import MeshGraphNet
from src.models.e3gnn import EGNN
from src.models.gmn import GMN
```

### Générer des données synthétiques (sans accès à The WELL)

```python
from src.data.dataset import generate_synthetic_shear_flow
data = generate_synthetic_shear_flow(n_traj=10, T=50, nx=64, ny=64)
```

---

## Résultats

Les trois modèles atteignent un **temps de validité t\* = 31** pas de temps avec un seuil NMSE de 15%.

L'**EGNN** se distingue avec :
- Meilleur NMSE moyen : **0.0002**
- Rollout le plus rapide : **0.26 s**
- Modèle le plus compact : **32 362 paramètres** (8x moins que les autres)

---

## Références

- Sanchez-Gonzalez et al. (2020) — *Learning to simulate complex physics with graph networks* — [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)
- Pfaff et al. (2021) — *Learning mesh-based simulation with graph networks* — [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- Satorras et al. (2021) — *E(n) equivariant graph neural networks* — [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
- Ohana et al. (2024) — *The Well* — [arXiv:2412.00568](https://arxiv.org/abs/2412.00568)

---

## Licence

Projet académique — Master 1 — 2025/2026
