"""
preprocessing.py
================
Utilitaires de pré-traitement des données de simulation de fluides.

Ce module gère :
  - La normalisation des champs physiques (z-score, min-max)
  - La construction de graphes à partir de grilles régulières
  - La construction de graphes à rayon variable
  - L'ajout de bruit de marche aléatoire (robustesse au rollout)
  - Les conversions entre représentations grille et graphe
"""

from __future__ import annotations
from typing import Tuple, Optional, Literal, Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Normalisation z-score avec statistiques calculées sur l'ensemble train.

    La normalisation est cruciale pour l'entraînement des GNN :
      - Elle équilibre les gradients entre les différentes quantités
      - Elle permet au modèle de se concentrer sur les variations relatives
      - Elle améliore la convergence numérique

    Formule : x_norm = (x - μ) / (σ + ε)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[float] = None
        self.std:  Optional[float] = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        """Calcule μ et σ sur les données d'entraînement."""
        self.mean = float(data.mean())
        self.std  = float(data.std()) + self.eps
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Applique la normalisation."""
        assert self.mean is not None, "Appeler fit() avant transform()"
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Dénormalise les données."""
        assert self.mean is not None, "Appeler fit() avant inverse_transform()"
        return data * self.std + self.mean

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def transform_tensor(self, t: Tensor) -> Tensor:
        """Version PyTorch de transform."""
        return (t - self.mean) / self.std

    def inverse_transform_tensor(self, t: Tensor) -> Tensor:
        return t * self.std + self.mean

    def __repr__(self) -> str:
        return f"Normalizer(mean={self.mean:.4f}, std={self.std:.4f})"


class PerFieldNormalizer:
    """
    Normalisation indépendante pour chaque champ physique (vx, vy, p).
    """

    def __init__(self, fields: List[str] = ("vx", "vy", "p")):
        self.fields = fields
        self.normalizers: Dict[str, Normalizer] = {
            f: Normalizer() for f in fields
        }

    def fit(self, data: Dict[str, np.ndarray]) -> "PerFieldNormalizer":
        for f in self.fields:
            self.normalizers[f].fit(data[f])
        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {f: self.normalizers[f].transform(data[f]) for f in self.fields}

    def inverse_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {f: self.normalizers[f].inverse_transform(data[f]) for f in self.fields}

    def get(self, field: str) -> Normalizer:
        return self.normalizers[field]


# ---------------------------------------------------------------------------
# Construction de graphes
# ---------------------------------------------------------------------------

def build_grid_edges(
    Nx: int,
    Ny: int,
    connectivity: Literal["4-connected", "8-connected"] = "4-connected",
    periodic: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Construit les arêtes d'une grille 2D régulière.

    La grille a Nx × Ny nœuds. L'index du nœud (i, j) est i*Ny + j.

    Encodage des attributs d'arête :
        e_ij = (Δx/Nx, Δy/Ny, ‖(Δx, Δy)‖/max_dist)
    Ces valeurs normalisées permettent au modèle de raisonner sur les
    distances relatives indépendamment de la résolution.

    Args:
        Nx, Ny       : Résolution de la grille.
        connectivity : Voisinage à 4 ou 8 connexions.
        periodic     : Conditions aux limites périodiques.

    Returns:
        edge_index (2, E), edge_attr (E, 3)
    """
    if connectivity == "4-connected":
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif connectivity == "8-connected":
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError(f"Connectivity inconnue : {connectivity}")

    max_dist = np.sqrt(2) / max(Nx, Ny)  # Normalisation

    src_list, dst_list, attr_list = [], [], []

    for i in range(Nx):
        for j in range(Ny):
            src = i * Ny + j
            for di, dj in offsets:
                ni_raw, nj_raw = i + di, j + dj
                # Conditions périodiques ou bornées
                if periodic:
                    ni = ni_raw % Nx
                    nj = nj_raw % Ny
                else:
                    if not (0 <= ni_raw < Nx and 0 <= nj_raw < Ny):
                        continue
                    ni, nj = ni_raw, nj_raw

                dst = ni * Ny + nj
                dx = di / Nx
                dy = dj / Ny
                dist = np.sqrt(dx**2 + dy**2) / (max_dist + 1e-8)

                src_list.append(src)
                dst_list.append(dst)
                attr_list.append([dx, dy, dist])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float32)
    return edge_index, edge_attr


def build_radius_graph(
    positions: np.ndarray,
    radius: float,
    max_neighbors: int = 32,
    periodic: bool = True,
    domain_size: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Construit un graphe de voisinage par rayon.

    Pour des maillages non-structurés ou pour capturer des interactions
    à plus longue portée que le voisinage immédiat.

    Args:
        positions    : (N, 2) coordonnées des nœuds dans [0, domain_size]².
        radius       : Rayon de voisinage.
        max_neighbors: Nombre maximal de voisins par nœud.
        periodic     : Conditions aux limites périodiques.
        domain_size  : Taille du domaine.

    Returns:
        edge_index (2, E), edge_attr (E, 3)
    """
    N = len(positions)
    src_list, dst_list, attr_list = [], [], []

    for i in range(N):
        dists = positions - positions[i]  # (N, 2)

        # Correction périodique
        if periodic:
            dists = dists - domain_size * np.round(dists / domain_size)

        d_norm = np.linalg.norm(dists, axis=1)  # (N,)
        mask = (d_norm < radius) & (d_norm > 0)
        neighbors = np.where(mask)[0]

        # Limiter le nombre de voisins (prendre les plus proches)
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[np.argsort(d_norm[neighbors])[:max_neighbors]]

        for j in neighbors:
            src_list.append(i)
            dst_list.append(j)
            dx, dy = dists[j, 0], dists[j, 1]
            attr_list.append([dx, dy, d_norm[j]])

    if not src_list:
        # Aucune arête trouvée : retourner un graphe vide
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))

    return (
        torch.tensor([src_list, dst_list], dtype=torch.long),
        torch.tensor(attr_list, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Bruit de marche aléatoire
# ---------------------------------------------------------------------------

def add_random_walk_noise(
    x: np.ndarray,
    std: float = 3e-4,
    steps: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Ajoute un bruit de marche aléatoire aux features d'entrée.

    Motivation (Sanchez-Gonzalez et al., 2020) :
    Lors du rollout, le modèle reçoit ses propres prédictions (bruitées)
    comme entrées. Si on entraîne uniquement sur des données propres,
    le modèle n'est pas robuste à ces erreurs accumulées.

    La marche aléatoire simule cette accumulation d'erreurs :
        ε_0 ~ N(0, σ²)
        ε_k = ε_{k-1} + N(0, σ²)
    L'ajout de k pas multiplie environ la variance par k.

    Args:
        x     : (..., d) features à bruiter.
        std   : Écart-type du bruit par pas.
        steps : Nombre de pas de marche.
        rng   : Générateur aléatoire (reproductibilité).

    Returns:
        x_noisy : même forme que x.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = np.zeros_like(x, dtype=np.float32)
    for _ in range(steps):
        noise += rng.standard_normal(x.shape).astype(np.float32) * std

    return x.astype(np.float32) + noise


# ---------------------------------------------------------------------------
# Conversion grille ↔ graphe
# ---------------------------------------------------------------------------

def fields_to_graph(
    vx: np.ndarray,
    vy: np.ndarray,
    p: np.ndarray,
    edge_index: Tensor,
    edge_attr: Tensor,
    norm_vx: Optional[Normalizer] = None,
    norm_vy: Optional[Normalizer] = None,
    norm_p:  Optional[Normalizer] = None,
    noise_std: float = 0.0,
    noise_steps: int = 0,
) -> Data:
    """
    Convertit des champs physiques (grille) en un graphe PyTorch Geometric.

    Args:
        vx, vy : (Nx, Ny) champs de vitesse.
        p      : (Nx, Ny) champ de pression.
        edge_index, edge_attr : structure du graphe (pré-calculée).
        norm_*  : Normalisateurs (si None, pas de normalisation).
        noise_* : Paramètres de bruit (entraînement seulement).

    Returns:
        torch_geometric.data.Data
    """
    # Normalisation
    vx_n = norm_vx.transform(vx).flatten() if norm_vx else vx.flatten()
    vy_n = norm_vy.transform(vy).flatten() if norm_vy else vy.flatten()
    p_n  = norm_p.transform(p).flatten()   if norm_p  else p.flatten()

    x = np.stack([vx_n, vy_n, p_n], axis=-1).astype(np.float32)

    # Bruit de marche aléatoire (entraînement)
    if noise_std > 0 and noise_steps > 0:
        x = add_random_walk_noise(x, std=noise_std, steps=noise_steps)

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def graph_to_fields(
    node_features: Tensor,
    Nx: int,
    Ny: int,
    norm_vx: Optional[Normalizer] = None,
    norm_vy: Optional[Normalizer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconvertit les features de nœuds en champs physiques 2D.

    Args:
        node_features : (N, 2) ou (N, ≥2) prédictions sur les nœuds.
        Nx, Ny        : Résolution de la grille.
        norm_vx, norm_vy : Dénormalisation.

    Returns:
        (vx, vy) champs dénormalisés de forme (Nx, Ny).
    """
    nf = node_features.cpu().numpy()
    vx = nf[:, 0].reshape(Nx, Ny)
    vy = nf[:, 1].reshape(Nx, Ny)

    if norm_vx is not None:
        vx = norm_vx.inverse_transform(vx)
    if norm_vy is not None:
        vy = norm_vy.inverse_transform(vy)

    return vx, vy


# ---------------------------------------------------------------------------
# Vérifications et statistiques
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    data: Dict[str, np.ndarray],
    fields: List[str] = ("velocity_x", "velocity_y", "pressure"),
) -> None:
    """Affiche les statistiques descriptives du dataset."""
    print("─" * 60)
    print("STATISTIQUES DU DATASET")
    print("─" * 60)
    n_traj = data[fields[0]].shape[0]
    T      = data[fields[0]].shape[1]
    Nx, Ny = data[fields[0]].shape[2], data[fields[0]].shape[3]
    print(f"Trajectoires : {n_traj}  |  T : {T}  |  Grille : {Nx}×{Ny}")
    print(f"Nœuds totaux : {Nx*Ny:,}  |  Paires (traj,t) : {n_traj*(T-1):,}")
    print()

    for f in fields:
        arr = data[f]
        print(f"  {f:15s} : "
              f"μ={arr.mean():.4f}  "
              f"σ={arr.std():.4f}  "
              f"min={arr.min():.4f}  "
              f"max={arr.max():.4f}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Tests unitaires
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Tests unitaires — preprocessing.py")
    print("=" * 60)

    # ── Test Normalizer ──────────────────────────────────────────────
    print("\n[1] Normalizer")
    data = np.random.randn(100, 64, 64).astype(np.float32)
    norm = Normalizer()
    data_n = norm.fit_transform(data)
    assert abs(data_n.mean()) < 0.01
    assert abs(data_n.std() - 1.0) < 0.01
    data_rec = norm.inverse_transform(data_n)
    assert np.allclose(data, data_rec, atol=1e-5)
    print(f"    ✅ {norm}")

    # ── Test build_grid_edges ────────────────────────────────────────
    print("\n[2] build_grid_edges (4-connected)")
    ei, ea = build_grid_edges(8, 8, "4-connected", periodic=True)
    assert ei.shape == (2, 8*8*4), f"{ei.shape}"
    assert ea.shape == (8*8*4, 3)
    print(f"    ✅ edge_index : {ei.shape}  edge_attr : {ea.shape}")

    print("\n[3] build_grid_edges (8-connected)")
    ei8, ea8 = build_grid_edges(8, 8, "8-connected")
    assert ei8.shape == (2, 8*8*8)
    print(f"    ✅ edge_index : {ei8.shape}")

    # ── Test bruit ───────────────────────────────────────────────────
    print("\n[4] Random walk noise")
    x = np.zeros((256, 3), dtype=np.float32)
    x_noisy = add_random_walk_noise(x, std=1e-3, steps=3)
    std_obs = x_noisy.std()
    print(f"    σ observé : {std_obs:.5f}  (attendu ≈ {1e-3*np.sqrt(3):.5f})")

    # ── Test fields_to_graph ─────────────────────────────────────────
    print("\n[5] fields_to_graph")
    vx = np.random.randn(8, 8).astype(np.float32)
    vy = np.random.randn(8, 8).astype(np.float32)
    p  = np.random.randn(8, 8).astype(np.float32)
    ei_s, ea_s = build_grid_edges(8, 8)
    graph = fields_to_graph(vx, vy, p, ei_s, ea_s)
    assert graph.x.shape == (64, 3)
    print(f"    ✅ Graphe : {graph.x.shape} nœuds, {graph.edge_index.shape[1]} arêtes")

    print("\n✅ Tous les tests passent.")
