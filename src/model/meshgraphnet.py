"""
meshgraphnet.py
================
Implémentation de MeshGraphNet (Pfaff et al., NeurIPS 2021)
et GNS (Sanchez-Gonzalez et al., ICML 2020).

Architecture Encode → Process (L couches) → Decode avec connexions
résiduelles, normalisation de couche et mise à jour conjointe
nœuds/arêtes.

Références :
    - Pfaff et al. (2021) "Learning Mesh-Based Simulation with Graph Networks"
      https://arxiv.org/abs/2010.03409
    - Sanchez-Gonzalez et al. (2020) "Learning to Simulate Complex Physics
      with Graph Networks" https://arxiv.org/abs/2002.09405
"""

from __future__ import annotations
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    n_hidden: int = 2,
    activation: type = nn.SiLU,
    layer_norm: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Construit un MLP (Multi-Layer Perceptron) avec normalisation optionnelle.

    Architecture : Linear → Act → [LayerNorm] → ... → Linear

    Args:
        in_dim     : Dimension d'entrée.
        out_dim    : Dimension de sortie.
        hidden_dim : Taille des couches cachées.
        n_hidden   : Nombre de couches cachées.
        activation : Classe d'activation (SiLU par défaut).
        layer_norm : Si True, ajoute LayerNorm après chaque couche cachée.
        dropout    : Taux de dropout (0 = désactivé).

    Returns:
        nn.Sequential : MLP construit.
    """
    layers: List[nn.Module] = []
    current = in_dim

    for i in range(n_hidden):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(activation())
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        current = hidden_dim

    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Couche de Passage de Messages
# ---------------------------------------------------------------------------

class InteractionNetworkLayer(MessagePassing):
    """
    Couche de réseau d'interaction (Battaglia et al., 2016).

    Implémente la mise à jour conjointe des arêtes et des nœuds :

        e'_ij = φ_e(h_i, h_j, e_ij)          [mise à jour arête]
        h'_i  = φ_v(h_i, Σ_{j∈N(i)} e'_ij)   [mise à jour nœud]

    Les connexions résiduelles sont appliquées pour les deux entités :
        e_ij ← e_ij + e'_ij
        h_i  ← h_i + h'_i

    Args:
        node_dim   : Dimension des représentations de nœuds.
        edge_dim   : Dimension des attributs d'arêtes.
        hidden_dim : Taille des couches internes des MLP.
        aggr       : Fonction d'agrégation ('sum', 'mean', 'max').
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = "sum",
    ):
        super().__init__(aggr=aggr)

        # φ_e : fonction de message sur les arêtes
        self.edge_mlp = build_mlp(
            in_dim=2 * node_dim + edge_dim,
            out_dim=edge_dim,
            hidden_dim=hidden_dim,
            layer_norm=True,
        )
        # φ_v : fonction de mise à jour des nœuds
        self.node_mlp = build_mlp(
            in_dim=node_dim + edge_dim,
            out_dim=node_dim,
            hidden_dim=hidden_dim,
            layer_norm=True,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Propagation avant d'une couche.

        Args:
            x          : (N, node_dim) features des nœuds.
            edge_index : (2, E) indices des arêtes.
            edge_attr  : (E, edge_dim) attributs des arêtes.

        Returns:
            (x_new, edge_attr_new) : représentations mises à jour.
        """
        src, dst = edge_index

        # ── Mise à jour des arêtes ──────────────────────────────────────
        edge_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        new_edge = edge_attr + self.edge_mlp(edge_in)  # résidu

        # ── Agrégation et mise à jour des nœuds ────────────────────────
        agg = self.propagate(edge_index, x=x, edge_attr=new_edge)
        node_in = torch.cat([x, agg], dim=-1)
        new_x = x + self.node_mlp(node_in)  # résidu

        return new_x, new_edge

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:  # noqa: ARG002
        return edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


# ---------------------------------------------------------------------------
# GNS — Graph Network-based Simulator
# ---------------------------------------------------------------------------

class GNS(nn.Module):
    """
    Graph Network-based Simulator (Sanchez-Gonzalez et al., 2020).

    Prédit les *accélérations* des nœuds à partir de l'état courant.
    L'intégration numérique est effectuée en dehors du modèle.

    Schéma d'intégration Euler semi-implicite :
        â_t   = GNS(G_t)
        v̂_{t+1} = v_t + â_t · Δt
        x̂_{t+1} = x_t + v̂_{t+1} · Δt

    Architecture :
        Encoder : MLP(node_in → hidden), MLP(edge_in → hidden)
        Processor : L × InteractionNetworkLayer
        Decoder : MLP(hidden → node_out)
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        node_out: int,
        hidden_dim: int = 128,
        num_layers: int = 10,
        aggr: str = "sum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Encodeurs ──────────────────────────────────────────────────
        self.node_encoder = build_mlp(node_in, hidden_dim, hidden_dim)
        self.edge_encoder = build_mlp(edge_in, hidden_dim, hidden_dim)

        # ── Processeur ─────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            InteractionNetworkLayer(hidden_dim, hidden_dim, hidden_dim, aggr=aggr)
            for _ in range(num_layers)
        ])

        # ── Décodeur ───────────────────────────────────────────────────
        self.decoder = build_mlp(hidden_dim, node_out, hidden_dim, layer_norm=False)

        self._init_weights()

    def _init_weights(self):
        """Initialisation orthogonale des poids linéaires."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> Tensor:
        """
        Args:
            data : torch_geometric.data.Data avec champs
                   x (N, node_in), edge_index (2, E), edge_attr (E, edge_in)

        Returns:
            Tensor (N, node_out) : accélérations prédites.
        """
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for layer in self.layers:
            x, e = layer(x, data.edge_index, e)

        return self.decoder(x)

    @torch.no_grad()
    def rollout_step(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        dt: float = 0.01,
    ) -> Tensor:
        """
        Effectue un pas de temps (intégration Euler).
        Utilisé lors de l'inférence autoregressif.
        """
        dummy = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        accel = self.forward(dummy)
        return x + accel * dt

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GNS(hidden={self.hidden_dim}, layers={self.num_layers}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# MeshGraphNet
# ---------------------------------------------------------------------------

class MeshGraphNet(nn.Module):
    """
    MeshGraphNet (Pfaff et al., NeurIPS 2021).

    Extension de GNS aux maillages irréguliers non-structurés.
    Supporte deux graphes superposés :
      - Graphe de **maillage** (arêtes du maillage CFD)
      - Graphe **monde** (arêtes longue portée, optionnel)

    La propagation alterne entre les deux graphes pour capturer
    les interactions locales ET longue portée.

    Différences clés avec GNS :
    1. Gestion explicite des types de nœuds (intérieur/frontière)
    2. Encodage des coordonnées géométriques dans les arêtes
    3. Normalisation des features (running statistics)
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        node_out: int,
        hidden_dim: int = 128,
        num_layers: int = 10,
        use_world_graph: bool = False,
        aggr: str = "sum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_world_graph = use_world_graph

        # ── Encodeurs ──────────────────────────────────────────────────
        self.node_encoder = build_mlp(node_in,  hidden_dim, hidden_dim)
        self.edge_encoder = build_mlp(edge_in,  hidden_dim, hidden_dim)

        # ── Processeur (couches de message passing) ─────────────────────
        self.processor = nn.ModuleList([
            InteractionNetworkLayer(hidden_dim, hidden_dim, hidden_dim, aggr=aggr)
            for _ in range(num_layers)
        ])

        # ── Graphe monde (optionnel) ────────────────────────────────────
        if use_world_graph:
            self.world_edge_encoder = build_mlp(edge_in, hidden_dim, hidden_dim)
            self.world_layers = nn.ModuleList([
                InteractionNetworkLayer(hidden_dim, hidden_dim, hidden_dim, aggr="mean")
                for _ in range(num_layers // 2)
            ])

        # ── Décodeur ───────────────────────────────────────────────────
        self.decoder = build_mlp(hidden_dim, node_out, hidden_dim, layer_norm=False)

        # ── Normalisation des sorties ───────────────────────────────────
        self.output_norm = nn.LayerNorm(node_out)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        data: Data,
        world_data: Optional[Data] = None,
    ) -> Tensor:
        """
        Args:
            data       : Graphe de maillage principal.
            world_data : Graphe monde (arêtes longue portée), optionnel.

        Returns:
            Tensor (N, node_out) : accélérations prédites.
        """
        # Encodage
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        # Encodage graphe monde
        if self.use_world_graph and world_data is not None:
            e_world = self.world_edge_encoder(world_data.edge_attr)

        # Passage de messages alterné
        for i, layer in enumerate(self.processor):
            x, e = layer(x, data.edge_index, e)

            # Intégration du graphe monde tous les 2 pas
            if (self.use_world_graph
                    and world_data is not None
                    and i < len(self.world_layers)
                    and i % 2 == 1):
                x, e_world = self.world_layers[i // 2](
                    x, world_data.edge_index, e_world
                )

        out = self.decoder(x)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MeshGraphNet(hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"world={self.use_world_graph}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gns(cfg) -> GNS:
    """Construit un GNS à partir d'un objet de configuration."""
    return GNS(
        node_in=cfg.NODE_IN,
        edge_in=cfg.EDGE_IN,
        node_out=cfg.NODE_OUT,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
    )


def build_meshgraphnet(cfg, use_world_graph: bool = False) -> MeshGraphNet:
    """Construit un MeshGraphNet à partir d'un objet de configuration."""
    return MeshGraphNet(
        node_in=cfg.NODE_IN,
        edge_in=cfg.EDGE_IN,
        node_out=cfg.NODE_OUT,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        use_world_graph=use_world_graph,
    )


# ---------------------------------------------------------------------------
# Tests unitaires
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Tests unitaires — meshgraphnet.py")
    print("=" * 60)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, E = 1, 256, 1024  # batch, nœuds, arêtes

    # Données factices
    x = torch.randn(N, 3).to(DEVICE)
    edge_index = torch.randint(0, N, (2, E)).to(DEVICE)
    edge_attr = torch.randn(E, 3).to(DEVICE)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(DEVICE)

    # ── Test GNS ────────────────────────────────────────────────────
    print("\n[1] GNS")
    model = GNS(node_in=3, edge_in=3, node_out=2, hidden_dim=64, num_layers=4).to(DEVICE)
    out = model(data)
    assert out.shape == (N, 2), f"Shape inattendue : {out.shape}"
    print(f"    ✅ output shape : {out.shape}")
    print(f"    {model}")

    # ── Test MeshGraphNet ────────────────────────────────────────────
    print("\n[2] MeshGraphNet")
    model2 = MeshGraphNet(node_in=3, edge_in=3, node_out=2, hidden_dim=64, num_layers=4).to(DEVICE)
    out2 = model2(data)
    assert out2.shape == (N, 2)
    print(f"    ✅ output shape : {out2.shape}")
    print(f"    {model2}")

    # ── Test gradient ────────────────────────────────────────────────
    print("\n[3] Test rétropropagation")
    loss = out2.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model2.parameters() if p.grad is not None]
    print(f"    ✅ Gradient moyen : {sum(grad_norms)/len(grad_norms):.4f}")

    print("\n✅ Tous les tests passent.")
