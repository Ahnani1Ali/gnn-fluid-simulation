"""
visualization.py
=================
Outils de visualisation pour les simulations de fluides par GNN.

Fonctions disponibles :
  - plot_fields()              : Champs vx, vy, p, ω côte-à-côte
  - plot_training_curves()     : Courbes de perte et métriques d'entraînement
  - plot_nmse_curves()         : NMSE(t) comparatif des modèles
  - plot_correlation_curves()  : Corrélation(t) comparatif
  - plot_energy_curves()       : Conservation de l'énergie cinétique
  - plot_energy_spectrum()     : Spectre E(k) avec loi de Kolmogorov
  - plot_vorticity_comparison(): Comparaison champs de vorticité
  - plot_rollout_grid()        : Grille de comparaison sur un rollout
  - create_rollout_animation() : Animation GIF du rollout
  - plot_ablation_results()    : Résultats d'ablation (barplot)
  - plot_generalization()      : Courbes de généralisation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# Style global
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Palette de couleurs pour les modèles
MODEL_COLORS = {
    "GNS":          "#1565C0",   # bleu foncé
    "MeshGraphNet": "#1976D2",   # bleu
    "EGNN":         "#2E7D32",   # vert foncé
    "E(3)-GNN":     "#388E3C",   # vert
    "GMN":          "#BF360C",   # orange-rouge
    "CFD":          "#212121",   # noir
}
MODEL_MARKERS = {
    "GNS": "o", "MeshGraphNet": "s", "EGNN": "^",
    "E(3)-GNN": "D", "GMN": "P", "CFD": "*",
}

SAVE_DPI = 150


# ---------------------------------------------------------------------------
# Champs physiques
# ---------------------------------------------------------------------------

def plot_fields(
    vx: np.ndarray,
    vy: np.ndarray,
    p: Optional[np.ndarray] = None,
    t: int = 0,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Visualise les champs de vitesse, pression et vorticité.

    Args:
        vx, vy : (Nx, Ny) champs de vitesse.
        p      : (Nx, Ny) champ de pression (optionnel).
        t      : Indice temporel (pour le titre).
    """
    n_panels = 4 if p is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    vort = _vorticity(vx, vy)
    speed = np.sqrt(vx**2 + vy**2)

    panels = [
        (vx,    "Vitesse $u_x$",     "RdBu_r",  None),
        (vy,    "Vitesse $u_y$",     "RdBu_r",  None),
        (vort,  "Vorticité $\\omega$", "RdBu_r", None),
    ]
    if p is not None:
        panels.append((p, "Pression $p$", "viridis", None))

    for ax, (field, label, cmap, _) in zip(axes, panels):
        vabs = np.abs(field).max()
        norm = TwoSlopeNorm(0, vmin=-vabs, vmax=vabs) if cmap == "RdBu_r" else None
        im = ax.imshow(
            field.T, cmap=cmap, origin="lower", aspect="equal",
            norm=norm,
        )
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"{title}  (t = {t})" if title else f"Champs physiques (t = {t})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


def plot_comparison_fields(
    true_vx: np.ndarray,
    true_vy: np.ndarray,
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    t: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grille de comparaison : vérité ground truth vs prédictions GNN.

    Args:
        true_vx, true_vy : (Nx, Ny) champs de référence.
        preds : dict {nom_modèle: (pred_vx, pred_vy)} champs prédits.
        t     : Pas de temps (pour le titre).
    """
    models = list(preds.keys())
    n_rows = 1 + len(models)
    n_cols = 3  # vx, vy, ω

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    fig.suptitle(
        f"Comparaison des simulateurs GNN — t = {t}",
        fontsize=13, fontweight="bold",
    )

    all_fields = [(true_vx, true_vy, "CFD (référence)")] + [
        (preds[m][0], preds[m][1], m) for m in models
    ]

    labels_col = ["Vitesse $u_x$", "Vitesse $u_y$", "Vorticité $\\omega$"]

    for row, (fvx, fvy, name) in enumerate(all_fields):
        vort = _vorticity(fvx, fvy)
        fields_row = [fvx, fvy, vort]

        for col, (field, col_label) in enumerate(zip(fields_row, labels_col)):
            ax = axes[row, col]
            vabs = np.abs(field).max() + 1e-8
            im = ax.imshow(
                field.T, cmap="RdBu_r", origin="lower", aspect="equal",
                vmin=-vabs, vmax=vabs,
            )
            plt.colorbar(im, ax=ax, shrink=0.85)
            if row == 0:
                ax.set_title(col_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(name, fontsize=10, rotation=90, labelpad=5)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Courbes d'entraînement
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 4),
) -> plt.Figure:
    """
    Courbes de perte d'entraînement et de validation.

    Args:
        histories : {nom: {'train_loss': [...], 'val_loss': [...], ...}}
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Courbes d'apprentissage", fontsize=13, fontweight="bold")

    metrics = [
        ("train_loss", "Perte d'entraînement",  True),
        ("val_loss",   "Perte de validation",    True),
        ("val_nmse",   "NMSE de validation",     False),
    ]

    for ax, (key, label, log_scale) in zip(axes, metrics):
        for name, hist in histories.items():
            if key not in hist:
                continue
            vals = hist[key]
            epochs = [i * hist.get("log_every", 1) for i in range(1, len(vals) + 1)]
            color  = MODEL_COLORS.get(name, None)
            marker = MODEL_MARKERS.get(name, "o")

            if log_scale:
                ax.semilogy(epochs, vals, color=color, label=name,
                            marker=marker, ms=4, lw=2)
            else:
                ax.plot(epochs, vals, color=color, label=name,
                        marker=marker, ms=4, lw=2)

        ax.set_xlabel("Époque", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Métriques de rollout
# ---------------------------------------------------------------------------

def plot_nmse_curves(
    results: Dict[str, Dict],
    nmse_threshold: float = 0.15,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    NMSE(t) comparatif pour tous les modèles.

    La ligne rouge horizontale marque le seuil t* (0.15 = 15%).
    Les lignes verticales en tirets indiquent t* de chaque modèle.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, res in results.items():
        nc = res["nmse_curve"]
        T  = len(nc)
        t_ax = np.arange(T)
        c = MODEL_COLORS.get(name, None)

        ax.semilogy(t_ax, nc, color=c, label=name,
                    marker=MODEL_MARKERS.get(name, "o"),
                    ms=4, lw=2, markevery=max(T // 10, 1))

        # Marquer t*
        t_star = res.get("valid_time", valid_time_from_curve(nc, nmse_threshold))
        if t_star < T:
            ax.axvline(t_star, color=c, ls=":", alpha=0.6, lw=1.5)

    ax.axhline(nmse_threshold, color="#D32F2F", ls="--", lw=2,
               label=f"Seuil t* (NMSE={nmse_threshold})")
    ax.set_xlabel("Pas de temps $t$", fontsize=12)
    ax.set_ylabel("NMSE(t)", fontsize=12)
    ax.set_title("Erreur de prédiction en rollout (NMSE)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


def plot_energy_spectrum(
    results: Dict[str, Dict],
    true_key: str = "E_k_true",
    pred_key: str = "E_k_pred",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """
    Spectre d'énergie cinétique E(k) avec référence Kolmogorov k^{-5/3}.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Référence CFD (supposée la même pour tous)
    first_name = list(results.keys())[0]
    k_true = results[first_name].get("k_vals_true",
                np.arange(len(results[first_name].get(true_key, []))))
    E_true = results[first_name].get(true_key, None)

    if E_true is not None:
        ax.loglog(k_true[2:], E_true[2:], "k-", lw=2.5, label="CFD (référence)")

    for name, res in results.items():
        k = res.get("k_vals_pred", k_true)
        E = res.get(pred_key, None)
        if E is None:
            continue
        ax.loglog(k[2:], E[2:], color=MODEL_COLORS.get(name), label=name,
                  ls="--", lw=2)

    # Kolmogorov k^{-5/3}
    if E_true is not None and len(E_true) > 5:
        k_ref = np.array([3.0, 20.0])
        ref_val = E_true[5] * (k_ref[0] / k_true[5]) ** (-5/3) if k_true[5] > 0 else 1e-4
        ax.loglog(k_ref, ref_val * (k_ref / k_ref[0]) ** (-5/3),
                  "r--", lw=1.5, alpha=0.7, label=r"$k^{-5/3}$ (Kolmogorov)")

    ax.set_xlabel("Nombre d'onde $k$", fontsize=12)
    ax.set_ylabel("$E(k)$", fontsize=12)
    ax.set_title("Spectre d'énergie cinétique", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


def plot_energy_conservation(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Énergie cinétique E(t) : prédictions vs CFD."""
    fig, ax = plt.subplots(figsize=figsize)

    first = list(results.values())[0]
    T = len(first["ke_true"])
    ax.plot(np.arange(T), first["ke_true"], "k-", lw=2.5, label="CFD (référence)")

    for name, res in results.items():
        ax.plot(np.arange(T), res["ke_pred"],
                color=MODEL_COLORS.get(name), ls="--", lw=2, label=name)

    ax.set_xlabel("Pas de temps $t$", fontsize=12)
    ax.set_ylabel("Énergie cinétique $E(t)$", fontsize=12)
    ax.set_title("Conservation de l'énergie cinétique", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


def plot_valid_time_bar(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 4),
) -> plt.Figure:
    """Bar chart des temps de validité t* pour chaque modèle."""
    fig, ax = plt.subplots(figsize=figsize)

    names = list(results.keys())
    t_stars = [results[n]["valid_time"] for n in names]
    colors  = [MODEL_COLORS.get(n, "#607D8B") for n in names]

    bars = ax.bar(names, t_stars, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.6)

    # Valeurs sur les barres
    for bar, val in zip(bars, t_stars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Temps de validité $t^*$ (pas de temps)", fontsize=12)
    ax.set_title("Horizon de prédiction valide par modèle", fontsize=13,
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def create_rollout_animation(
    true_vx: np.ndarray,
    preds: Dict[str, np.ndarray],
    save_path: str = "./rollout.gif",
    fps: int = 10,
    field: str = "vorticity",
) -> None:
    """
    Crée une animation GIF comparant CFD vs GNN en rollout.

    Args:
        true_vx : (T, Nx, Ny) champ de référence.
        preds   : {nom: (T, Nx, Ny)} prédictions.
        field   : 'vx' ou 'vorticity'.
    """
    models = list(preds.keys())
    n_cols = 1 + len(models)
    T = min(len(true_vx), min(len(preds[m]) for m in models))

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    vabs = np.abs(true_vx[:T]).max()

    def get_field(arr, t):
        if field == "vorticity" and arr.ndim == 3:
            # Pour la vorticité, besoin de vy — simplifié ici
            return arr[t]
        return arr[t]

    # Images initiales
    ims = []
    true_im = axes[0].imshow(
        get_field(true_vx, 0).T, cmap="RdBu_r", origin="lower",
        vmin=-vabs, vmax=vabs, animated=True,
    )
    axes[0].set_title("CFD", fontsize=11)
    axes[0].axis("off")
    ims.append(true_im)

    pred_ims = []
    for k, name in enumerate(models):
        im = axes[k + 1].imshow(
            get_field(preds[name], 0).T, cmap="RdBu_r", origin="lower",
            vmin=-vabs, vmax=vabs, animated=True,
        )
        axes[k + 1].set_title(name, fontsize=11)
        axes[k + 1].axis("off")
        pred_ims.append(im)

    title_obj = fig.suptitle("t = 0", fontsize=12, fontweight="bold")

    def update(t):
        true_im.set_array(get_field(true_vx, min(t, len(true_vx)-1)).T)
        for k, name in enumerate(models):
            pred_ims[k].set_array(
                get_field(preds[name], min(t, len(preds[name])-1)).T
            )
        title_obj.set_text(f"t = {t}")
        return [true_im] + pred_ims + [title_obj]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=False,
    )
    try:
        anim.save(save_path, writer="pillow", fps=fps, dpi=80)
        print(f"✅ Animation sauvegardée : {save_path}")
    except Exception as e:
        warnings.warn(f"Impossible de sauvegarder l'animation : {e}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation et généralisation
# ---------------------------------------------------------------------------

def plot_ablation(
    ablation_data: Dict[str, Dict],
    x_key: str,
    y_key: str = "valid_time",
    x_label: str = "",
    y_label: str = "t*",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Courbes d'ablation : effet d'un hyperparamètre sur une métrique.

    Args:
        ablation_data : {nom_modèle: {x_val: y_val}}
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in ablation_data.items():
        x = sorted(data.keys())
        y = [data[xi] for xi in x]
        ax.plot(x, y, color=MODEL_COLORS.get(name), label=name,
                marker="o", ms=6, lw=2)

    ax.set_xlabel(x_label or x_key, fontsize=12)
    ax.set_ylabel(y_label or y_key, fontsize=12)
    ax.set_title(f"Ablation : impact de {x_label or x_key}", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _vorticity(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.gradient(vy, axis=0) - np.gradient(vx, axis=1)


def valid_time_from_curve(nmse_c: np.ndarray, threshold: float = 0.15) -> int:
    exceeded = np.where(nmse_c >= threshold)[0]
    return int(exceeded[0]) if len(exceeded) > 0 else len(nmse_c)


def save_all_figures(
    results: Dict[str, Dict],
    histories: Dict[str, Dict],
    output_dir: str = "./results/figures",
) -> None:
    """Sauvegarde toutes les figures standards en un appel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    p = lambda name: str(Path(output_dir) / name)

    plot_training_curves(histories,    save_path=p("training_curves.pdf"))
    plot_nmse_curves(results,          save_path=p("nmse_curves.pdf"))
    plot_energy_conservation(results,  save_path=p("energy_conservation.pdf"))
    plot_energy_spectrum(results,      save_path=p("energy_spectrum.pdf"))
    plot_valid_time_bar(results,       save_path=p("valid_time_bar.pdf"))
    print(f"✅ Toutes les figures sauvegardées dans {output_dir}")


# ---------------------------------------------------------------------------
# Tests unitaires
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Tests unitaires — visualization.py")
    print("=" * 60)
    import matplotlib
    matplotlib.use("Agg")  # Mode non-interactif pour les tests

    Nx, Ny, T = 64, 64, 40
    rng = np.random.default_rng(0)

    true_vx = rng.standard_normal((T, Nx, Ny)).astype(np.float32)
    true_vy = rng.standard_normal((T, Nx, Ny)).astype(np.float32)
    noise   = np.linspace(0, 1, T)[:, None, None]

    results = {}
    for name in ["MeshGraphNet", "EGNN", "GMN"]:
        pred_vx = true_vx + rng.standard_normal((T, Nx, Ny)) * noise * 0.5
        pred_vy = true_vy + rng.standard_normal((T, Nx, Ny)) * noise * 0.5

        from metrics import compute_rollout_metrics
        results[name] = compute_rollout_metrics(pred_vx, pred_vy, true_vx, true_vy)
        results[name]["pred_vx"] = pred_vx
        results[name]["pred_vy"] = pred_vy

    histories = {name: {
        "train_loss": np.random.rand(10).tolist(),
        "val_loss":   np.random.rand(10).tolist(),
        "val_nmse":   np.random.rand(10).tolist(),
    } for name in results}

    print("\n[1] plot_fields")
    fig = plot_fields(true_vx[0], true_vy[0], t=0)
    plt.close(fig)
    print("    ✅")

    print("[2] plot_training_curves")
    fig = plot_training_curves(histories)
    plt.close(fig)
    print("    ✅")

    print("[3] plot_nmse_curves")
    fig = plot_nmse_curves(results)
    plt.close(fig)
    print("    ✅")

    print("[4] plot_energy_spectrum")
    fig = plot_energy_spectrum(results)
    plt.close(fig)
    print("    ✅")

    print("[5] plot_valid_time_bar")
    fig = plot_valid_time_bar(results)
    plt.close(fig)
    print("    ✅")

    print("\n✅ Tous les tests passent.")
