"""
metrics.py
===========
Métriques d'évaluation pour les simulateurs de fluides par GNN.

Métriques implémentées :
  - NMSE(t)          : Erreur quadratique moyenne normalisée en fonction du temps
  - Valid Time t*    : Horizon de prédiction valide (NMSE < seuil)
  - Corrélation(t)   : Corrélation spatiale entre prédiction et référence
  - Énergie cinétique: Conservation de l'énergie E(t)
  - Spectre E(k)     : Spectre d'énergie cinétique (loi de Kolmogorov)
  - Divergence       : ‖div(u)‖ au cours du rollout
  - Vorticité        : Comparaison du champ de vorticité
  - SSIM             : Similarité structurelle des champs
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
import scipy.fft
import scipy.signal


# ---------------------------------------------------------------------------
# Métriques de base
# ---------------------------------------------------------------------------

def nmse(
    pred: np.ndarray,
    true: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Normalized Mean Squared Error.

    NMSE = ‖pred - true‖² / ‖true‖²

    Contrairement au MSE, le NMSE est sans dimension et permet de comparer
    des trajectoires de différentes amplitudes.

    Args:
        pred, true : tableaux de même forme.
        eps        : stabilité numérique.

    Returns:
        float : NMSE ∈ [0, +∞[, 0 = parfait.
    """
    num = ((pred - true) ** 2).mean()
    den = (true ** 2).mean() + eps
    return float(num / den)


def nmse_curve(
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
    true_vx: np.ndarray,
    true_vy: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    NMSE en fonction du temps sur un rollout.

    Args:
        pred_vx, pred_vy : (T, Nx, Ny) champs prédits.
        true_vx, true_vy : (T, Nx, Ny) champs de référence.

    Returns:
        ndarray (T,) : NMSE à chaque pas de temps.
    """
    T = min(len(pred_vx), len(true_vx))
    curve = np.zeros(T, dtype=np.float64)
    for t in range(T):
        num = (((pred_vx[t] - true_vx[t])**2 + (pred_vy[t] - true_vy[t])**2)).mean()
        den = ((true_vx[t]**2 + true_vy[t]**2)).mean() + eps
        curve[t] = num / den
    return curve


def valid_time(
    nmse_c: np.ndarray,
    threshold: float = 0.15,
) -> int:
    """
    Temps de validité t* : premier instant où NMSE dépasse le seuil.

    t* = inf{ t : NMSE(t) ≥ threshold }

    Un t* plus grand indique un meilleur simulateur.
    La valeur du seuil 0.15 est conventionnelle dans la littérature GNN.

    Args:
        nmse_c    : (T,) courbe NMSE.
        threshold : seuil NMSE (défaut 0.15 = 15%).

    Returns:
        int : indice de t* (= T si jamais atteint).
    """
    exceeded = np.where(nmse_c >= threshold)[0]
    return int(exceeded[0]) if len(exceeded) > 0 else len(nmse_c)


def spatial_correlation(
    pred: np.ndarray,
    true: np.ndarray,
) -> float:
    """
    Corrélation de Pearson spatiale entre deux champs.

    r = Σ(pred - mean_pred)(true - mean_true) / [σ_pred · σ_true · N]

    Args:
        pred, true : (Nx, Ny) ou (...) champs.

    Returns:
        float : corrélation ∈ [-1, 1], 1 = identique.
    """
    p = pred.flatten()
    t = true.flatten()
    if p.std() < 1e-10 or t.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def correlation_curve(
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
    true_vx: np.ndarray,
    true_vy: np.ndarray,
) -> np.ndarray:
    """Corrélation spatiale moyenne (vx+vy)/2 en fonction du temps."""
    T = min(len(pred_vx), len(true_vx))
    curve = np.zeros(T)
    for t in range(T):
        c_vx = spatial_correlation(pred_vx[t], true_vx[t])
        c_vy = spatial_correlation(pred_vy[t], true_vy[t])
        curve[t] = 0.5 * (c_vx + c_vy)
    return curve


# ---------------------------------------------------------------------------
# Métriques physiques
# ---------------------------------------------------------------------------

def kinetic_energy(vx: np.ndarray, vy: np.ndarray) -> float:
    """
    Énergie cinétique moyenne :
        E = (1/N) Σ_i (vx_i² + vy_i²) / 2

    La conservation de l'énergie est une propriété fondamentale des
    écoulements non-dissipatifs. Sa vérification valide la cohérence
    physique du simulateur.
    """
    return float(0.5 * (vx**2 + vy**2).mean())


def kinetic_energy_curve(
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
    true_vx: np.ndarray,
    true_vy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Énergie cinétique en fonction du temps (prédite et vraie)."""
    T = min(len(pred_vx), len(true_vx))
    ke_pred = np.array([kinetic_energy(pred_vx[t], pred_vy[t]) for t in range(T)])
    ke_true = np.array([kinetic_energy(true_vx[t], true_vy[t]) for t in range(T)])
    return ke_pred, ke_true


def kinetic_energy_spectrum(
    vx: np.ndarray,
    vy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spectre d'énergie cinétique en fonction du nombre d'onde k.

    E(k) = (1/2) Σ_{|k̃|=k} |û(k̃)|² + |v̂(k̃)|²

    Pour un écoulement turbulent développé, la loi de Kolmogorov prédit :
        E(k) ∝ k^{-5/3}   (zone inertielle)

    La vérification de ce spectre confirme que le GNN reproduit
    correctement les interactions multi-échelles.

    Args:
        vx, vy : (Nx, Ny) champs de vitesse.

    Returns:
        k_vals (kmax,), E_k (kmax,) : nombres d'onde et densité d'énergie.
    """
    Nx, Ny = vx.shape
    u_hat = scipy.fft.fft2(vx) / (Nx * Ny)
    v_hat = scipy.fft.fft2(vy) / (Nx * Ny)
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    K  = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    k_max = int(min(Nx, Ny) // 2)
    E_k   = np.zeros(k_max, dtype=np.float64)

    for k in range(1, k_max):
        mask   = (K >= k - 0.5) & (K < k + 0.5)
        E_k[k] = E_hat[mask].sum()

    return np.arange(k_max, dtype=float), E_k


def vorticity(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    Vorticité scalaire 2D :
        ω = ∂vy/∂x - ∂vx/∂y

    La vorticité mesure la rotation locale du fluide. Dans l'instabilité de
    Kelvin-Helmholtz, des structures tourbillonnaires distinctes se forment
    et leur reproduction fidèle par le GNN est un indicateur de qualité.
    """
    dx = np.gradient(vy, axis=0)
    dy = np.gradient(vx, axis=1)
    return dx - dy


def divergence_field(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    Divergence numérique :
        div(u) = ∂vx/∂x + ∂vy/∂y

    Pour un fluide incompressible, div(u) = 0. Un simulateur de qualité
    doit maintenir cette contrainte au cours du rollout.
    """
    dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / 2
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / 2
    return dvx_dx + dvy_dy


def divergence_curve(
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
) -> np.ndarray:
    """‖div(u)‖₁ moyen au cours du rollout."""
    T = len(pred_vx)
    return np.array([
        divergence_field(pred_vx[t], pred_vy[t]).abs_().mean()
        if isinstance(pred_vx[t], np.ndarray)
        else np.abs(divergence_field(pred_vx[t], pred_vy[t])).mean()
        for t in range(T)
    ])


# ---------------------------------------------------------------------------
# SSIM (Structural Similarity Index)
# ---------------------------------------------------------------------------

def ssim_2d(
    pred: np.ndarray,
    true: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """
    Indice de similarité structurelle (SSIM) entre deux champs 2D.

    SSIM ∈ [-1, 1], 1 = identique.
    Mesure simultanément la luminance, le contraste et la structure.
    Plus robuste que le MSE pour évaluer la qualité perceptuelle.

    Args:
        pred, true   : (Nx, Ny) champs à comparer.
        data_range   : Plage des données (max - min). Si None, calculé auto.

    Returns:
        float : SSIM moyen.
    """
    if data_range is None:
        data_range = float(np.abs(true).max() - np.abs(true).min() + 1e-8)

    # Paramètres SSIM standards
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window_size = 11
    sigma = 1.5

    # Fenêtre gaussienne
    gauss = scipy.signal.windows.gaussian(window_size, sigma)
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()

    def convolve(x):
        return scipy.signal.fftconvolve(x, kernel, mode="valid")

    mu_x  = convolve(pred)
    mu_y  = convolve(true)
    mu_x2 = convolve(pred * pred)
    mu_y2 = convolve(true * true)
    mu_xy = convolve(pred * true)

    sigma_x2 = mu_x2 - mu_x**2
    sigma_y2 = mu_y2 - mu_y**2
    sigma_xy = mu_xy - mu_x * mu_y

    ssim_map = (
        (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Calcul complet d'un bilan de métriques
# ---------------------------------------------------------------------------

def compute_rollout_metrics(
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
    true_vx: np.ndarray,
    true_vy: np.ndarray,
    nmse_threshold: float = 0.15,
    compute_ssim: bool = False,
) -> Dict:
    """
    Calcule l'ensemble complet des métriques pour un rollout.

    Args:
        pred_vx, pred_vy : (T, Nx, Ny) prédictions.
        true_vx, true_vy : (T, Nx, Ny) références.
        nmse_threshold   : seuil pour t*.
        compute_ssim     : Calculer ou non le SSIM (coûteux).

    Returns:
        dict avec toutes les métriques.
    """
    T = min(len(pred_vx), len(true_vx))

    # Courbes temporelles
    nmse_c   = nmse_curve(pred_vx[:T], pred_vy[:T], true_vx[:T], true_vy[:T])
    corr_c   = correlation_curve(pred_vx[:T], pred_vy[:T], true_vx[:T], true_vy[:T])
    ke_p, ke_t = kinetic_energy_curve(pred_vx[:T], pred_vy[:T], true_vx[:T], true_vy[:T])
    div_c    = divergence_curve(pred_vx[:T], pred_vy[:T])

    # Scalaires
    t_star = valid_time(nmse_c, nmse_threshold)

    # Spectre au dernier pas de temps
    k_vals_p, E_k_p = kinetic_energy_spectrum(pred_vx[-1], pred_vy[-1])
    k_vals_t, E_k_t = kinetic_energy_spectrum(true_vx[-1], true_vy[-1])

    metrics = {
        # Courbes temporelles
        "nmse_curve":       nmse_c,
        "corr_curve":       corr_c,
        "ke_pred":          ke_p,
        "ke_true":          ke_t,
        "div_curve":        div_c,
        # Scalaires
        "valid_time":       t_star,
        "mean_nmse":        float(nmse_c.mean()),
        "final_nmse":       float(nmse_c[-1]),
        "mean_corr":        float(corr_c.mean()),
        "ke_rmse":          float(np.sqrt(((ke_p - ke_t)**2).mean())),
        "mean_div":         float(div_c.mean()),
        # Spectres
        "k_vals_pred":      k_vals_p,
        "E_k_pred":         E_k_p,
        "k_vals_true":      k_vals_t,
        "E_k_true":         E_k_t,
    }

    # SSIM (optionnel, coûteux)
    if compute_ssim:
        dr = float(np.abs(true_vx).max() - np.abs(true_vx).min())
        metrics["ssim_curve"] = np.array([
            ssim_2d(pred_vx[t], true_vx[t], data_range=dr) for t in range(T)
        ])
        metrics["mean_ssim"] = float(metrics["ssim_curve"].mean())

    return metrics


def print_metrics_summary(
    metrics: Dict,
    model_name: str = "Modèle",
) -> None:
    """Affiche un résumé lisible des métriques."""
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  NMSE moyen (rollout) : {metrics['mean_nmse']:.4f}")
    print(f"  NMSE final           : {metrics['final_nmse']:.4f}")
    print(f"  Temps de validité t* : {metrics['valid_time']} pas")
    print(f"  Corrélation moyenne  : {metrics['mean_corr']:.4f}")
    print(f"  RMSE énergie cin.    : {metrics['ke_rmse']:.6f}")
    print(f"  Divergence moyenne   : {metrics['mean_div']:.6f}")
    if "mean_ssim" in metrics:
        print(f"  SSIM moyen           : {metrics['mean_ssim']:.4f}")
    print(f"{'─'*50}")


# ---------------------------------------------------------------------------
# Tests unitaires
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Tests unitaires — metrics.py")
    print("=" * 60)

    T, Nx, Ny = 50, 64, 64
    rng = np.random.default_rng(0)

    # Générer des données de test
    true_vx = rng.standard_normal((T, Nx, Ny)).astype(np.float32)
    true_vy = rng.standard_normal((T, Nx, Ny)).astype(np.float32)

    # Prédiction avec bruit croissant
    noise_levels = np.linspace(0, 1.0, T)[:, None, None]
    pred_vx = true_vx + rng.standard_normal((T, Nx, Ny)) * noise_levels
    pred_vy = true_vy + rng.standard_normal((T, Nx, Ny)) * noise_levels

    # ── NMSE ────────────────────────────────────────────────────────
    print("\n[1] NMSE")
    nc = nmse_curve(pred_vx, pred_vy, true_vx, true_vy)
    assert nc.shape == (T,)
    assert nc[0] < nc[-1], "NMSE devrait croître"
    print(f"    ✅ NMSE@t=0 : {nc[0]:.4f}  NMSE@t=T : {nc[-1]:.4f}")

    # ── Temps de validité ────────────────────────────────────────────
    print("\n[2] Valid time")
    t_star = valid_time(nc, threshold=0.5)
    print(f"    ✅ t* = {t_star} (seuil 0.5)")

    # ── Spectre ──────────────────────────────────────────────────────
    print("\n[3] Spectre d'énergie cinétique")
    k, E = kinetic_energy_spectrum(true_vx[0], true_vy[0])
    assert len(k) == len(E)
    print(f"    ✅ {len(k)} bins de nombre d'onde")

    # ── Métriques complètes ──────────────────────────────────────────
    print("\n[4] Bilan complet")
    metrics = compute_rollout_metrics(pred_vx, pred_vy, true_vx, true_vy)
    print_metrics_summary(metrics, "Modèle test")

    print("\n✅ Tous les tests passent.")
