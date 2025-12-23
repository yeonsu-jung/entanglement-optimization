#!/usr/bin/env python3
"""
Random rods: pairwise distance stats and contact scaling.

Now with:
- sweep over N (contacts vs N)
- sweep over alpha (contacts vs aspect ratio alpha = L / D)

Assumes available local functions:
  - create_random_rods(N, prng_key, size=box_size) -> (N, q_dim)
  - q_to_x(q) -> (..., 6) endpoint representation [p1(3), p2(3)]
  - dist_lin_seg_over_ij(r1, r2, i_indices, j_indices) -> (num_pairs,)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from protocols4 import create_random_rods
from visualizations import plot_many_rods
from potentials import dist_lin_seg_over_ij
from transforms import q_to_x

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # Geometry / density
    alpha: float = 100.0          # aspect ratio: L / D  (D = L / alpha)
    rod_length: float = 1.0       # rod length L
    box_size: float = 1.5         # cubic box side length (so V = box_size^3)
    Z: float = 10.0               # target density-like parameter

    # RNG and plotting
    seed: int = 11
    make_plots: bool = True

    # Sweep over N (scale factor relative to baseline N)
    sweep_trials: int = 30
    sweep_scale_max: float = 3.0

    # Sweep over alpha
    alpha_sweep_min: float = 20.0
    alpha_sweep_max: float = 300.0
    alpha_sweep_points: int = 10       # >= 2
    alpha_sweep_space: str = "log"     # "log" or "lin"

    # Contacts counted when distance < contact_threshold.
    # None -> use diameter D = L/alpha (recommended).
    contact_threshold: float | None = None

    @property
    def rod_diameter(self) -> float:
        return self.rod_length / self.alpha

    @property
    def volume(self) -> float:
        return self.box_size ** 3

    @property
    def N_baseline(self) -> int:
        # Your original scaling: N ~ V * Z / (D * L^2) with D = L/alpha
        # => N = alpha * (V * Z / L^3)
        return int(self.volume * self.Z / (self.rod_diameter * self.rod_length ** 2))

    @property
    def D_contact(self) -> float:
        return self.rod_diameter if self.contact_threshold is None else self.contact_threshold


# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------

def generate_rods(N: int, key: jax.Array, box_size: float) -> jnp.ndarray:
    """Create N random rods (q-format), reshape to (N, -1) for convenience."""
    rods = create_random_rods(N, key, size=box_size)
    return rods.reshape((N, -1)) if rods.ndim > 2 else rods


def rods_to_endpoints(rods_q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert rods (q) to endpoints (r1, r2), each shape (N, 3)."""
    x = jnp.asarray(q_to_x(rods_q), dtype=jnp.float64)  # (..., 6)
    x6 = x.reshape(-1, 6)
    r1 = x6[:, :3]
    r2 = x6[:, 3:]
    return r1, r2


def pairwise_min_distances(r1: jnp.ndarray, r2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute upper-triangular pairwise distances between line segments.

    Returns:
        i_idx, j_idx, dists  (each (num_pairs,))
    """
    N = r1.shape[0]
    i_idx, j_idx = jnp.triu_indices(N, k=1)
    dists = dist_lin_seg_over_ij(r1, r2, i_idx, j_idx)
    return i_idx, j_idx, dists


def contact_count(dists: jnp.ndarray, threshold: float) -> int:
    """Number of pairs with distance < threshold."""
    return int(jnp.count_nonzero(dists < threshold))


# -----------------------------------------------------------------------------
# Single run
# -----------------------------------------------------------------------------

def run_single(cfg: Config) -> None:
    print(f"rod_length = {cfg.rod_length}")
    print(f"rod_diameter = {cfg.rod_diameter}")
    print(f"box_size = {cfg.box_size}  (V = {cfg.volume})")
    print(f"Z = {cfg.Z}")
    N = cfg.N_baseline
    print(f"Baseline N = {N}")

    key = jax.random.PRNGKey(cfg.seed)
    rods = generate_rods(N, key, cfg.box_size)
    r1, r2 = rods_to_endpoints(rods)

    _, _, dists = pairwise_min_distances(r1, r2)
    num_contacts = contact_count(dists, cfg.D_contact)

    print(f"#pairs = {dists.size}")
    print(f"Contacts (d < {cfg.D_contact}): {num_contacts}")
    print(f"#rods = {N}")

    if cfg.make_plots:
        # Histogram of pairwise distances
        plt.figure()
        plt.hist(jnp.asarray(dists), bins=30)
        plt.xlabel("pairwise distance")
        plt.ylabel("count")
        plt.title("Histogram of pairwise distances between rods")
        plt.tight_layout()

        # Quick 3D view (best-effort)
        try:
            plot_many_rods(rods)
        except Exception as e:
            print(f"(plot_many_rods failed: {e})")

        plt.show()


# -----------------------------------------------------------------------------
# Sweep: contacts vs N
# -----------------------------------------------------------------------------

def sweep_contacts_vs_N(cfg: Config) -> None:
    """Sweep N from ~1x to sweep_scale_max× baseline and fit a power law."""
    key = jax.random.PRNGKey(cfg.seed)
    scales = jnp.linspace(1.0, cfg.sweep_scale_max, cfg.sweep_trials)
    Ns = jnp.maximum(1, jnp.round(cfg.N_baseline * scales)).astype(int)

    contacts = []
    k = key
    for N in Ns:
        k, subk = jax.random.split(k)
        rods = generate_rods(int(N), subk, cfg.box_size)
        r1, r2 = rods_to_endpoints(rods)
        _, _, dists = pairwise_min_distances(r1, r2)
        contacts.append(contact_count(dists, cfg.D_contact))
        print(f"N = {int(N):6d} | contacts = {contacts[-1]}")

    contacts = jnp.asarray(contacts, dtype=jnp.float64)
    x = jnp.asarray(Ns, dtype=jnp.float64)

    # Fit log-log: log(contacts) = a * log(N) + b  (guard zeros)
    safe_contacts = jnp.clip(contacts, 1e-12, None)
    coeffs = jnp.polyfit(jnp.log(x), jnp.log(safe_contacts), 1)
    exponent = float(coeffs[0])

    if cfg.make_plots:
        plt.figure()
        plt.loglog(x, contacts, "o-", label="data")
        fit = jnp.exp(jnp.polyval(coeffs, jnp.log(x)))
        plt.loglog(x, fit, "--", label=f"fit slope = {exponent:.2f}")
        plt.xlabel("Number of rods N")
        plt.ylabel(f"Contacts (d < {'D' if cfg.contact_threshold is None else cfg.D_contact})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"[N sweep] Estimated exponent p in contacts ~ N^p: p = {exponent:.4f}")


# -----------------------------------------------------------------------------
# NEW: Sweep: contacts vs alpha
# -----------------------------------------------------------------------------

def _alpha_grid(cfg: Config) -> jnp.ndarray:
    if cfg.alpha_sweep_space.lower().startswith("log"):
        return jnp.exp(jnp.linspace(jnp.log(cfg.alpha_sweep_min),
                                    jnp.log(cfg.alpha_sweep_max),
                                    cfg.alpha_sweep_points))
    else:
        return jnp.linspace(cfg.alpha_sweep_min, cfg.alpha_sweep_max, cfg.alpha_sweep_points)


def sweep_contacts_vs_alpha(cfg: Config) -> None:
    """
    Sweep aspect ratio alpha = L/D. For each alpha:
      - D = L/alpha
      - N = V * Z / (D * L^2)
      - Generate rods, compute pairwise distances, count contacts (d < D or custom threshold).
    Fits log(contacts) ~ m * log(alpha) + b.
    """
    assert cfg.alpha_sweep_points >= 2, "alpha_sweep_points must be >= 2"

    key = jax.random.PRNGKey(cfg.seed)
    alphas = _alpha_grid(cfg)

    contacts = []
    Ns = []
    thresholds = []
    k = key

    for a in map(float, alphas):
        # Create a per-alpha config (immutable dataclass)
        cfg_a = replace(cfg, alpha=a)
        N = cfg_a.N_baseline
        thr = cfg_a.D_contact

        k, subk = jax.random.split(k)
        rods = generate_rods(N, subk, cfg.box_size)
        r1, r2 = rods_to_endpoints(rods)
        _, _, dists = pairwise_min_distances(r1, r2)
        c = contact_count(dists, thr)

        Ns.append(N)
        contacts.append(c)
        thresholds.append(thr)
        print(f"alpha = {a:7.2f} | D = {thr:.4g} | N = {N:6d} | contacts = {c}")

    alphas = jnp.asarray(alphas, dtype=jnp.float64)
    contacts = jnp.asarray(contacts, dtype=jnp.float64)

    # Fit log-log: contacts ~ alpha^m
    safe_contacts = jnp.clip(contacts, 1e-12, None)
    coeffs = jnp.polyfit(jnp.log(alphas), jnp.log(safe_contacts), 1)
    slope = float(coeffs[0])

    if cfg.make_plots:
        # Contacts vs alpha
        plt.figure()
        plt.loglog(alphas, contacts, "o-", label="data")
        fit = jnp.exp(jnp.polyval(coeffs, jnp.log(alphas)))
        plt.loglog(alphas, fit, "--", label=f"fit slope = {slope:.2f}")
        plt.xlabel("Aspect ratio α = L/D")
        plt.ylabel(f"Contacts (d < {'D' if cfg.contact_threshold is None else cfg.D_contact})")
        plt.legend()
        plt.tight_layout()

        # Optional: also visualize N vs alpha (since N ∝ alpha here)
        plt.figure()
        plt.loglog(alphas, jnp.asarray(Ns), "o-")
        plt.xlabel("Aspect ratio α = L/D")
        plt.ylabel("N used at fixed Z, L, V")
        plt.tight_layout()

        plt.show()

    print(f"[alpha sweep] Estimated exponent m in contacts ~ alpha^m: m = {slope:.4f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = Config(
        alpha=100.0,
        rod_length=1.0,
        box_size=1.5,
        Z=10.0,
        seed=11,
        make_plots=True,
        sweep_trials=30,
        sweep_scale_max=3.0,
        alpha_sweep_min=20.0,
        alpha_sweep_max=300.0,
        alpha_sweep_points=10,
        alpha_sweep_space="log",
        contact_threshold=None,  # None -> use rod_diameter per-alpha
    )

    # Single run on baseline N (histogram + 3D plot)
    run_single(cfg)

    # Sweep N and fit power law
    # sweep_contacts_vs_N(cfg)

    # NEW: Sweep alpha and fit power law
    sweep_contacts_vs_alpha(cfg)


if __name__ == "__main__":
    main()
