#!/usr/bin/env python3
"""
Random rods: pairwise distance stats and contact scaling.

- Generates N random rods in a cubic box of side `box_size`.
- Computes pairwise minimum distances between rods.
- Counts "contacts" where distance < rod_diameter.
- Optionally sweeps N and fits a power-law to #contacts vs N.

Assumes available local functions:
  - create_random_rods(N, prng_key, size=box_size) -> (N, q_dim)
  - q_to_x(q) -> (..., 6) endpoint representation [p1(3), p2(3)]
  - dist_lin_seg_over_ij(r1, r2, i_indices, j_indices) -> (num_pairs,)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    alpha: float = 100.0          # aspect ratio: L / D  (D = L / alpha)
    rod_length: float = 1.0       # rod length L
    box_size: float = 1.5         # cubic box side length (so V = box_size^3)
    Z: float = 10.0               # target "reduced" density-like parameter
    seed: int = 11                # PRNG seed
    make_plots: bool = True       # show plots
    sweep_trials: int = 30        # how many N points in the sweep
    sweep_scale_max: float = 3.0  # sweep up to this multiple of baseline N
    # Contacts counted when distance < contact_threshold. By default, use diameter.
    contact_threshold: float | None = None

    @property
    def rod_diameter(self) -> float:
        return self.rod_length / self.alpha

    @property
    def volume(self) -> float:
        return self.box_size ** 3

    @property
    def N_baseline(self) -> int:
        # Your original scaling: N ~ V * Z / (D * L^2)
        return int(self.volume * self.Z / (self.rod_diameter * self.rod_length ** 2) / 10)

    @property
    def D_contact(self) -> float:
        return self.rod_diameter if self.contact_threshold is None else self.contact_threshold


# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------

def generate_rods(N: int, key: jax.Array, box_size: float, rod_qdim_hint: int | None = None) -> jnp.ndarray:
    """Create N random rods (q-format), reshape to (N, -1) for convenience."""
    rods = create_random_rods(N, key, size=box_size)  # expects keyword 'size'
    rods = rods.reshape((N, -1)) if rods.ndim > 2 else rods
    if rod_qdim_hint is not None and rods.shape[1] != rod_qdim_hint:
        # Not strictly necessary; kept as a sanity check if you care.
        pass
    return rods


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


def make_dist_matrix(N: int, i_idx: jnp.ndarray, j_idx: jnp.ndarray, dists: jnp.ndarray) -> jnp.ndarray:
    """Symmetric distance matrix with zeros on diagonal."""
    D = jnp.zeros((N, N), dtype=dists.dtype)
    D = D.at[i_idx, j_idx].set(dists)
    D = D + D.T
    return D


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

    i_idx, j_idx, dists = pairwise_min_distances(r1, r2)
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

        # Quick 3D view
        try:
            plot_many_rods(rods)
        except Exception as e:
            print(f"(plot_many_rods failed: {e})")

        plt.show()


# -----------------------------------------------------------------------------
# Sweep: contacts vs N
# -----------------------------------------------------------------------------

def sweep_contacts_vs_N(cfg: Config) -> None:
    """Sweep N from ~1x to `sweep_scale_max`× baseline and fit a power law."""
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

    # Fit log-log: log(contacts) = a * log(N) + b
    # Guard tiny zeros to avoid log(0)
    safe_contacts = jnp.clip(contacts, 1e-12, None)
    coeffs = jnp.polyfit(jnp.log(x), jnp.log(safe_contacts), 1)
    exponent = float(coeffs[0])

    if cfg.make_plots:
        plt.figure()
        plt.plot(x, contacts, "o-", label="data")
        fit = jnp.exp(jnp.polyval(coeffs, jnp.log(x)))
        plt.plot(x, fit, "--", label=f"fit slope = {exponent:.2f}")

        plt.xlabel("Number of rods N")
        plt.ylabel("Number of contacts (d < D)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Estimated power-law exponent (contacts ~ N^p): p = {exponent:.4f}")


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
        sweep_trials=10,
        sweep_scale_max=3.0,
        contact_threshold=None,  # None -> use rod_diameter
    )

    # Single run on baseline N (histogram + 3D plot)
    run_single(cfg)

    # Sweep N and fit power law
    sweep_contacts_vs_N(cfg)

    


if __name__ == "__main__":
    main()
