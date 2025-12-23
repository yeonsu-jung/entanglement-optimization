# %%
#!/usr/bin/env python3
"""
Random rods: pairwise distance stats and contact scaling.

Metrics:
  - Average degree per rod:
        <k> = 2 * (#pairs with distance < threshold) / N
  - Contact gap histogram (single run):
        Histogram of signed gaps g = d_ij - D, using only g < 0 (contacts)

Sweeps:
- vs N  (scale around baseline N)
- vs α  (aspect ratio, with D = L/α and N recomputed at fixed Z, L, V)

Assumes available local functions:
  - create_random_rods(N, prng_key, size=box_size) -> (N, q_dim)
  - q_to_x(q) -> (..., 6) endpoint representation [p1(3), p2(3)]
  - dist_lin_seg_over_ij(r1, r2, i_indices, j_indices) -> (num_pairs,)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
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
    Z: float = 1.0               # density-like parameter used in N scaling

    # RNG and plotting
    seed: int = 11
    make_plots: bool = True
    plot_negative_gap_hist: bool = True  # NEW: histogram of (d - D) < 0 in single run

    # Sweep over N (scale factor relative to baseline N)
    sweep_trials: int = 30
    sweep_scale_max: float = 3.0

    # Sweep over alpha
    alpha_sweep_min: float = 20.0
    alpha_sweep_max: float = 300.0
    alpha_sweep_points: int = 10       # >= 2
    alpha_sweep_space: str = "log"     # "log" or "lin"

    # Contact threshold:
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
        # Original scaling: N ~ V * Z / (D * L^2) with D = L/alpha
        return int(self.volume * self.Z / (self.rod_diameter * self.rod_length ** 2))

    @property
    def D_contact(self) -> float:
        return self.rod_diameter if self.contact_threshold is None else self.contact_threshold


# in note
cfg = Config(
    alpha=300.0,
    rod_length=1.0,
    box_size=1.5,
    Z=20.0,
    seed=11,
    make_plots=True,
    plot_negative_gap_hist=True,  # show histogram of (d - D) < 0
    sweep_trials=30,
    sweep_scale_max=3.0,
    alpha_sweep_min=20.0,
    alpha_sweep_max=300.0,
    alpha_sweep_points=10,
    alpha_sweep_space="log",
    contact_threshold=None,  # None -> use per-alpha diameter D
)

# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------

def generate_rods(N: int, key: jax.Array, box_size: float) -> jnp.ndarray:
    """Create N random rods (q-format), reshape to (N, -1) for convenience."""
    rods = create_random_rods(N, key, size=box_size)
    return rods.reshape((N, -1)) if rods.ndim > 2 else rods

def generate_points(N: int, key: jax.Array, box_size: float) -> jnp.ndarray:
    """Create N random points in a cubic box of given size."""
    points = jax.random.uniform(key, shape=(N, 3), minval=0.0, maxval=box_size)
    return points


def rods_to_endpoints(rods_q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert rods (q) to endpoints (r1, r2), each shape (N, 3)."""
    x = jnp.asarray(q_to_x(rods_q), dtype=jnp.float64)  # (..., 6)
    x6 = x.reshape(-1, 6)
    r1 = x6[:, :3]
    r2 = x6[:, 3:]
    return r1, r2


def pairwise_min_distances(
    r1: jnp.ndarray, r2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute upper-triangular pairwise distances between line segments.

    Returns:
        i_idx, j_idx, dists  (each (num_pairs,))
    """
    N = r1.shape[0]
    i_idx, j_idx = jnp.triu_indices(N, k=1)
    dists = dist_lin_seg_over_ij(r1, r2, i_idx, j_idx)
    return i_idx, j_idx, dists


def contact_pairs_count(dists: jnp.ndarray, threshold: float) -> int:
    """Number of *pairs* with distance < threshold."""
    return int(jnp.count_nonzero(dists < threshold))


def average_degree_per_rod(total_pairs_below: int, N: int) -> float:
    """Average degree per rod = 2 * (#contact pairs) / N."""
    if N <= 0:
        return 0.0
    return (2.0 * float(total_pairs_below)) / float(N)

# %%
from jax import jit
from typing import Tuple

@jit
def pairwise_point_distances(points: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute upper-triangular pairwise Euclidean distances between points.

    Args:
        points: (N, 3) array of point coordinates.

    Returns:
        (i_idx, j_idx, dists):
            i_idx: (num_pairs,) int32 indices for the first point
            j_idx: (num_pairs,) int32 indices for the second point
            dists: (num_pairs,) float64 distances
    """
    points = points.astype(jnp.float64)
    N = points.shape[0]

    # Gram-matrix trick: ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
    sq_norms = jnp.sum(points * points, axis=1, keepdims=True)                # (N,1)
    G = points @ points.T                                                     # (N,N)
    sq_d2 = sq_norms + sq_norms.T - 2.0 * G                                   # (N,N)
    sq_d2 = jnp.maximum(sq_d2, 0.0)                                           # numerical safety
    D = jnp.sqrt(sq_d2)                                                       # (N,N)

    i_idx, j_idx = jnp.triu_indices(N, k=1)
    dists = D[i_idx, j_idx]
    return i_idx, j_idx, dists


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
    total_pairs = contact_pairs_count(dists, cfg.D_contact)
    avg_deg = average_degree_per_rod(total_pairs, N)

    print(f"#pairs evaluated = {dists.size}")
    print(f"Average degree per rod (d < {cfg.D_contact}): {avg_deg:.6g}")
    print(f"#rods = {N}")

    if cfg.make_plots:
        # Histogram of all pairwise distances
        plt.figure()
        plt.hist(np.asarray(dists), bins=30)
        plt.xlabel("pairwise distance d_ij")
        plt.ylabel("count")
        plt.title("Histogram of pairwise distances between rods")
        plt.tight_layout()

        # NEW: Histogram of signed gaps (d_ij - D) for *negative* values only (contacts)
        if cfg.plot_negative_gap_hist:
            signed_gaps = jnp.asarray(dists) - float(cfg.D_contact)  # d - D
            negative = signed_gaps[signed_gaps < 0.0]
            num_contacts_pairs = int(negative.size)
            print(f"Pairs with (d - D) < 0 (contacts): {num_contacts_pairs}")

            if num_contacts_pairs > 0:
                plt.figure()
                plt.hist(np.asarray(negative), bins=30)
                plt.xlabel("signed gap g = d_ij - D (only g < 0)")
                plt.ylabel("count")
                plt.title("Histogram of contact gaps (negative only)")
                plt.tight_layout()
            else:
                print("No contacting pairs (no negative signed gaps); skipping contact-gap histogram.")

        # Quick 3D visualization (best-effort)
        try:
            plot_many_rods(rods)
        except Exception as e:
            print(f"(plot_many_rods failed: {e})")

        plt.show()


# -----------------------------------------------------------------------------
# Sweep: avg degree vs N
# -----------------------------------------------------------------------------

def sweep_avg_degree_vs_N(cfg: Config) -> None:
    """Sweep N from ~1× to sweep_scale_max× baseline and fit a power law."""
    key = jax.random.PRNGKey(cfg.seed)
    scales = jnp.linspace(1.0, cfg.sweep_scale_max, cfg.sweep_trials)
    Ns = jnp.maximum(1, jnp.round(cfg.N_baseline * scales)).astype(int)

    avg_deg_list = []
    k = key
    for N in Ns:
        k, subk = jax.random.split(k)
        rods = generate_rods(int(N), subk, cfg.box_size)
        r1, r2 = rods_to_endpoints(rods)
        _, _, dists = pairwise_min_distances(r1, r2)
        total_pairs = contact_pairs_count(dists, cfg.D_contact)
        avg_deg = average_degree_per_rod(total_pairs, int(N))
        avg_deg_list.append(avg_deg)
        print(f"N = {int(N):6d} | avg degree = {avg_deg:.6g}")

    y = jnp.asarray(avg_deg_list, dtype=jnp.float64)
    x = jnp.asarray(Ns, dtype=jnp.float64)

    # Fit log-log: log(y) = a * log(N) + b  (guard zeros)
    y_safe = jnp.clip(y, 1e-12, None)
    coeffs = jnp.polyfit(jnp.log(x), jnp.log(y_safe), 1)
    exponent = float(coeffs[0])

    if cfg.make_plots:
        plt.figure()
        plt.loglog(x, y, "o-", label="data")
        fit = jnp.exp(jnp.polyval(coeffs, jnp.log(x)))
        plt.loglog(x, fit, "--", label=f"fit slope = {exponent:.2f}")
        plt.xlabel("Number of rods N")
        plt.ylabel(f"Avg degree per rod (d < {'D' if cfg.contact_threshold is None else cfg.D_contact})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"[N sweep] slope in <k> ~ N^p: p = {exponent:.4f}")


# -----------------------------------------------------------------------------
# Sweep: avg degree vs alpha
# -----------------------------------------------------------------------------

def _alpha_grid(cfg: Config) -> jnp.ndarray:
    if cfg.alpha_sweep_space.lower().startswith("log"):
        return jnp.exp(jnp.linspace(jnp.log(cfg.alpha_sweep_min),
                                    jnp.log(cfg.alpha_sweep_max),
                                    cfg.alpha_sweep_points))
    else:
        return jnp.linspace(cfg.alpha_sweep_min, cfg.alpha_sweep_max, cfg.alpha_sweep_points)


def sweep_avg_degree_vs_alpha(cfg: Config) -> None:
    """
    Sweep aspect ratio alpha = L/D. For each alpha:
      - D = L/alpha
      - N = V * Z / (D * L^2)
      - Generate rods, compute distances, compute <k> = 2 * (#pairs<thr) / N
    Fits log(<k>) ~ m * log(alpha) + b.
    """
    assert cfg.alpha_sweep_points >= 2, "alpha_sweep_points must be >= 2"

    key = jax.random.PRNGKey(cfg.seed)
    alphas = _alpha_grid(cfg)

    avg_deg = []
    Ns = []
    k = key

    for a in map(float, alphas):
        cfg_a = replace(cfg, alpha=a)
        N = cfg_a.N_baseline
        thr = cfg_a.D_contact

        k, subk = jax.random.split(k)
        rods = generate_rods(N, subk, cfg.box_size)
        r1, r2 = rods_to_endpoints(rods)
        _, _, dists = pairwise_min_distances(r1, r2)
        total_pairs = contact_pairs_count(dists, thr)
        avg_k = average_degree_per_rod(total_pairs, N)

        Ns.append(N)
        avg_deg.append(avg_k)
        print(f"alpha = {a:7.2f} | D = {thr:.4g} | N = {N:6d} | avg degree = {avg_k:.6g}")

    alphas = jnp.asarray(alphas, dtype=jnp.float64)
    y = jnp.asarray(avg_deg, dtype=jnp.float64)

    # Fit log-log: <k> ~ alpha^m
    y_safe = jnp.clip(y, 1e-12, None)
    coeffs = jnp.polyfit(jnp.log(alphas), jnp.log(y_safe), 1)
    slope = float(coeffs[0])

    if cfg.make_plots:
        # <k> vs alpha
        plt.figure()
        plt.loglog(alphas, y, "o-", label="data")
        fit = jnp.exp(jnp.polyval(coeffs, jnp.log(alphas)))
        plt.loglog(alphas, fit, "--", label=f"fit slope = {slope:.2f}")
        plt.xlabel("Aspect ratio α = L/D")
        plt.ylabel(f"Avg degree per rod (d < {'D' if cfg.contact_threshold is None else cfg.D_contact})")
        plt.legend()
        plt.tight_layout()

        # Also visualize N vs alpha (since N ∝ alpha in this density model)
        plt.figure()
        plt.loglog(alphas, jnp.asarray(Ns), "o-")
        plt.xlabel("Aspect ratio α = L/D")
        plt.ylabel("N used at fixed Z, L, V")
        plt.tight_layout()

        plt.show()

    print(f"[alpha sweep] slope in <k> ~ alpha^m: m = {slope:.4f}")


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
        plot_negative_gap_hist=True,  # show histogram of (d - D) < 0
        sweep_trials=30,
        sweep_scale_max=3.0,
        alpha_sweep_min=20.0,
        alpha_sweep_max=300.0,
        alpha_sweep_points=10,
        alpha_sweep_space="log",
        contact_threshold=None,  # None -> use per-alpha diameter D
    )

    # Single run on baseline N (plots include contact-gap histogram if enabled)
    run_single(cfg)

    # Sweep N and fit power law on <k>
    # sweep_avg_degree_vs_N(cfg)

    # Sweep alpha and fit power law on <k>
    # sweep_avg_degree_vs_alpha(cfg)


if __name__ == "__main__":
    main()

# %%



print(f"rod_length = {cfg.rod_length}")
print(f"rod_diameter = {cfg.rod_diameter}")
print(f"box_size = {cfg.box_size}  (V = {cfg.volume})")
print(f"Z = {cfg.Z}")
N = cfg.N_baseline
print(f"Baseline N = {N}")

key = jax.random.PRNGKey(cfg.seed)
rods = generate_rods(N, key, cfg.box_size)
points = generate_points(N, key, cfg.box_size)
# %%
if cfg.make_plots:
    plt.figure()
    # 3d
    ax = plt.axes(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o')
    plt.xlim(0, cfg.box_size)
    plt.ylim(0, cfg.box_size)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Random points (N={N}) in box of size {cfg.box_size}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

# distance statistics
# %%
_, _, point_distances = pairwise_point_distances(points)

yy = np.asarray(point_distances)
yy = yy[yy < cfg.rod_diameter]

plt.figure()
plt.hist(yy, bins=30)
plt.xlabel('pairwise distance')
plt.ylabel('count')
plt.title('Histogram of pairwise distances between points')
plt.tight_layout()
plt.show()
        

# %%

r1, r2 = rods_to_endpoints(rods)

_, _, dists = pairwise_min_distances(r1, r2)
total_pairs = contact_pairs_count(dists, cfg.D_contact)
avg_deg = average_degree_per_rod(total_pairs, N)

print(f"#pairs evaluated = {dists.size}")
print(f"Average degree per rod (d < {cfg.D_contact}): {avg_deg:.6g}")
print(f"#rods = {N}")

if cfg.make_plots:
    # rod visualization
    ax = plot_many_rods(rods.reshape((N,-1)))
    ax.set_aspect('equal', adjustable='box')
    # ax.grid(False)
    plt.tight_layout()


if cfg.make_plots:
    # Histogram of all pairwise distances
    plt.figure()
    plt.hist(np.asarray(dists), bins=30)
    plt.xlabel("pairwise distance d_ij")
    plt.ylabel("count")
    plt.title("Histogram of pairwise distances between rods")
    plt.tight_layout()


if cfg.make_plots:
    # Histogram of all pairwise distances
    plt.figure()
    plt.hist(np.asarray(dists[dists < cfg.rod_length/30]), bins=30, density=True)
    plt.xlabel("pairwise distance d_ij")
    plt.ylabel("count")
    plt.title("Histogram of pairwise distances between rods")
    plt.tight_layout()

# %%
# simple 

import numpy as np
import matplotlib.pyplot as plt

# ----- Simulation -----
N = 200_000
z = np.random.uniform(-1.0, 1.0, size=N)   # cos(gamma) ~ Uniform(-1,1)
x = np.sqrt(1.0 - z**2)                    # sin(gamma)

# ----- Histogram -----
plt.figure(figsize=(6,4))
plt.hist(x, bins=80, density=True, alpha=0.5, label="Monte Carlo (histogram)")

# ----- Theoretical PDF -----
xs = np.linspace(1e-4, 1-1e-4, 500)
f = xs / np.sqrt(1 - xs**2)
# plt.plot(xs, f, 'r-', lw=2, label=r"$f(x)=\tfrac{x}{\sqrt{1-x^2}}$")
plt.plot(xs, f, 'r-', lw=2)
plt.ylim(0,10)

# ----- Labels -----
plt.xlabel(r"$x = \sin\gamma$")
plt.ylabel("Probability density")
plt.title("Distribution of $\sin \gamma$ for random $u_1,u_2 \in S^2$")
plt.legend()
plt.tight_layout()
plt.show()
# %%


import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters (tweak as needed) -----
N = 20_000         # number of random points in [-1,1]^3
M = 300_000        # number of random unordered pairs to sample
rng = np.random.default_rng(12345)

# ----- Sample points in [-1,1]^3 -----
pts = rng.uniform(-1.0, 1.0, size=(N, 3))

# ----- Sample M unordered distinct pairs (i<j) -----
i_idx = rng.integers(0, N, size=M)
j_idx = rng.integers(0, N, size=M)
same = i_idx == j_idx
while np.any(same):
    j_idx[same] = rng.integers(0, N, size=np.sum(same))
    same = i_idx == j_idx
swap = i_idx > j_idx
i_idx[swap], j_idx[swap] = j_idx[swap], i_idx[swap]

# ----- Compute distances -----
diff = pts[i_idx] - pts[j_idx]
d = np.linalg.norm(diff, axis=1)

# ----- Moments -----
mean_d = d.mean()
var_d = d.var()
std_d = d.std()
E_d2 = np.mean(d**2)
E_d3 = np.mean(d**3)

print(f"Box: [-1,1]^3   (side length L = 2)")
print(f"Sample size: N = {N:,} points, M = {M:,} pairs")
print(f"E[|r_i - r_j|]      ≈ {mean_d:.6f}")
print(f"Var[|r_i - r_j|]    ≈ {var_d:.6f}  (std ≈ {std_d:.6f})")
print(f"E[|r_i - r_j|^2]    ≈ {E_d2:.6f}")
print(f"E[|r_i - r_j|^3]    ≈ {E_d3:.6f}")
print()
print("Reference: mean distance in the unit cube [0,1]^3 is ≈ 0.661707;")
print("for side length 2 (our box), the mean scales to ≈ 2 × 0.661707 ≈ 1.323414.")

# ----- Histogram (density) -----
plt.figure()
plt.hist(d, bins=120, density=True, alpha=0.6)
plt.xlabel(r"$|r_i - r_j|$")
plt.ylabel("Density")
plt.title("Pairwise distance distribution in $[-1,1]^3$")
plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt

def pdf_unit_cube(v):
    """PDF of distance between two random points in [0,1]^3 (Philip, 1974).
       Domain: 0 < v <= sqrt(3).
    """
    f = np.zeros_like(v)
    # region 0 < v <= 1
    mask1 = (v > 0) & (v <= 1)
    f[mask1] = v[mask1]**2 * (4*np.pi - 6*np.pi*v[mask1] + 8*v[mask1]**2 - v[mask1]**3)

    # region 1 < v <= sqrt(2)
    mask2 = (v > 1) & (v <= np.sqrt(2))
    vv = v[mask2]
    f[mask2] = ((6*np.pi-1)*vv - 8*np.pi*vv**2 + 6*vv**3 + 2*vv**5
                + 24*vv**3*np.arctan(np.sqrt(vv**2 - 1))
                - 8*vv*(1+2*vv**2)*np.sqrt(vv**2 - 1))

    # region sqrt(2) < v <= sqrt(3)
    mask3 = (v > np.sqrt(2)) & (v <= np.sqrt(3))
    vv = v[mask3]
    f[mask3] = ((6*np.pi-5)*vv - 8*np.pi*vv**2 + 6*(np.pi-1)*vv**3 - vv**5
                + 8*vv*(1+vv**2)*np.sqrt(vv**2 - 2)
                - 24*vv*(1+vv**2)*np.arctan(np.sqrt(vv**2 - 2))
                + 24*vv**2*np.arctan(vv*np.sqrt(vv**2 - 2)))
    return f

def pdf_box(r, L=2.0):
    """PDF of distances in a cube [-L/2, L/2]^3.
       Scale factor: r = L * v.
    """
    v = r / L
    return (1.0 / L) * pdf_unit_cube(v)

# Monte Carlo sample
N, M = 20000, 200000
rng = np.random.default_rng(42)
pts = rng.uniform(-1, 1, (N, 3))
i, j = rng.integers(0, N, size=(2, M))
mask = i == j
while np.any(mask):
    j[mask] = rng.integers(0, N, size=np.sum(mask))
    mask = i == j
dist = np.linalg.norm(pts[i] - pts[j], axis=1)

# Histogram vs. theory
plt.figure(figsize=(6,4))
plt.hist(dist, bins=120, density=True, alpha=0.5, label="Monte Carlo")

rs = np.linspace(1e-4, 2*np.sqrt(3), 500)
plt.plot(rs, pdf_box(rs), 'r-', lw=2, label="Theory (Philip)")

plt.xlabel(r"$r = |r_i - r_j|$")
plt.ylabel("Density")
plt.title("Pairwise distance distribution in $[-1,1]^3$")
plt.legend()
plt.tight_layout()
plt.show()

# %%

# Distribution of the minimum distance between two random skew lines in 3D.
# Lines: (r1, u1) and (r2, u2), with u1,u2 ~ uniform on S^2, r1,r2 ~ uniform in [-1,1]^3.
# Distance D = | (u1 x u2) · (r1 - r2) | / ||u1 x u2|| = | n · (r1 - r2) |, n = unit(u1 x u2).
#
# Key result:
#   Let R = ||r1 - r2|| with PDF f_R(r) for the box (we have closed form).
#   For fixed R = r and n ~ uniform on S^2 independent of r12, |cos θ| ~ Uniform(0,1).
#   Thus D | R=r  ~ Uniform(0, r).
#   Therefore the unconditional PDF is
#       f_D(d) = ∫_{r=d}^{r_max} f_R(r) * (1/r) dr,
#   and CDF F_D(d) = ∫ f_R(r) * min(1, d/r) dr.
#
# Below: (1) implement f_R via Philip's formula for the unit cube; scale to [-1,1]^3.
#        (2) compute f_D by numerically integrating f_R(r)/r over r≥d.
#        (3) Monte Carlo simulation to verify.
#        (4) Plot histogram + theoretical curve.
#
# Note: we'll skip the measure-zero event u1 x u2 ~ 0 (nearly parallel) by resampling those rare pairs.


import numpy as np
import matplotlib.pyplot as plt

# ---------- Distance PDF in [0,1]^3 (Philip) ----------
def pdf_unit_cube(v):
    f = np.zeros_like(v)
    m1 = (v > 0) & (v <= 1)
    vv = v[m1]
    f[m1] = vv**2 * (4*np.pi - 6*np.pi*vv + 8*vv**2 - vv**3)

    m2 = (v > 1) & (v <= np.sqrt(2))
    vv = v[m2]
    f[m2] = ((6*np.pi-1)*vv - 8*np.pi*vv**2 + 6*vv**3 + 2*vv**5
             + 24*vv**3*np.arctan(np.sqrt(vv**2 - 1))
             - 8*vv*(1+2*vv**2)*np.sqrt(vv**2 - 1))

    m3 = (v > np.sqrt(2)) & (v <= np.sqrt(3))
    vv = v[m3]
    f[m3] = ((6*np.pi-5)*vv - 8*np.pi*vv**2 + 6*(np.pi-1)*vv**3 - vv**5
             + 8*vv*(1+vv**2)*np.sqrt(vv**2 - 2)
             - 24*vv*(1+vv**2)*np.arctan(np.sqrt(vv**2 - 2))
             + 24*vv**2*np.arctan(vv*np.sqrt(vv**2 - 2)))
    return f

def pdf_box_distance(r, L=2.0):
    """PDF of R = ||r1 - r2|| when r1,r2 are uniform in [-L/2, L/2]^3"""
    v = r / L
    return (1.0 / L) * pdf_unit_cube(v)

# ---------- Our target: f_D(d) = ∫_{r=d}^{rmax} f_R(r) * (1/r) dr ----------
def pdf_skew_line_distance(d, L=2.0, ngrid=4000):
    rmax = np.sqrt(3) * L
    # numerical quadrature using trapezoid on r-grid truncated at rmax
    d = np.atleast_1d(d)
    out = np.zeros_like(d, dtype=float)
    # precompute r-grid and f_R(r)/r
    r = np.linspace(1e-8, rmax, ngrid)
    g = pdf_box_distance(r, L=L) / r
    # cumulative tail integral of g from r to rmax
    # compute G(r_i) = ∫_{r_i}^{rmax} g(r) dr via reversed cumsum
    G = np.cumsum(g[::-1]) * (r[1]-r[0])
    G = G[::-1]
    # interpolate G at each d
    from numpy import interp
    out = interp(d, r, G, left=G[0], right=0.0)
    return out

# ---------- Monte Carlo simulation ----------
def random_unit_vectors(n):
    # Sample from normal and normalize
    x = np.random.normal(size=(n,3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def sample_skew_line_distance(M=300000, L=2.0, parallel_tol=1e-8):
    # points uniform in [-L/2, L/2]^3
    r1 = np.random.uniform(-L/2, L/2, size=(M,3))
    r2 = np.random.uniform(-L/2, L/2, size=(M,3))
    u1 = random_unit_vectors(M)
    u2 = random_unit_vectors(M)
    r12 = r1 - r2
    cross = np.cross(u1, u2)
    denom = np.linalg.norm(cross, axis=1)
    # resample where nearly parallel
    bad = denom < parallel_tol
    n_bad = np.sum(bad)
    while n_bad > 0:
        u2[bad] = random_unit_vectors(n_bad)
        cross[bad] = np.cross(u1[bad], u2[bad])
        denom[bad] = np.linalg.norm(cross[bad], axis=1)
        bad = denom < parallel_tol
        n_bad = np.sum(bad)
    n = cross / denom[:,None]
    D = np.abs(np.einsum('ij,ij->i', n, r12))
    return D

# ---------- Compare theory vs simulation ----------
np.random.seed(0)
M = 300_000
L = 2.0
D = sample_skew_line_distance(M=M, L=L)

# Theoretical curve
ds = np.linspace(0, np.sqrt(3)*L, 600)
f_theory = pdf_skew_line_distance(ds, L=L)

# Plot
plt.figure(figsize=(7,4))
plt.hist(D, bins=150, density=True, alpha=0.55, label="Monte Carlo")
plt.plot(ds, f_theory, lw=2, label="Theory: $f_D(d)=\\int_d^{r_{\\max}} f_R(r)\\,\\frac{dr}{r}$")
plt.xlabel("Minimum distance between two random lines (D)")
plt.ylabel("Density")
plt.title("Skew-line distance in $[-1,1]^3$, directions uniform on $S^2$")
plt.legend()
plt.tight_layout()
plt.show()

# Print empirical vs theoretical mean via the identity E[D] = ∫_0^{rmax} P(D>t) dt = ∫_0^{rmax} (1 - F_D(t)) dt
# We'll compute E[D] from the theoretical pdf numerically as a check.
dr = ds[1]-ds[0]
E_theory = np.trapz(ds * f_theory, ds)
print(f"Empirical mean ≈ {D.mean():.6f}")
print(f"Theoretical mean (numeric) ≈ {E_theory:.6f}")

# %%

