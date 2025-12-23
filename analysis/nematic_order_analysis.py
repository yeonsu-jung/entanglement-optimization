import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from core.potentials import compute_nematic_order, directors_from_q
from core.potentials import create_pairs, all_pairwise_angles


def spherical_histogram(u, bins=36):
    """Compute a spherical histogram of undirected axes.

    Parameters
    ----------
    u : (N,3) array of unit vectors
    bins : int or (int,int)
        Number of bins for (theta, phi). theta in [0,pi], phi in [0,2pi).

    Returns
    -------
    H : (Btheta, Bphi) histogram counts normalized to sum to 1.
    theta_edges, phi_edges : bin edges.
    """
    if isinstance(bins, int):
        Btheta = bins
        Bphi = bins * 2
    else:
        Btheta, Bphi = bins

    # Map axes to undirected: use polar angle in [0,pi/2]
    # We'll fold u and -u to the same direction by ensuring z >= 0
    u = np.array(u)
    u_fold = np.where(u[:,2:3] < 0, -u, u)

    x, y, z = u_fold[:,0], u_fold[:,1], u_fold[:,2]
    hxy = np.hypot(x, y)
    theta = np.arctan2(hxy, z)             # [0, pi/2]
    phi = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)

    H, theta_edges, phi_edges = np.histogram2d(theta, phi,
                                               bins=[Btheta, Bphi],
                                               range=[[0, np.pi/2], [0, 2*np.pi]],
                                               density=False)
    H = H.astype(float)
    H /= H.sum() + 1e-12
    return H, theta_edges, phi_edges


def plot_s2_distribution(u, out_prefix="directors_s2"):
    """Plot spherical histogram on the unit hemisphere, save PNG and SVG."""
    H, theta_edges, phi_edges = spherical_histogram(u)
    # Build mesh on sphere for visualization
    theta_c = 0.5*(theta_edges[:-1] + theta_edges[1:])
    phi_c = 0.5*(phi_edges[:-1] + phi_edges[1:])
    TH, PH = np.meshgrid(theta_c, phi_c, indexing='ij')
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    C = H

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=cm.viridis(C), rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_box_aspect([1,1,1])
    ax.set_title("Director distribution on S2 (hemisphere)")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=200)
    fig.savefig(f"{out_prefix}.svg")
    plt.close(fig)


def main(path_to_q_npy=None):
    """Load q, compute nematic order tensor and plot S2 distribution.

    If path_to_q_npy is None, tries 'rod.npy' in repo root.
    """
    if path_to_q_npy is None:
        path_to_q_npy = "rod.npy"
    q = np.load(path_to_q_npy)
    # q can be shaped (N,5) or flattened; handle both
    if q.ndim == 1:
        q = q.reshape(-1,5)
    Q_avg, Q, S, evals, evecs = compute_nematic_order(jnp.array(q))
    print("N rods:", q.shape[0])
    print("<u u^T> (Q_avg):\n", np.array(Q_avg))
    print("Traceless Q tensor:\n", np.array(Q))
    print("Eigenvalues:", np.array(evals))
    print("Principal director (evec for largest λ):", np.array(evecs)[:, -1])
    print("Nematic order S (largest eigenvalue):", float(S))

    # Pairwise minimal angles (optional diagnostic)
    q_pairs = create_pairs(jnp.array(q))
    angles = all_pairwise_angles(q_pairs)
    print("Mean minimal pairwise angle (deg):", float(jnp.mean(angles) * 180/np.pi))

    # Distribution on S2
    u = directors_from_q(jnp.array(q))
    plot_s2_distribution(np.array(u))
    print("Saved director distribution plots: directors_s2.png, directors_s2.svg")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
