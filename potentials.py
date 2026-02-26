import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from jax import jit

import jax.numpy as jnp
from jax import jit, lax

import numpy as onp

def fixbound_nonjax(num):
    """ Ensure the number is within the bounds [0, 1]. """
    if num < 0:
        return 0
    elif num > 1:
        return 1
    return num

@jit
def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

@jit
def safe_norm(x, axis=None, keepdims=False, eps=1e-14):
    """Compute the norm of x safely for gradients at zero."""
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)

@jit
def safe_normalize(x, eps=1e-14):
    """Normalize x safely for gradients at zero."""
    return x / safe_norm(x, eps=eps)

@jit
def _segment_segment_parameters(point1s, point1e, point2s, point2e):
    """
    Compute parameters t, u for the closest points on two segments.
    Bulletproof against parallel and zero-length segments.
    """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    eps = 1e-10

    # Standard case
    den_safe = jnp.where(jnp.abs(den) > eps, den, 1.0)
    t = fixbound(jnp.where(jnp.abs(den) > eps, (S1 * D2 - S2 * R) / den_safe, 0.))
    
    D2_safe = jnp.where(jnp.abs(D2) > eps, D2, 1.0)
    u = jnp.where(jnp.abs(D2) > eps, (t * R - S2) / D2_safe, 0.)
    
    # Clip u and re-solve for t
    uf = fixbound(u)
    
    def _recompute_t(new_u):
        D1_safe = jnp.where(jnp.abs(D1) > eps, D1, 1.0)
        return fixbound(jnp.where(jnp.abs(D1) > eps, (new_u * R + S1) / D1_safe, 0.))

    t, u = lax.cond(uf != u,
                    lambda _: (_recompute_t(uf), uf),
                    lambda _: (t, u),
                    None)
    
    # Degenerate case (parallel or one rod is a point)
    def _degenerate():
        # Fallback to t=0
        D2_safe_deg = jnp.where(jnp.abs(D2) > eps, D2, 1.0)
        u_fallback = fixbound(jnp.where(jnp.abs(D2) > eps, -S2 / D2_safe_deg, 0.))
        return 0.0, u_fallback

    t_final, u_final = lax.cond(jnp.abs(den) <= eps,
                                lambda _: _degenerate(),
                                lambda _: (t, u),
                                None)
    return t_final, u_final

@jit
def compute_distance(d1, d2, d12, t, u):
    """Compute the distance for given parameters t and u."""
    return safe_norm(d1 * t - d2 * u - d12)

@jit
def aabb_overlap_capsule(p1s, p1e, p2s, p2e, threshold):
    """
    Check if AABBs of two capsules overlap.
    threshold should be r1 + r2.
    """
    return jnp.all((jnp.minimum(p1s, p1e) <= jnp.maximum(p2s, p2e) + threshold) & 
                   (jnp.minimum(p2s, p2e) <= jnp.maximum(p1s, p1e) + threshold))

###########
@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments using JAX with safe divisions."""
    t, u = _segment_segment_parameters(point1s, point1e, point2s, point2e)
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s
    diff = d1 * t - d2 * u - d12
    return safe_norm(diff)

@jit
def dist_lin_seg_vector(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments, returning params and contact points."""
    t, u = _segment_segment_parameters(point1s, point1e, point2s, point2e)
    d1 = point1e - point1s
    d2 = point2e - point2s
    r1 = d1 * t + point1s
    r2 = d2 * u + point2s
    return t, u, r1, r2, r1-r2, safe_norm(r1 - r2)

@jit
def minimum_image(delta, box):
    """Map componentwise displacement 'delta' into [-box/2, box/2]."""
    return delta - box * jnp.round(delta / box)

@jit
def dist_lin_seg_pbc(point1s, point1e, point2s, point2e, box):
    """Shortest distance between two line segments under periodic boundary conditions."""
    m1 = 0.5 * (point1s + point1e)
    m2 = 0.5 * (point2s + point2e)
    d_mi = minimum_image(m2 - m1, box)
    shift = d_mi - (m2 - m1)
    p2s_shift = point2s + shift
    p2e_shift = point2e + shift
    return dist_lin_seg(point1s, point1e, p2s_shift, p2e_shift)

def dist_point_seg(point0, point1s, point1e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    pma = point0 - point1s
    
    l = safe_norm(point1e - point1s)
    
    n = point1e - point1s
    n = n/safe_norm(n)
    
    point1s + jnp.dot(pma,n)*n
    
    dist = safe_norm(pma - jnp.dot(pma,n)*n)

    # def case1(D1,D2,S1,S2,R):
    #     u = 0.
    #     t = fixbound(S1 / D1)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case2(D1,D2,S1,S2,R):
    #     t = 0
    #     u = fixbound(-S2 / D2)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case3(D1,D2,S1,S2,R):
    #     t = 0.
    #     u = 0.
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case4(D1,D2,S1,S2,R):
    #     t = 0.
    #     u = fixbound(-S2 / D2)
    #     uf = fixbound(u)
    #     t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case5(D1,D2,S1,S2,R):
    #     t = fixbound((S1 * D2 - S2 * R) / den)
    #     u = fixbound((t * R - S2) / D2)
    #     uf = fixbound(u)        
    #     t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # lax.cond((D1 == 0) & (D2 == 0) , lambda _: 0., lambda _: 0., None)

    # dist = lax.cond((D1 != 0.) & (D2 == 0.),
    #                     lambda _: case1(D1,D2,S1,S2,R),
    #                     lambda _: 0.,
    #                     None)
    
    # dist = lax.cond((D1 == 0.) & (D2 != 0.),
    #                     lambda _: case2(D1,D2,S1,S2,R),
    #                     lambda _: 0.,
    #                     None)    
    
    return dist

################

def dist_lin_seg_nonjax(point1s, point1e, point2s, point2e):    
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s
        
    # d1  = point1e - point1s;
    # d2  = point2e - point2s;
    # d12 = point2s - point1s;

    # D1 = onp.dot(d1, d1)
    # D2 = onp.dot(d2, d2)
    # S1 = onp.dot(d1, d12)
    # S2 = onp.dot(d2, d12)
    # R = onp.dot(d1, d2)
    D1 = np.sum(d1*d1, axis=0)
    D2 = np.sum(d2*d2, axis=0)
    S1 = np.sum(d1*d12, axis=0)
    S2 = np.sum(d2*d12, axis=0)
    R = np.sum(d1*d2, axis=0)

    den = D1 * D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:  # line1 is a segment and line2 is a point
            u = 0
            t = fixbound_nonjax(S1 / D1)
        elif D2 != 0:  # line2 is a segment and line1 is a point
            t = 0
            u = fixbound_nonjax(-S2 / D2)
        else:  # both segments are points
            t = u = 0
    elif den == 0:  # lines are parallel
        t = 0
        u = fixbound_nonjax(-S2 / D2)
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf
    else:  # general case
        t = fixbound_nonjax((S1 * D2 - S2 * R) / den)
        u = ((t * R - S2) / D2)
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf

    # Compute distance
    dist = safe_norm(d1 * t - d2 * u - d12)
    # vec = , (point1s + d1 * t, point2s + d2 * u)
    return dist


def compute_linking_number_vectorized(q):
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]
    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    return compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)

def compute_linking_number_vectorized_with_l(q,l):
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]
    
    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    return compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l)

def compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    n1 = jnp.cross(r_ij, r_ijj)
    n1 = safe_normalize(n1)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = safe_normalize(n2)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = safe_normalize(n3)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = safe_normalize(n4)
    
    tol = 1e-7
    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))

    # return -1/4/jnp.pi*(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
    #                            + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
    #                            + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
    #                            + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))

def compute_linking_number_with_6coord(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    n1 = jnp.cross(r_ij, r_ijj)
    n1 = safe_normalize(n1)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = safe_normalize(n2)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = safe_normalize(n3)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = safe_normalize(n4)
    
    tol = 0.

    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))
    
    
def compute_linking_number_arai(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    a = p_i - p_j
    b = p_i - p_jj
    c = p_ii - p_jj
    d = p_ii - p_j

    cross_bc = jnp.cross(b, c)
    cross_da = jnp.cross(d, a)

    term1 = jnp.arctan2(jnp.dot(a, cross_bc),
                    (safe_norm(a) * safe_norm(b) * safe_norm(c) +
                     jnp.dot(a, b) * safe_norm(c) +
                     jnp.dot(c, a) * safe_norm(b) +
                     jnp.dot(b, c) * safe_norm(a)))
    term1 = jnp.abs(term1)

    term2 = jnp.arctan2(jnp.dot(c, cross_da),
                    (safe_norm(c) * safe_norm(d) * safe_norm(a) +
                     jnp.dot(c, d) * safe_norm(a) +
                     jnp.dot(a, c) * safe_norm(d) +
                     jnp.dot(d, a) * safe_norm(c)))
    term2 = jnp.abs(term2)

    lk_ij = -1 / (2 * jnp.pi) * (term1 + term2)
    return lk_ij

@jit
def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
    """Compute linking number directly from Cartesian coordinates."""
    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/(safe_norm(n1)+tol)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/(safe_norm(n2)+tol)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/(safe_norm(n3)+tol)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/(safe_norm(n4)+tol)
    
    tol = 1e-7 # Use small tol for arcsin robustness
    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))

    
# def fast_effective_potential_all(w):
#     def body(carry,w):
#         conv = jnp.convolve(carry * w, kernel, mode='valid')
#         out = jnp.zeros_like(w).at[1:-1].set(conv)
#         return out, out
    
#     init = jnp.ones(w.shape[0])
#     kernel = jnp.ones(3)
#     return jnp.vstack([init, lax.scan(body, jnp.ones(w.shape[0]), w.T)[1]]).T

# def total_effective_potential(q_pairs):    
#     def body_fun(carry, q_pair):
#         # Increment carry by the result of effective_potential applied to q_pair
#         return carry + effective_potential(q_pair), None
    
#     # Perform scan; initial carry value is 0
#     total, _ = lax.scan(body_fun, 0, q_pairs)
    
#     return total
def create_pairs(m):
    N, M = m.shape
    # Get the upper triangular indices excluding the diagonal
    i, j = jnp.triu_indices(N, k=1)
    # Retrieve rows for each index in the pairs
    m_i = m[i]  # Shape will be (N(N-1)/2, M)
    m_j = m[j]  # Shape will be (N(N-1)/2, M)
    # Concatenate the rows from each pair horizontally
    m_pairs = jnp.concatenate([m_i, m_j], axis=1)  # Resulting shape will be (N(N-1)/2, 2M)
    return m_pairs

def create_pairs2(m,n):
    M, _ = m.shape
    N, _ = n.shape
    
    i, j = jnp.triu_indices(n=M, k=1, m=N)
    m_i = m[i]
    n_j = n[j]
    
    return jnp.concatenate([m_i, n_j], axis=1)

def linear_to_triangular(N, i, j):
    """Convert linear index to upper triangular index."""
    # Get the upper triangular indices excluding the diagonal
    i, j = jnp.triu_indices(N, k=1)
    return i, j

@jit
def total_effective_potential_aabb(q, aabb_threshold=4.0):
    """
    Optimized version using nested vmap to avoid large memory allocations.
    aabb_threshold: if > 0, use AABB pruning for Linking Number. 
                   Set to -1.0 to disable pruning.
    """
    q_mat = jnp.reshape(q, (-1, 5))
    N = q_mat.shape[0]

    def _potential_one_pair(rod_i, rod_j):
        return collision_penalized_entanglement_potential(jnp.concatenate([rod_i, rod_j]))

    # Vectorize over all pairs
    def _rod_i_to_all(rod_i_val):
        return vmap(lambda rj: _potential_one_pair(rod_i_val, rj))(q_mat)
    
    pot_matrix = vmap(_rod_i_to_all)(q_mat)
    
    # Sum upper triangle
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    return jnp.sum(pot_matrix * mask)

@jit
def total_effective_potential(q):
    """Sum -lk^2 over all unique rod pairs (no AABB pruning).

    Matches the working entangle_only reference implementation.
    Using plain vmap avoids the AABB guard that zeros out all
    well-separated pairs and causes trivial convergence.
    """
    q_mat = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q_mat)
    return jnp.sum(vmap(collision_penalized_entanglement_potential)(q_pairs))

def total_effective_potential_ref(q):
    q = jnp.reshape(q, (-1, 5))
    N = q.shape[0]
    total = 0
    for i in range(N):
        for j in range(i+1,N):
            total += collision_penalized_entanglement_potential(jnp.concatenate([q[i], q[j]]))
            
    return total

@jit
def pairwise_distance(q_pair):
    x_i =     q_pair[0]
    y_i =     q_pair[1]
    z_i =     q_pair[2]
    phi_i =   q_pair[3]
    theta_i = q_pair[4]
  
    x_j =     q_pair[5]
    y_j =     q_pair[6]
    z_j =     q_pair[7]
    phi_j =   q_pair[8]
    theta_j = q_pair[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    return dist_lin_seg(p_i, p_ii, p_j, p_jj)

@jit
def nematic_tensor(q):
    phi_i =   q[3]
    theta_i = q[4]
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])

    return jnp.outer(u_i, u_i)

@jit
def compute_nematic_order(qq):
    qq = jnp.reshape(qq, (-1, 5))
    
    def body_fun(carry, q):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + nematic_tensor(q), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, qq)



@jit
def pairwise_angle(q_pair):
    x_i =     q_pair[0]
    y_i =     q_pair[1]
    z_i =     q_pair[2]
    phi_i =   q_pair[3]
    theta_i = q_pair[4]
  
    x_j =     q_pair[5]
    y_j =     q_pair[6]
    z_j =     q_pair[7]
    phi_j =   q_pair[8]
    theta_j = q_pair[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    # return jnp.arctan2(safe_norm(jnp.cross(u_i, u_j)), jnp.dot(u_i, u_j))
    return jnp.abs(jnp.arctan(safe_norm(jnp.cross(u_i, u_j))/jnp.dot(u_i, u_j)))

@jit
def pairwise_skewness(q_pair):
    x_i =     q_pair[0]
    y_i =     q_pair[1]
    z_i =     q_pair[2]
    phi_i =   q_pair[3]
    theta_i = q_pair[4]
  
    x_j =     q_pair[5]
    y_j =     q_pair[6]
    z_j =     q_pair[7]
    phi_j =   q_pair[8]
    theta_j = q_pair[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    return skewness_lin_seg(p_i, p_ii, p_j, p_jj)

def pairwise_contact_point(q_pair):
    x_i =     q_pair[0]
    y_i =     q_pair[1]
    z_i =     q_pair[2]
    phi_i =   q_pair[3]
    theta_i = q_pair[4]
  
    x_j =     q_pair[5]
    y_j =     q_pair[6]
    z_j =     q_pair[7]
    phi_j =   q_pair[8]
    theta_j = q_pair[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    return contact_point_lin_seg(p_i, p_ii, p_j, p_jj)


@jit
def identity(x):
    return x

@jit
def skewness_lin_seg(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (identity(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,identity(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = identity(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (identity((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)
    
    def case2_2():
        t = identity((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = identity(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (identity((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    return t,u

def contact_point_lin_seg(point1s, point1e, point2s, point2e):
    """Calculate the midpoint between the two closest points on line segments."""
    t, u = _segment_segment_parameters(point1s, point1e, point2s, point2e)
    p1 = point1s + (point1e - point1s) * t
    p2 = point2s + (point2e - point2s) * u
    return (p1 + p2) / 2
    
    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    p1 = point1s + d1*t
    p2 = point2s + d2*u
    # d1 * t - d2 * u - d12
    # d12 = point2s - point1s

    return (p1+p2)/2

@jit
def all_pairwise_distances(q_pairs):
    return vmap(pairwise_distance)(q_pairs)

@jit
def all_pairwise_angles(q_pairs):
    return vmap(pairwise_angle)(q_pairs)

@jit
def pairwise_distance_xyz(q_pair):
    p_i = jnp.array([q_pair[0], q_pair[1], q_pair[2]])
    p_j = jnp.array([q_pair[6], q_pair[7], q_pair[8]])
    p_ii = jnp.array([q_pair[3], q_pair[4], q_pair[5]])
    p_jj = jnp.array([q_pair[9], q_pair[10], q_pair[11]])
    return dist_lin_seg(p_i, p_ii, p_j, p_jj)

@jit
def all_pairwise_distances_xyz(q_pairs):
    return vmap(pairwise_distance_xyz)(q_pairs)

@jit
def all_pairwise_skewness(q_pairs):
    return vmap(pairwise_skewness)(q_pairs)

@jit
def dist_lin_seg_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

@jit
def dist_lin_seg_pbc_over_ij(r1, r2, box, i_indices, j_indices):
    return vmap(lambda i, j: dist_lin_seg_pbc(r1[i], r2[i], r1[j], r2[j], box))(i_indices, j_indices)

def angle_r1_r2(point1s,point1e,point2s,point2e):
    u_i = point1e - point1s
    u_j = point2e - point2s
    return jnp.abs(jnp.arctan(safe_norm(jnp.cross(u_i, u_j))/jnp.dot(u_i, u_j)))

@jit
def angle_over_ij(r1,r2,i_indices,j_indices):
    return vmap(lambda i, j: angle_r1_r2(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

@jit
def acn_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: compute_linking_number_cartesian(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

@jit
def skewness_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: skewness_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

def create_pair3(m,n):
    M, _ = m.shape
    N, _ = n.shape
    assert(M == N)
    
    i, j = jnp.indices((M,N))
    i = i.flatten()
    j = j.flatten()
    
    m_i = m[i]
    n_j = n[j]
    
    return jnp.concatenate([m_i, n_j], axis=1)

@jit
def distance_between_two_curves(r1, r2):
    # r1 is 30, vector     
    r1 = jnp.reshape(r1, (10, 3))
    r2 = jnp.reshape(r2, (10, 3))
    e1 = jnp.concatenate([r1[0:-1,:],r1[1:,:]],axis=1)
    e2 = jnp.hstack([r2[0:-1,:],r2[1:,:]])
    pairs = create_pair3(e1,e2)
    
    d = vmap(dist_lin_seg)(pairs[:,0:3], pairs[:,3:6], pairs[:,6:9], pairs[:,9:12])
    
    return jnp.min(d)


@jit
def all_distances_between_curves(curves):
    # curves is num_rods, num_vertices, 3 array
    reshaped = curves.reshape(100,3*10)
    pairs = create_pairs2(reshaped,reshaped)
    d = vmap(distance_between_two_curves)(pairs[:,:30], pairs[:,30:])
    return d

@jit
def all_distances_between_curves2(pairs1,pairs2):
    # d = vmap(distance_between_two_curves)(pairs[:,:30], pairs[:,30:])
    d = vmap(distance_between_two_curves)(pairs1, pairs2)
    return d

@jit
def total_angle_repulsion(q):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    angles = all_pairwise_angles(q_pairs)
    return -jnp.min(angles)

@jit
def rod_floor_interaction(q):
    rf_int = lax.cond(q < -1,
                        lambda _: 1.*(q+1)**2,
                        lambda _: -0.0000001*(q+1)**2,
                        None)
    return rf_int

@jit
def floor_potential(z_m):
    def body_fun(carry, z):
        return carry + rod_floor_interaction(z), None
    total, _ = lax.scan(body_fun, 0., z_m)
    return total

@jit
def pairwise_distance_xyz_pbc(q_pair, box):
    p_i = jnp.array([q_pair[0], q_pair[1], q_pair[2]])
    p_j = jnp.array([q_pair[6], q_pair[7], q_pair[8]])
    p_ii = jnp.array([q_pair[3], q_pair[4], q_pair[5]])
    p_jj = jnp.array([q_pair[9], q_pair[10], q_pair[11]])
    return dist_lin_seg_pbc(p_i, p_ii, p_j, p_jj, box)

@jit
def all_pairwise_distances_xyz_pbc(q_pairs,box):
    return vmap(lambda x: pairwise_distance_xyz_pbc(x,box))(q_pairs)
    

def entanglement_potential(q):
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1

    return compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)

def effective_potential(q):
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
    collision_radius = 0.01
    # dist_cont = lax.cond(dist < collision_radius,
    #                      lambda _: 1./K*jnp.log(1+jnp.exp(K*(collision_radius-dist))),
    #                      lambda _: 1./K*(dist-collision_radius)**2,
    #                      None)

    dist_cont = 1.e4*(dist-collision_radius)**2

    eff_pot = dist_cont + compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)
    # eff_pot = dist_cont

    return eff_pot

# Original distance function (non-contact constraint)
def dist_penalty(dist, min_dist, epsilon=1.e-5):
    # Apply barrier only when the constraint is violated
    # return -jnp.log(dist - min_dist) if dist > min_dist else jnp.inf

    return lax.cond(dist > (min_dist),
                        lambda _: -jnp.log( (dist-min_dist) ),
                        lambda _: jnp.inf, # decrease to get more contacts
                        None)


@jit
def collision_penalized_entanglement_potential(q):
    """Per-pair potential: collision penalty + negated squared linking number.

    Minimising this objective maximises Σ lk_ij^2 (entanglement)
    while penalising rods closer than collision_radius.
    """
    x_i = q[0];  y_i = q[1];  z_i = q[2];  phi_i = q[3];  theta_i = q[4]
    x_j = q[5];  y_j = q[6];  z_j = q[7];  phi_j = q[8];  theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l * u_i
    p_jj = p_j + l * u_j

    # Collision penalty: penalise rods closer than collision_radius
    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
    # Entanglement objective: MINIMISE lk
    lk = compute_linking_number(x_i, y_i, z_i, phi_i, theta_i,
                                x_j, y_j, z_j, phi_j, theta_j, 1)
    return lk


 
@jit
def seg_seg_distance(q):
    q_mat = jnp.reshape(q, (2, 5))
    x_i, y_i, z_i, phi_i, theta_i = q_mat[0]
    x_j, y_j, z_j, phi_j, theta_j = q_mat[1]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    p_ii = p_i + u_i
    p_jj = p_j + u_j

    return dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
@jit
def total_harmonic_line(q,params):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    return jnp.sum(vmap(lambda qp: simple_harmonic_line_jump(qp, params))(q_pairs))

@jit
def total_harmonic_line_relax(q,params):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    
    # def body_fun(carry, q_pair):
    #     # Increment carry by the result of effective_potential applied to q_pair
    #     return carry + simple_harmonic_line(q_pair,params), None
    
    # total, _ = lax.scan(body_fun, 0, q_pairs) # Perform scan; initial carry value is 0

    f = lambda q_pairs: simple_harmonic_line(q_pairs,params)
    # total = vmap(f)(q_pairs)
    total = jnp.sum(vmap(f)(q_pairs))

    return total

    return dist_cont

@jit
def total_harmonic_line_with_hook(q,params):
    q = jnp.reshape(q, (-1, 5))
    pairwise_sum = total_harmonic_line(q,params)
    
    half_side = 0.05
    h1 = jnp.array([-half_side,0,half_side,-half_side,0,-half_side])
    h2 = jnp.array([-half_side,0,-half_side,half_side,0,-half_side])
    h3 = jnp.array([half_side,0,-half_side,half_side,0,half_side])
    h4 = jnp.array([half_side,0,half_side,-half_side,0,half_side])
    h5 = jnp.array([0,0,half_side,0,0,5*half_side])
        
    q = jnp.reshape(q, (-1, 5))    
    N = q.shape[0]
    
    # repeat h1, h2, h3, h4, h5 N times
    h1 = jnp.tile(h1, (N,1))
    h2 = jnp.tile(h2, (N,1))
    h3 = jnp.tile(h3, (N,1))
    h4 = jnp.tile(h4, (N,1))
    h5 = jnp.tile(h5, (N,1))
    
    qh1 = jnp.concatenate([q,h1],axis=1)
    qh2 = jnp.concatenate([q,h2],axis=1)
    qh3 = jnp.concatenate([q,h3],axis=1)
    qh4 = jnp.concatenate([q,h4],axis=1)
    qh5 = jnp.concatenate([q,h5],axis=1)
    
    total_qh = jnp.concatenate([qh1,qh2,qh3,qh4,qh5],axis=0)
    
    params["col_rad"] = params["col_rad"]/2+params["col_rad"]*0.00001
    
    # TO DO: only contacting rods........
    def body_fun(carry, qh):
        return carry + simple_harmonic_line_xyz(qh,params), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, total_qh)
    
    return total + pairwise_sum

@jit
def total_harmonic_line_with_hook(q,params):
    q = jnp.reshape(q, (-1, 5))
    pairwise_sum = total_harmonic_line(q,params)
    
    half_side = 0.05
    h1 = jnp.array([-half_side,0,half_side,-half_side,0,-half_side])
    h2 = jnp.array([-half_side,0,-half_side,half_side,0,-half_side])
    h3 = jnp.array([half_side,0,-half_side,half_side,0,half_side])
    h4 = jnp.array([half_side,0,half_side,-half_side,0,half_side])
    h5 = jnp.array([0,0,half_side,0,0,5*half_side])
        
    q = jnp.reshape(q, (-1, 5))    
    N = q.shape[0]
    
    # repeat h1, h2, h3, h4, h5 N times
    h1 = jnp.tile(h1, (N,1))
    h2 = jnp.tile(h2, (N,1))
    h3 = jnp.tile(h3, (N,1))
    h4 = jnp.tile(h4, (N,1))
    h5 = jnp.tile(h5, (N,1))
    
    qh1 = jnp.concatenate([q,h1],axis=1)
    qh2 = jnp.concatenate([q,h2],axis=1)
    qh3 = jnp.concatenate([q,h3],axis=1)
    qh4 = jnp.concatenate([q,h4],axis=1)
    qh5 = jnp.concatenate([q,h5],axis=1)
    
    total_qh = jnp.concatenate([qh1,qh2,qh3,qh4,qh5],axis=0)
    
    params["col_rad"] = params["col_rad"]/2+params["col_rad"]*0.00001
    
    # TO DO: only contacting rods........
    def body_fun(carry, qh):
        return carry + simple_harmonic_line_xyz(qh,params), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, total_qh)
    
    return total + pairwise_sum

@jit
def total_harmonic_line_with_gravity_floor(q,params):
    total_ent = total_effective_potential(q)
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    def body_fun(carry, q_pair):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + simple_harmonic_line(q_pair,params), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, q_pairs)
    
    # x_i = q[0]
    # y_i = q[1]
    z_i = q[:,2]
    phi_i = q[:,3]
    theta_i = q[:,4]
    
    # x_ii = x_i + jnp.sin(phi_i)*jnp.cos(theta_i)
    # y_ii = y_i + jnp.sin(phi_i)*jnp.sin(theta_i)
    zc = z_i + jnp.cos(phi_i)/2
    ze = z_i + jnp.cos(phi_i)
    z_m = jnp.min(jnp.array([z_i,ze]),axis=0)
    grav_cont = jnp.sum(zc)
    
    # for i in range(q.shape[0]):
    #     if z_m[i] < -1:
    #         floor_contact_cont += 1*(z_m[i])**2
    #     if z_m[i] > -1:
    #         floor_contact_cont -= 0.001*(z_m[i])**2
            
    # floor_contact_cont = 1*jnp.sum((z_m + 1)**2)
    
    # Perform scan; initial carry value is 0
    # total_floor_contact_potential, _ = lax.scan(body_fun, 0, q_pairs)
    
    return 1e-3*total_ent + total # + grav_cont + total_floor_contact_potential

@jit
def total_angle_repulsion(q):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    
    # def body_fun(carry, q_pair):
    #     # Increment carry by the result of effective_potential applied to q_pair
    #     return carry + pairwise_angle(q_pair), None
    # # Perform scan; initial carry value is 0
    # total, _ = lax.scan(body_fun, 0, q_pairs)

    # get minimum angle between rods
    angles = all_pairwise_angles(q_pairs)
    return -jnp.min(angles)

@jit
def rod_floor_interaction(q):
    rf_int = lax.cond(q < -1,
                        lambda _: 1.*(q+1)**2,
                        lambda _: -0.0000001*(q+1)**2,
                        None)
    return rf_int

@jit
def floor_potential(z_m):
    def body_fun(carry, z):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + rod_floor_interaction(z), None
    
    total, _ = lax.scan(body_fun, 0, z_m)
    return total
        
@jit
def simple_harmonic_line(q,params):
    col_rad = params["col_rad"]
    amp = params["amp"]    
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    threshold = col_rad * 2.0

    def _compute_full():
        dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
        return lax.cond(dist < threshold,
                             lambda _: amp*(dist-threshold)**2,
                             lambda _: -1.e-4*amp*(dist-threshold)**2, # decrease to get more contacts
                             None)

    return lax.cond(aabb_overlap_capsule(p_i, p_ii, p_j, p_jj, threshold),
                    lambda _: _compute_full(),
                    lambda _: -1.e-4*amp*(threshold)**2, # Approximate value for far field? 
                    # Wait, the original code had: lambda _: -1.e-4*amp*(dist-col_rad*2)**2
                    # If AABB doesn't overlap, dist > threshold.
                    # But if we want the same functional form, we need 'dist'.
                    # If we use AABB, we are trying to AVOID computing 'dist'.
                    # For simple_harmonic_line_jump, it returned 0, which was easy.
                    # For simple_harmonic_line, it has this tiny attraction -1e-4*amp*(dist-2*col_rad)**2.
                    # If this attraction is important for stability, we might STILL need dist.
                    # BUT, usually this attraction is just a numerical trick to keep things together.
                    # For large N, we probably only care about near-contacts.
                    None)
    

@jit
def simple_harmonic_line_jump(q,params):
    col_rad = params["col_rad"]
    amp = params["amp"]    
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    threshold = col_rad * 2.0
    
    def _compute_full_dist():
        dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
        return lax.cond(dist < threshold,
                        lambda _: amp*(dist-threshold)**2,
                        lambda _: 0.,
                        None)

    return lax.cond(aabb_overlap_capsule(p_i, p_ii, p_j, p_jj, threshold),
                    lambda _: _compute_full_dist(),
                    lambda _: 0.,
                    None)
@jit
def total_gaussian_line(q,params):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    
    def body_fun(carry, q_pair):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + gaussian_line(q_pair,params), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, q_pairs)
    
    return total

@jit
def gaussian_line(q,params):
    col_rad = params["col_rad"]
    amp = params["amp"]
    sigma = params["sigma"]
    
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
    return -amp*jnp.exp(-((dist-col_rad)/sigma)**2)

def simple_harmonic_line_force(q,params):
    col_rad = params["col_rad"]
    amp = params["amp"]
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    return amp*(dist-col_rad)**10

def total_harmonline_nonjax(q,params):
    q = onp.reshape(q, (-1, 5))
    
    total = 0
    for i in range(q.shape[0]):
        for j in range(i+1,q.shape[0]):
            total += simple_harmonic_line_nonjax(onp.concatenate([q[i], q[j]]), params)
    
    return total

def simple_harmonic_line_nonjax(q,params):
    col_rad = params["col_rad"]
    amp = params["amp"]
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = onp.array([x_i, y_i, z_i])
    p_j = onp.array([x_j, y_j, z_j])
    u_i = onp.array([onp.sin(phi_i)*onp.cos(theta_i), onp.sin(phi_i)*onp.sin(theta_i), onp.cos(phi_i)])
    u_j = onp.array([onp.sin(phi_j)*onp.cos(theta_j), onp.sin(phi_j)*onp.sin(theta_j), onp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
    # dist_cont = lax.cond(dist < collision_radius,
    #                      lambda _: 1./K*jnp.log(1+jnp.exp(K*(collision_radius-dist))),
    #                      lambda _: 1./K*(dist-collision_radius)**2,
    #                      None)
    return amp*(dist-col_rad)**2

def check_arai_formula():
    p1 = jnp.array([-0.5,0,0],dtype=jnp.float64)
    p2 = jnp.array([0.5,0,0],dtype=jnp.float64)
    z = 0.00000001
    q1 = jnp.array([0.,-0.5,z],dtype=jnp.float64)
    q2 = jnp.array([0.,0.5,z],dtype=jnp.float64)
    
    u1 = p2 - p1
    u1 /= safe_norm(u1)
    u2 = q2 - q1
    u2 /= safe_norm(u2)
    
    a = p1 - q1
    b = p1 - q2
    c = p2 - q2
    d = p2 - q1
    
    cross_bc = np.cross(b, c)
    cross_da = np.cross(d, a)
    
    term1 = np.arctan2(np.dot(a, cross_bc),
                    (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c) +
                     np.dot(a, b) * np.linalg.norm(c) +
                     np.dot(c, a) * np.linalg.norm(b) +
                     np.dot(b, c) * np.linalg.norm(a)))
    
    term2 = np.arctan2(np.dot(c, cross_da),
                    (np.linalg.norm(c) * np.linalg.norm(d) * np.linalg.norm(a) +
                     np.dot(c, d) * np.linalg.norm(a) +
                     np.dot(a, c) * np.linalg.norm(d) +
                     np.dot(d, a) * np.linalg.norm(c)))
    
    lk_ij = 1 / (2 * np.pi) * (term1 + term2)
    print(lk_ij)
    
    def my_cart2_sph(u):
        x,y,z=u
        hxy = jnp.hypot(x, y)
        r = jnp.hypot(hxy, z)
        theta = jnp.arctan2(hxy, z)  # Polar angle (inclination)
        phi = jnp.arctan2(y, x)      # Azimuthal angle
        return r, theta, phi

    
    r1, theta1, phi1 = my_cart2_sph(u1)
    r2, theta2, phi2 = my_cart2_sph(u2)
   
    lk = compute_linking_number_arai(p1[0], p1[1], p1[2], theta1, phi1, q1[0], q1[1], q1[2], theta2, phi2, 1)
    lk2 = compute_linking_number(p1[0], p1[1], p1[2], theta1, phi1, q1[0], q1[1], q1[2], theta2, phi2, 1)
    # lk = compute_linking_number(p1[0], p1[1], p1[2], az1, el1, q1[0], q1[1], q1[2], az2, el2, 1)
    print(lk)
    print(lk2)
    print()
    

def check_arai_formula2():
    
    p1 = np.array([704.37623584, 769.77342393, 234.01175294])
    p2 = np.array([673.55325552,791.00241085, 238.54406944])
    q1 = np.array([670.9797848, 792.98033422, 238.72843565])
    q2 = np.array([627.21769586,823.2844119, 245.21007349])
    
    d = dist_lin_seg_nonjax(p1,p2,q1,q2)
    
     
    def my_cart2_sph(u):
        x = u[0]
        y = u[1]
        z = u[2]
        
        hxy = jnp.hypot(x, y)
        r = jnp.hypot(hxy, z)
        theta = jnp.arctan2(hxy, z)  # Polar angle (inclination)
        phi = jnp.arctan2(y, x)      # Azimuthal angle
        return r, theta, phi
    
    
    
    def x_to_q(x):
        # p1 = x[0:3]
        # p2 = x[3:6]
        # q1 = x[6:9]
        # q2 = x[9:12]
        # return np.concatenate([p1,my_cart2_sph(p2),q1,my_cart2_sph(q2)])
        num_rows = x.shape[0]
        q = jnp.zeros((num_rows,5))
        for i in range(num_rows):
            u = x[i,3:6] - x[i,0:3]
            # assert(np.linalg.norm(u) == 1)            
            _,theta1,phi1=my_cart2_sph(u)
            
            # q[i] = jnp.array([x[i,0],x[i,1],x[i,2],theta1,phi1])
            q[i].at[0].set(x[i,0])
            q[i].at[1].set(x[i,1])
            q[i].at[2].set(x[i,2])
            q[i].at[3].set(theta1)
            q[i].at[4].set(phi1)
            
        return q
    
    
    x = np.vstack([p1,p2,q1,q2])
    x = x.reshape(-1,6)
    q = x_to_q(x)
    print(q)
    
    
    print(d)

def test_random_data():
    from optimization import optimize_fire_nonjax_individual
    from protocols import create_random_rods
    q0 = create_random_rods(100)
    f = total_angle_repulsion
    df = grad(f)
    Nmax = 3000
    atol = 1e-6
    dt = 1e-3
    q, f_val, num_iterations, error = optimize_fire_nonjax_individual(q0, f, df, Nmax,atol, dt)

    q_pair = create_pairs(q.reshape(-1,5))
    angles = all_pairwise_angles(q_pair)
    np.min(angles)

    from matplotlib import pyplot as plt
    plt.hist(angles*180/np.pi, bins=100)

    plt.show()

    from visualizations import plot_many_rods

    plot_many_rods(q.reshape(-1,5))
    plt.show()

def test_skewness():
    from optimization import optimize_fire_nonjax_individual
    from protocols import create_random_rods
    from transforms import x_to_q, q_to_x

    line_1 = [
        [-.5,0,0],
        [.5,0,0]
    ]

    line_2 = [
        [0,-.5,1],
        [0,.5,1]
    ]

    line_1 = np.array(line_1).flatten()
    line_2 = np.array(line_2).flatten()

    q1 = x_to_q(line_1)
    q2 = x_to_q(line_2)

    x1 = q_to_x(q1)
    x2 = q_to_x(q2)

    print(line_1 - x1)
    print(line_2 - x2)

    from matplotlib import pyplot as plt
    
    q1 = x_to_q(line_1)
    q2 = x_to_q(line_2)

    x1 = q_to_x(q1)
    x2 = q_to_x(q2)

    q_pair = np.concatenate([q1,q2],axis=1)
    q_pair = q_pair.flatten()
    q_pair = jnp.array(q_pair)
    skewness = pairwise_skewness(q_pair)
    print(skewness)

    return

def test_contact_point():
    # pairwise_contact_point
    from optimization import optimize_fire_nonjax_individual
    from protocols import create_random_rods
    from transforms import x_to_q, q_to_x

    line_1 = [
        [-.5,0,0],
        [.5,0,0]
    ]

    line_2 = [
        [0,-.5,0.2],
        [0,.5,0.2]
    ]

    line_1 = np.array(line_1).flatten()
    line_2 = np.array(line_2).flatten()

    q1 = x_to_q(line_1)
    q2 = x_to_q(line_2)

    x1 = q_to_x(q1)
    x2 = q_to_x(q2)

    print(line_1 - x1)
    print(line_2 - x2)

    from matplotlib import pyplot as plt
    
    q1 = x_to_q(line_1)
    q2 = x_to_q(line_2)

    x1 = q_to_x(q1)
    x2 = q_to_x(q2)

    q_pair = np.concatenate([q1,q2],axis=1)
    q_pair = q_pair.flatten()
    q_pair = jnp.array(q_pair)
    contact_point = pairwise_contact_point(q_pair)
    print(contact_point)

def test_dist_lin_seg_periodic():
    d12 = 0.2
    L = 0.01
    box = 1.5
    offset = jnp.array([10.,0.,0.])
    r1 = jnp.array([0., 0., -d12/2]) + offset
    r2 = jnp.array([0., 0.,  d12/2])
    p1s = r1 - jnp.array([L/2, 0., 0.])
    p1e = r1 + jnp.array([L/2, 0., 0.])
    p2s = r2 - jnp.array([0., L/2, 0.])
    p2e = r2 + jnp.array([0., L/2, 0.])
    d_no_pbc  = dist_lin_seg(p1s, p1e, p2s, p2e)
    d_with_pbc = dist_lin_seg_pbc(p1s, p1e, p2s, p2e, box)
    print("Euclidean distance:", d_no_pbc)
    print("PBC distance     :", d_with_pbc)

@jit
def dist_infinite_lines(p1s, p1e, p2s, p2e):
    u1 = p1e - p1s
    u2 = p2e - p2s
    v = p2s - p1s
    n = jnp.cross(u1, u2)
    n_norm = safe_norm(n)
    def _parallel_case():
        u1_norm2 = jnp.dot(u1, u1) + 1e-12
        proj = v - (jnp.dot(v, u1) / u1_norm2) * u1
        return safe_norm(proj)
    def _skew_case():
        return jnp.abs(jnp.dot(v, n)) / (n_norm + 1e-12)
    return lax.cond(n_norm < 1e-12, lambda _: _parallel_case(), lambda _: _skew_case(), operand=None)

@jit
def dist_infinite_lines_over_ij(r1, r2, i_indices, j_indices):
    p1s = r1[i_indices]; p1e = r2[i_indices]
    p2s = r1[j_indices]; p2e = r2[j_indices]
    u1 = p1e - p1s; u2 = p2e - p2s
    v = p2s - p1s
    n = jnp.cross(u1, u2)
    n_norm = safe_norm(n, axis=1)
    skew_dist = jnp.abs(jnp.einsum('ij,ij->i', v, n)) / (n_norm + 1e-12)
    u1_norm2 = jnp.einsum('ij,ij->i', u1, u1) + 1e-12
    proj = v - (jnp.einsum('ij,ij->i', v, u1) / u1_norm2)[:, None] * u1
    par_dist = safe_norm(proj, axis=1)
    return jnp.where(n_norm < 1e-12, par_dist, skew_dist)

@jit
def dist_lin_seg_over_ij_q(q, i_indices, j_indices, length=1.0):
    from transforms import sph2cart
    q = q.reshape(-1,5)
    center = q[:, :3]
    u = vmap(lambda p: sph2cart(p[0], p[1]))(q[:,3:])
    r1 = center - 0.5*length*u
    r2 = center + 0.5*length*u
    p1s = r1[i_indices]; p1e = r2[i_indices]
    p2s = r1[j_indices]; p2e = r2[j_indices]
    return vmap(dist_lin_seg)(p1s, p1e, p2s, p2e)

if __name__ == "__main__":
    # test_skewness()
    test_contact_point()
    test_dist_lin_seg_periodic()