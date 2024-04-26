import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from jax import jit

import numpy as onp

# def fixbound(num):
#     """ Ensure the number is within the bounds [0, 1]. """
#     # if num < 0:
#     #     return 0
#     # elif num > 1:
#     #     return 1
#     # return num
#     return jnp.clip(num, 0, 1)
def fixbound_nonjax(num):
    """ Ensure the number is within the bounds [0, 1]. """
    if num < 0:
        return 0
    elif num > 1:
        return 1
    return num

def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

def compute_distance(d1, d2, d12, t, u):
    """Compute the distance for given parameters t and u."""
    return jnp.linalg.norm(d1 * t - d2 * u - d12)

import jax.numpy as jnp
from jax import jit, lax

def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

def compute_distance(d1, d2, d12, t, u):
    """Compute the distance for given parameters t and u."""
    return jnp.linalg.norm(d1 * t - d2 * u - d12)


###########
@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
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

    def case1(D1,D2,S1,S2,R):
        u = 0.
        t = fixbound(S1 / D1)
        return compute_distance(d1, d2, d12, t, u)
    
    def case2(D1,D2,S1,S2,R):
        t = 0
        u = fixbound(-S2 / D2)
        return compute_distance(d1, d2, d12, t, u)
    
    def case3(D1,D2,S1,S2,R):
        t = 0.
        u = 0.
        return compute_distance(d1, d2, d12, t, u)
    
    def case4(D1,D2,S1,S2,R):
        t = 0.
        u = fixbound(-S2 / D2)
        uf = fixbound(u)
        t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
        return compute_distance(d1, d2, d12, t, u)
    
    def case5(D1,D2,S1,S2,R):
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = fixbound((t * R - S2) / D2)
        uf = fixbound(u)        
        t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
        return compute_distance(d1, d2, d12, t, u)
    
    # lax.cond((D1 == 0) & (D2 == 0) , lambda _: 0., lambda _: 0., None)

    dist = lax.cond((D1 != 0.) & (D2 == 0.),
                        lambda _: case1(D1,D2,S1,S2,R),
                        lambda _: 0.,
                        None)
    
    dist = lax.cond((D1 == 0.) & (D2 != 0.),
                        lambda _: case2(D1,D2,S1,S2,R),
                        lambda _: 0.,
                        None)
    
    dist = lax.cond((D1 == 0.) & (D2 == 0.),
                        lambda _: case3(D1,D2,S1,S2,R),
                        lambda _: 0.,
                        None)
    
    dist = lax.cond((D1 != 0.) & (D2 != 0.) & (den == 0.), # parallel
                        lambda _: case4(D1,D2,S1,S2,R),
                        lambda _: case5(D1,D2,S1,S2,R),
                        None)
    
    return dist

def dist_point_seg(point0, point1s, point1e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    pma = point0 - point1s
    
    l = jnp.linalg.norm(point1e - point1s)
    
    n = point1e - point1s
    n = n/jnp.linalg.norm(n)
    
    point1s + jnp.dot(pma,n)*n
    
    dist = jnp.linalg.norm(pma - jnp.dot(pma,n)*n)

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

    D1 = onp.dot(d1, d1)
    D2 = onp.dot(d2, d2)
    S1 = onp.dot(d1, d12)
    S2 = onp.dot(d2, d12)
    R = onp.dot(d1, d2)

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
        u = fixbound_nonjax((t * R - S2) / D2)
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf

    # Compute distance
    dist = onp.linalg.norm(d1 * t - d2 * u - d12)
    # vec = , (point1s + d1 * t, point2s + d2 * u)
    return dist
    
    
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

    tol = 1e-6
    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    tol = 0.

    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))

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

    tol = 1e-6
    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    tol = 0.

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
def total_effective_potential(q):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    
    def body_fun(carry, q_pair):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + collision_penalized_entanglement_potential(q_pair), None    
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, q_pairs)
    
    return total

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
def all_pairwise_distances(q_pairs):
    return vmap(pairwise_distance)(q_pairs)

@jit
def distance_between_two_curves(r1, r2):
    # r1 is 30, vector     
    r1 = jnp.reshape(r1, (10, 3))
    r2 = jnp.reshape(r2, (10, 3))    
    e1 = jnp.concatenate([r1[0:-1,:],r1[1:,:]],axis=1)
    e2 = jnp.hstack([r2[0:-1,:],r2[1:,:]])        
    pairs = create_pairs2(e1,e2)    
    d = vmap(dist_lin_seg)(pairs[:,0:3], pairs[:,3:6], pairs[:,6:9], pairs[:,9:12])
    
    return jnp.min(d)


@jit
def all_distnaces_between_curves(curves):
    # curves is num_rods, num_vertices, 3 array
    reshaped = curves.reshape(100,3*10)
    
    pairs = create_pairs2(reshaped,reshaped)
    
    d = vmap(distance_between_two_curves)(pairs[:,:30], pairs[:,30:])
    return d
    

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

@jit
def collision_penalized_entanglement_potential(q):
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

    # p_i = jnp.array([x_i, y_i, z_i])
    # p_j = jnp.array([x_j, y_j, z_j])
    # u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    # u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    # l = 1
    # p_ii = p_i + l*u_i
    # p_jj = p_j + l*u_j
    # dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
    # collision_radius = 0.001
    # dist_cont = 1.*(dist-collision_radius)**2
    # dist_cont = 0.
    
    eff_pot = compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)
    return eff_pot

def seg_seg_distance(q):
    # assumming seg-seg contacts (not point-seg contacts)
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
    
    d1 = l*u_i # p_ii - p_i
    d2 = l*u_j # p_jj - p_j
    d12 = p_j - p_i

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)
    den = D1 * D2 - R**2    
    t = fixbound((S1 * D2 - S2 * R) / den)
    u = fixbound((t * R - S2) / D2)
    uf = fixbound(u)
    t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
    return compute_distance(d1, d2, d12, t, u)
    
@jit
def total_harmonic_line(q,params):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    
    def body_fun(carry, q_pair):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + simple_harmonic_line(q_pair,params), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, q_pairs)
    
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

    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)    
    
    dist_cont = lax.cond(dist < col_rad,
                         lambda _: amp*(dist-col_rad)**2,
                         lambda _: 0.0000001*amp*(dist-col_rad)**2,
                         None)
    return dist_cont



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
    return amp*(dist-col_rad)**2

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

if __name__ == "__main__":
    
    p1s = jnp.array([0, -10, 0])
    p1e = jnp.array([0, 10, 0])
    p2s = jnp.array([0, -10, 5.2323423])
    p2e = jnp.array([0, 10, 5.234234])

    # dist = dist_lin_seg(p1s, p1e, p2s, p2e)
    # print(dist)
    p1s = jnp.array([-0.5, 0, 0])    
    # sph to cart    
    phi1 = jnp.pi/2
    theta1 = 0
    
    p2s = jnp.array([0, -0.5, 1])
    phi2 = jnp.pi/2
    theta2 = jnp.pi/2

    q0 = jnp.array([*p1s, phi1, theta1, *p2s, phi2, theta2])

    p1e = p1s + jnp.array([jnp.sin(phi1)*jnp.cos(theta1), jnp.sin(phi1)*jnp.sin(theta1), jnp.cos(phi1)])
    p2e = p2s + jnp.array([jnp.sin(phi2)*jnp.cos(theta2), jnp.sin(phi2)*jnp.sin(theta2), jnp.cos(phi2)])
    
    print(p1s,p1e,p2s,p2e)
    dist = dist_lin_seg(p1s, p1e, p2s, p2e)
    print(dist)
    tmp = simple_harmonic_line(q0)



    print(tmp)

    # visualize line
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot([p1s[0], p1e[0]], [p1s[1], p1e[1]], [p1s[2], p1e[2]], 'r')
    # ax.plot([p2s[0], p2e[0]], [p2s[1], p2e[1]], [p2s[2], p2e[2]], 'b')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    

    