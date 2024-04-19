import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax

# def fixbound(num):
#     """ Ensure the number is within the bounds [0, 1]. """
#     # if num < 0:
#     #     return 0
#     # elif num > 1:
#     #     return 1
#     # return num
#     return jnp.clip(num, 0, 1)


# def dist_lin_seg(point1s, point1e, point2s, point2e):
#     """ Calculate the shortest distance between two line segments. """
#     d1 = point1e - point1s
#     d2 = point2e - point2s
#     d12 = point2s - point1s

#     D1 = jnp.dot(d1, d1)
#     D2 = jnp.dot(d2, d2)
#     S1 = jnp.dot(d1, d12)
#     S2 = jnp.dot(d2, d12)
#     R = jnp.dot(d1, d2)

#     den = D1 * D2 - R**2

#     if D1 == 0 or D2 == 0:
#         if D1 != 0:  # line1 is a segment and line2 is a point
#             u = 0
#             t = fixbound(S1 / D1)
#         elif D2 != 0:  # line2 is a segment and line1 is a point
#             t = 0
#             u = fixbound(-S2 / D2)
#         else:  # both segments are points
#             t = u = 0
#     elif den == 0:  # lines are parallel
#         t = 0
#         u = fixbound(-S2 / D2)
#         uf = fixbound(u)
#         if uf != u:
#             t = fixbound((uf * R + S1) / D1)
#             u = uf
#     else:  # general case
#         t = fixbound((S1 * D2 - S2 * R) / den)
#         u = fixbound((t * R - S2) / D2)
#         uf = fixbound(u)
#         if uf != u:
#             t = fixbound((uf * R + S1) / D1)
#             u = uf

#     # Compute distance
#     dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
#     # vec = , (point1s + d1 * t, point2s + d2 * u)
#     return dist

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
################

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

    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/jnp.linalg.norm(n1)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/jnp.linalg.norm(n2)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/jnp.linalg.norm(n3)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/jnp.linalg.norm(n4)

    return -1/4/jnp.pi*(jnp.arcsin(jnp.dot(n1,n2)) + jnp.arcsin(jnp.dot(n2,n3)) + jnp.arcsin(jnp.dot(n3,n4)) + jnp.arcsin(jnp.dot(n4,n1)))

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

    eff_pot = -1/dist + compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)

    return eff_pot

def simple_harmonic_line(q):
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

    # eff_pot = -1/dist + compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)
    eff_pot = -1/(dist-1)
    return eff_pot

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

    

    