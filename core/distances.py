import numpy as np
from numba import jit


@jit(nopython=True)
def fixbound(x):
    if x < 0.:
        return 0.
    elif x > 1:
        return 1.
    else:
        return x

@jit(nopython=True)
def lumelsky_dist(point1s, point1e, point2s, point2e):
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)
    R = np.dot(d1, d2)

    den = D1 * D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:  # line1 is a segment and line2 is a point
            u = 0
            t = fixbound(S1 / D1)
        elif D2 != 0:  # line2 is a segment and line1 is a point
            t = 0
            u = fixbound(-S2 / D2)
        else:  # both segments are points
            t = u = 0
    elif den == 0:  # lines are parallel
        t = 0
        u = (-S2 / D2)
        uf = fixbound(u)
        if uf != u:
            t = fixbound((uf * R + S1) / D1)
            u = uf
    else:  # general case
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = ((t * R - S2) / D2)
        uf = fixbound(u)
        if uf != u:
            t = fixbound((uf * R + S1) / D1)
            u = uf

    # Compute distance
    dist = np.linalg.norm(d1 * t - d2 * u - d12)
    return dist

from numba import jit
@jit(nopython=True)
def pdist2(rr1,rr2):
    n = rr1.shape[0]
    m = rr2.shape[0]
    dist_matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i,j] = np.linalg.norm(rr1[i] - rr2[j])
    return dist_matrix

@jit(nopython=True)
def fast_lumelsky_dist_mat(rr_i,rr_j):
    dist_mat = np.zeros((len(rr_i)-1,len(rr_j)-1))            
    for i in range(len(rr_i)-1):
        p1 = rr_i[i]
        q1 = rr_i[i+1]
        for j in range(len(rr_j)-1):
            p2 = rr_j[j]
            q2 = rr_j[j+1]
            dist_mat[i,j] = lumelsky_dist(p1,q1,p2,q2)
    return dist_mat

@jit(nopython=True)
def lumelsky_dist_vec(point1s, point1e, point2s, point2e):
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)
    R = np.dot(d1, d2)

    den = D1 * D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:  # line1 is a segment and line2 is a point
            u = 0
            t = fixbound(S1 / D1)
        elif D2 != 0:  # line2 is a segment and line1 is a point
            t = 0
            u = fixbound(-S2 / D2)
        else:  # both segments are points
            t = u = 0
    elif den == 0:  # lines are parallel
        t = 0
        u = (-S2 / D2)
        uf = fixbound(u)
        if uf != u:
            t = fixbound((uf * R + S1) / D1)
            u = uf
    else:  # general case
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = ((t * R - S2) / D2)
        uf = fixbound(u)
        if uf != u:
            t = fixbound((uf * R + S1) / D1)
            u = uf

    # # Compute distance
    # dist = np.linalg.norm(d1 * t - d2 * u - d12)
    return t,u,d1,d2,d12

@jit(nopython=True)
def calculate_lumelsky_dist_mat(centerlines,neighbors):
    N = len(centerlines)
    ultimate_dist_mat = np.ones((N,N))*np.inf
    for i in range(N):
        if not neighbors[i]:
            continue
        
        rr_i = centerlines[i]
        for j in neighbors[i]:
            rr_j = centerlines[j]
            ultimate_dist_mat[i,j] = fast_lumelsky_dist(rr_i,rr_j)

@jit(nopython=True)
def compute_cylinder_distance_matrix(svd_cylinders):
    N = svd_cylinders.shape[0]
    distances = np.zeros((N, N))
    for i in range(N):
        p1 = svd_cylinders[i, 0:3]
        q1 = svd_cylinders[i, 3:6]
        for j in range(i+1, N):
            p2 = svd_cylinders[j, 0:3]
            q2 = svd_cylinders[j, 3:6]
            distances[i, j] = lumelsky_dist(p1, q1, p2, q2)
    
    return distances