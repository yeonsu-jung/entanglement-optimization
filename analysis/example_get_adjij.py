import numpy as np
from scipy.io import loadmat
import time
import pickle
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

@jit(nopython=True,fastmath=True)
def calculate_alignment_adjacency_numba(svd_cylinders,orientations,threshold=0.1):
    N = svd_cylinders.shape[0]
    
    lst = []    
    for i in range(N):
        point1s = svd_cylinders[i][0:3]
        point1e = svd_cylinders[i][3:6]
        orientation1 = orientations[i]

        for j in range(i+1,N):
            point2s = svd_cylinders[j][0:3]
            point2e = svd_cylinders[j][3:6]
            orientation2 = orientations[j]
            t,u,d1,d2,d12=lumelsky_dist_vec(point1s, point1e, point2s, point2e)
            
            vec = d1 * t - d2 * u - d12
            dist = np.linalg.norm(vec)
            vec = vec / dist
            score = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            if score < threshold:
                # add to adjacency matrix
                lst.append([i,j,score,dist,t,u])
                
                # if i % 1000 == 0:
                #     print(i,j)
            
    return lst


@jit(nopython=True,fastmath=True)
def calculate_lumelsky_params_numba(svd_cylinders,orientations,threshold=0.1):
    N = svd_cylinders.shape[0]
    
    lst = []    
    for i in range(N):
        point1s = svd_cylinders[i][0:3]
        point1e = svd_cylinders[i][3:6]
        orientation1 = orientations[i]

        for j in range(i+1,N):
            point2s = svd_cylinders[j][0:3]
            point2e = svd_cylinders[j][3:6]
            orientation2 = orientations[j]
            t,u,d1,d2,d12=lumelsky_dist_vec(point1s, point1e, point2s, point2e)
            
            vec = d1 * t - d2 * u - d12
            dist = np.linalg.norm(vec)
            vec = vec / dist
            score = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            if score < threshold:
                # add to adjacency matrix
                lst.append([i,j,t,u,d1,d2,d12])
                
                # if i % 1000 == 0:
                #     print(i,j)
            
    return lst


# data in: svd_cylinders; N x 7 array
# orietations are derived from svd_cylinders, N x 3 array; precomputed to save time

start = time.time()
adjij = calculate_alignment_adjacency_numba(svd_cylinders,orientations,threshold=0.3)
print(f'Elapsed time: {time.time()-start} sec')

pickle_in = open('adjacency_distance_scale0p98_threshold0p3_ij_score.pkl','wb')
pickle.dump(adjij,pickle_in)

print('Done!')