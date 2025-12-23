import numpy as np
import re
import os
from transforms import x_to_q, q_to_x

def create_folder(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

def get_N500_data():
    pathlist = []
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
    pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")
    pathlist = sorted(pathlist,key=lambda x: float(x.split('-AR')[1].split('-')[0]))
    return pathlist

def parse_pathname(pathname):
    dt_string = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2})',pathname).group(1)
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))
    random_keys = re.search('(\d+),(\d+),(\d+)',pathname).group(0)
    return dt_string, AR, num_rods,random_keys

def compute_nematic_order(q):
    q = np.reshape(q, (-1, 5))
    phi =   q[:,3]
    theta = q[:,4]

    u = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]).T
    outer_products = np.einsum('ni,nj->nij', u, u)  # Shape (N, 3, 3)
    S = np.mean(outer_products, axis=0)  # Shape (3, 3)

    S = S - np.eye(3)/3
    Q = 1.5*S
    return Q

def check_self_similarity(q):
    # q: (N*5,) array, packing
    from transforms import x_to_q, q_to_x
    x = q_to_x(q)
    centroids = (x[:,0:3] + x[:,3:6])/2
    packing_center = np.mean(centroids, axis=0)

    # to do: cut off around the center and see if that's self-similar (in what sense?)
    # to lower aspect ratio ones.

    def cut_packing(x, centroids):
        idx = np.where(np.linalg.norm(centroids - packing_center, axis=1) < 1.5)
        return x
    
    return 0

def get_contact_sphere(q):
    from potentials import create_pairs
    q_pairs = create_pairs(q.reshape(-1,5))
    distances = all_pairwise_distances(q_pairs)
    return distances

def q_pair_to_x_pair(q_pair):
    import jax.numpy as jnp
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

    return [p_i,p_ii,p_j,p_jj]

def cut_(q,cut_off=0.1):
    x = q_to_x(q)
    packing_center = np.mean(x.reshape(-1,3),axis=0)
    centroids = (x[:,0:3] + x[:,3:6])/2
    
    # new ends: centroids - cut_off*vector
    u = x[:,3:6] - x[:,0:3]
    new_starting_ends = centroids - cut_off*u
    new_ending_ends = centroids + cut_off*u

    x_new = np.concatenate([new_starting_ends,new_ending_ends],axis=1)
    q_new = x_to_q(x_new)

    return x_new,q_new

def get_contact_sphere(q):
    # contact_points = vmap(jit(pairwise_contact_point))(q_pairs)
    # contact_radii = np.linalg.norm(contact_points,axis=1)

    from potentials import create_pairs, all_pairwise_distances
    q_pairs = create_pairs( q.reshape(-1,5) )
    distances = all_pairwise_distances(q_pairs)



    return distances

def find_csv_file(folder_path):
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            continue
        else:
            possible_paths.append(pth)    
    if len(possible_paths) == 0:
        print('No csv files found in the folder')
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        # find heaviest file
        max_size = 0
        for pth in possible_paths:
            size = os.path.getsize(pth)
            if size > max_size:
                max_size = size
                heaviest_file = pth
        possible_paths = [heaviest_file]
        
    pth = str(possible_paths[0])
    return pth