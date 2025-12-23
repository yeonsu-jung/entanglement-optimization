from protocols import create_nonintersecting_random_rods, create_nonintersecting_random_rods_contained
from protocols import create_nonintersecting_random_rods_contained_centroids, create_nonintersecting_random_rods_contained_pbc
from transforms import q_to_x
import numpy as onp
import jax.numpy as jnp

import matplotlib.pyplot as plt
from visualizations import plot_many_rods


# get current folder and make output folder named the current folder's name
import os
# current filename
current_file = os.path.basename(__file__)
filename = os.path.splitext(current_file)[0]
output_folder = f'{filename}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f'Output folder: {output_folder}')


if __name__ == "__main__":
    # working()
    # projection_every_step() # too slow...
    num_rods = 300
    alpha = 100
    rod_diameter = 1/alpha
    
    # save folder: /Users/yeonsu/GitHub/entanglement-optimization/non-intersectiong
    save_folder = f'{output_folder}'

    # q0 = create_nonintersecting_random_rods(num_rods,rod_diameter)

    # rod_diameter,container_size,max_attempts=10000):
    container_size = 1.5
    # q0 = create_nonintersecting_random_rods_contained_centroids(num_rods,rod_diameter, container_size)
    q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
    # q0 = create_nonintersecting_random_rods_contained(num_rods,rod_diameter, container_size)

    # check distance
    # distance = dist_lin_seg_pbc(p_i, p_ii, p_j, p_jj, container_size)

    # q_pair = create_pairs(q.reshape(-1,5))
    # angles = all_pairwise_angles(q_pair)
    
    from potentials import dist_lin_seg_over_ij, all_pairwise_distances_xyz, create_pairs, dist_lin_seg_pbc_over_ij
    from potentials import all_pairwise_distances_xyz_pbc

    x0 = q_to_x(q0)


    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    r1 = x0.reshape(-1,6)[:,:3]
    r2 = x0.reshape(-1,6)[:,3:]
    d0 = dist_lin_seg_over_ij(r1,r2,i_indices,j_indices)

    d1 = dist_lin_seg_pbc_over_ij(r1,r2, container_size, i_indices,j_indices)



    q_pairs = create_pairs(q0.reshape(-1,5))
    d2 = all_pairwise_distances_xyz(q_pairs)
    d3 = all_pairwise_distances_xyz_pbc(q_pairs,container_size)

    print(f'min dist with pbc: {jnp.min(d0)}, min dist with pbc: {jnp.min(d1)}')
    print(f'min dist without pbc (from q_pairs): {jnp.min(d2)}, min dist with pbc (from q_pairs): {jnp.min(d3)}')

    import time

    start = time.time()
    for _ in range(10):
        all_pairwise_distances_xyz(q_pairs)

    end = time.time()
    print(f'time for 10 calls of all_pairwise_distances_xyz: {end - start} seconds')

    start = time.time()
    for _ in range(10):
        dist_lin_seg_over_ij(r1,r2,i_indices,j_indices)
    end = time.time()
    print(f'time for 10 calls of dist_lin_seg_over_ij: {end - start} seconds') 
    
    x = q_to_x(q0)
    onp.savetxt(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale1.txt',x)

    ax = plot_many_rods(q0.reshape(-1,5))
    # flat perspective
    ax.view_init(elev=0, azim=-90)
    ax.set_xlim([-container_size/2*3,container_size/2*3])
    ax.set_ylim([-container_size/2*3,container_size/2*3])
    ax.set_zlim([-container_size/2*3,container_size/2*3])

    plt.savefig(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale1.png',dpi=300)

    # check containment
    max_coord = jnp.max(x)
    min_coord = jnp.min(x)
    print(f'max_coord: {max_coord}, min_coord: {min_coord}')
    

    # todo: mujoco type visualization vs polyscope