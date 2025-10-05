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
    save_folder = f'{output_folder}'


    num_rods = 300
    alpha = 100
    rod_diameter = 1/alpha
    container_size = 3
       
    # BOUNDARY CONDITION
    # q0 = create_nonintersecting_random_rods_contained_centroids(num_rods,rod_diameter, container_size)
    q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
    # q0 = create_nonintersecting_random_rods_contained(num_rods,rod_diameter, container_size)
    
    
    from potentials import dist_lin_seg_over_ij, all_pairwise_distances_xyz, create_pairs, dist_lin_seg_pbc_over_ij
    from potentials import all_pairwise_distances_xyz_pbc, acn_over_ij

    x0 = q_to_x(q0)
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    r1 = x0.reshape(-1, 6)[:,:3]
    r2 = x0.reshape(-1, 6)[:,3:]
    d0 = dist_lin_seg_over_ij(r1,r2,i_indices,j_indices)
    d1 = dist_lin_seg_pbc_over_ij(r1,r2, container_size, i_indices,j_indices)

    print(f'min dist without pbc: {jnp.min(d0)}, min dist with pbc: {jnp.min(d1)}')

    num_rods_list = [10, 50, 100, 200, 300, 500, 1000]
    for i in range(len(num_rods_list)):
        num_rods = num_rods_list[i]
        q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
        x = q_to_x(q0)
        onp.savetxt(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale3.txt',x)
        ax = plot_many_rods(q0.reshape(-1,5))
        # flat perspective
        ax.view_init(elev=0, azim=-90)
        ax.set_xlim([-container_size/2*3,container_size/2*3])
        ax.set_ylim([-container_size/2*3,container_size/2*3])
        ax.set_zlim([-container_size/2*3,container_size/2*3])


        x0 = q_to_x(q0)
        i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
        r1 = x0.reshape(-1, 6)[:,:3]
        r2 = x0.reshape(-1, 6)[:,3:]

        acn_matrix = acn_over_ij(r1, r2, i_indices, j_indices)
        print(f'Average crossing number for N={num_rods}: {jnp.sum(jnp.abs(acn_matrix)) }')

        plt.savefig(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale3.png',dpi=300)
        plt.clf()
        plt.close()

        
    
    # x = q_to_x(q0)
    # onp.savetxt(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale1.txt',x)

    # ax = plot_many_rods(q0.reshape(-1,5))
    # # flat perspective
    # ax.view_init(elev=0, azim=-90)
    # ax.set_xlim([-container_size/2*3,container_size/2*3])
    # ax.set_ylim([-container_size/2*3,container_size/2*3])
    # ax.set_zlim([-container_size/2*3,container_size/2*3])

    # plt.savefig(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale1.png',dpi=300)
    

    # todo: mujoco type visualization vs polyscope