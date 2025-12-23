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

from pathlib import Path

output_folder = Path(output_folder)
MOVIE_DIR = f"{output_folder}/movie"
if not os.path.exists(MOVIE_DIR):
    os.makedirs(MOVIE_DIR)
    print(f'Movie folder: {MOVIE_DIR}')

if __name__ == "__main__":
    # working()
    # projection_every_step() # too slow...
    save_folder = f'{output_folder}'


    num_rods = 2
    alpha = 100
    rod_diameter = 1/alpha
    container_size = 1

    from protocols import create_entangled_rods
    import jax

    random_keys = jax.random.PRNGKey(11)

    from potentials import total_effective_potential, total_harmonic_line

    # simple grad descent

    grad_fn = jax.jit(jax.grad(total_effective_potential))

    
    col_rad = rod_diameter / 2
    amp = 1
    params = {'col_rad': col_rad,
              'amp': amp}

    grad_fn2 = jax.jit(jax.grad(lambda x: total_harmonic_line(x, params)))


    import polyscope as ps
    from transforms import q_to_x
    from visualizations import prep_for_polyscope

    



    q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
    q = q0.copy()


    ps.init()
    ps.set_autoscale_structures(False)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_ground_plane_mode("none")

    a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
    nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)
    min_z = onp.min(nodes[:, 2])
    ps_curves = ps.register_curve_network( "filaments", nodes, edges )
    ps_curves.add_color_quantity( "edge_colors", edge_colors, defined_on='edges', enabled=True )
    ps_curves.set_radius( rod_diameter / 2, relative=False )

    ps.set_length_scale(2.)
    sz = 2.
    low = onp.array((-sz, -sz, -sz))
    high = onp.array((sz, sz, sz))
    ps.set_bounding_box(low, high)
    ps.set_up_dir("z_up")

    nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)

    from potentials import dist_lin_seg_vector

    k = 0
    step_size = 1e-1
    for _ in range(3000):

        step_size = step_size 
        
        grad = grad_fn(q)
        q = q - step_size * grad

        # project a bit
        
        for __ in range(100):
            grad2 = grad_fn2(q)
            q = q - 1e-3 * grad2

        # save every 10 steps
        if _ % 100 == 0:

            x = q_to_x(q).reshape(2,-1,3)
            p1s = x[0,0]
            p1e = x[0,1]
            p2s = x[1,0]
            p2e = x[1,1]
            t,u,r1,r2,r12,d = dist_lin_seg_vector(p1s, p1e, p2s, p2e)

            print(f't: {t}, u: {u}, dist: {d}')

            a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
            ps_curves.update_node_positions(a_list_of_curves.reshape(-1,3))
            # ps_curves.get_color_quantity("edge_colors").update_values(edge_colors)
            pth = f"{MOVIE_DIR}/step-{k:04d}.png"
            ps.screenshot(str(pth))
            
            k += 1


    step_size = 0.1
    for _ in range(10000):

        
        
        grad = grad_fn(q)
        q = q - step_size * grad

        # project a bit
        
        for __ in range(30):
            grad2 = grad_fn2(q)
            q = q - 1e-3 * grad2

        # save every 10 steps
        if _ % 100 == 0:
            x = q_to_x(q).reshape(2,-1,3)
            p1s = x[0,0]
            p1e = x[0,1]
            p2s = x[1,0]
            p2e = x[1,1]
            t,u,r1,r2,r12,d = dist_lin_seg_vector(p1s, p1e, p2s, p2e)

            ori1 = p1e - p1s
            ori2 = p2e - p2s

            # angle
            cos_angle = jnp.dot(ori1, ori2) / (jnp.linalg.norm(ori1) * jnp.linalg.norm(ori2))
            angle = jnp.arccos(cos_angle) * 180 / jnp.pi 

            print(f't: {t}, u: {u}, dist: {d}')
            print(f'angle between rods: {angle}')

            a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
            ps_curves.update_node_positions(a_list_of_curves.reshape(-1,3))
            # ps_curves.get_color_quantity("edge_colors").update_values(edge_colors)
            pth = f"{MOVIE_DIR}/step-{k:04d}.png"
            ps.screenshot(str(pth))
            
            k += 1

    # q_ent = create_entangled_rods(num_rods,total_effective_potential,random_keys,rod_diameter=rod_diameter,Nmax=1e4,N_outer=1,atol=1e-4,dt=1e-3,initial_q=None,callback=None)

    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    x = q_to_x(q)
    r1 = x.reshape(-1, 6)[:,:3]
    r2 = x.reshape(-1, 6)[:,3:]

    dist_matrix = jnp.linalg.norm(r1[:,None,:] - r2[None,:,:], axis=-1)
    print(f'Distance matrix between rods: {dist_matrix}')
    


    # x_ent = q_to_x(q_ent)
    # r1 = x_ent.reshape(-1, 6)[:,:3]
    # r2 = x_ent.reshape(-1, 6)[:,3:]
    # acn_matrix = acn_over_ij(r1, r2, i_indices, j_indices)
    # print(f'Average crossing number for entangled rods: {jnp.sum(jnp.abs(acn_matrix)) }')



       
    # BOUNDARY CONDITION
    # q0 = create_nonintersecting_random_rods_contained_centroids(num_rods,rod_diameter, container_size)
    # q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
    # # q0 = create_nonintersecting_random_rods_contained(num_rods,rod_diameter, container_size)
    
    
    # from potentials import dist_lin_seg_over_ij, all_pairwise_distances_xyz, create_pairs, dist_lin_seg_pbc_over_ij
    # from potentials import all_pairwise_distances_xyz_pbc, acn_over_ij

    # x0 = q_to_x(q0)
    # i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    # r1 = x0.reshape(-1, 6)[:,:3]
    # r2 = x0.reshape(-1, 6)[:,3:]
    # d0 = dist_lin_seg_over_ij(r1,r2,i_indices,j_indices)
    # d1 = dist_lin_seg_pbc_over_ij(r1,r2, container_size, i_indices,j_indices)

    # print(f'min dist without pbc: {jnp.min(d0)}, min dist with pbc: {jnp.min(d1)}')

    # acn_list = []
    # container_size_list = onp.geomspace(1,100,30)
    # for i in range(len(container_size_list)):
    #     container_size = container_size_list[i]
    #     q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
    #     x0 = q_to_x(q0)
    #     i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    #     r1 = x0.reshape(-1, 6)[:,:3]
    #     r2 = x0.reshape(-1, 6)[:,3:]

    #     acn_matrix = acn_over_ij(r1, r2, i_indices, j_indices)
    #     print(f'Average crossing number for container size={container_size}: {jnp.sum(jnp.abs(acn_matrix)) }')
    #     acn_list.append(jnp.sum(jnp.abs(acn_matrix)))


    #     x = q_to_x(q0)
    #     # onp.savetxt(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale1.txt',x)

    #     ax = plot_many_rods(q0.reshape(-1,5))
    #     # flat perspective
    #     ax.view_init(elev=0, azim=-90)
    #     ax.set_xlim([-container_size/2*3,container_size/2*3])
    #     ax.set_ylim([-container_size/2*3,container_size/2*3])
    #     ax.set_zlim([-container_size/2*3,container_size/2*3])

    #     plt.savefig(f'{save_folder}/Rods-N{num_rods}-AR{alpha}-Scale{container_size}.png',dpi=300)
    #     plt.close('all')

    # plt.loglog(container_size_list, acn_list, marker='o')
    
    # upper_bound = num_rods * (num_rods - 1) / 2 * 0.5
    # plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
    
    # plt.xlabel('Container Size')
    # plt.ylabel('Average Crossing Number (ACN)')
    # plt.title(f'ACN vs Container Size for {num_rods} Rods (AR={alpha})')
    # plt.savefig(f'{save_folder}/ACN_vs_ContainerSize_N{num_rods}_AR{alpha}.png', dpi=300)
    # plt.clf()
        
# %%

    

    
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