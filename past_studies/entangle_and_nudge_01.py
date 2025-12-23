import sys
sys.path.append('../core')  # to import from parent folder

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


    num_rods = 20
    alpha = 100
    rod_diameter = 1/alpha
    container_size = 1

    from protocols import create_entangled_rods
    import jax

    random_keys = jax.random.PRNGKey(11)

    from potentials import total_effective_potential, total_harmonic_line, dist_lin_seg_over_ij

    # simple grad descent

    grad_fn = jax.jit(jax.grad(total_effective_potential))

    
    col_rad = rod_diameter / 2
    amp = 100
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

    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)

    k = 0
    step_size = 1e-3
    qq = []
    for _ in range(10000):

        # step_size = step_size * 0.9995
        
        grad = grad_fn(q)
        q = q - step_size * grad

        # project a bit
        
        for __ in range(1000):
            grad2 = grad_fn2(q)
            q = q - step_size * grad2

            x = q_to_x(q)
            r1 = x.reshape(-1, 6)[:,:3]
            r2 = x.reshape(-1, 6)[:,3:]

            dist_mat = dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
            min_dist = jnp.min(dist_mat)
            if min_dist > rod_diameter * 0.99:
                break
        qq.append(q)
        # save every 10 steps
        if _ % 10 == 0:

            # x = q_to_x(q).reshape(2,-1,3)
            # p1s = x[0,0]
            # p1e = x[0,1]
            # p2s = x[1,0]
            # p2e = x[1,1]
            # t,u,r1,r2,r12,d = dist_lin_seg_vector(p1s, p1e, p2s, p2e)

            # print(f't: {t}, u: {u}, dist: {d}')

            
            x = q_to_x(q)
            r1 = x.reshape(-1, 6)[:,:3]
            r2 = x.reshape(-1, 6)[:,3:]

            dist_mat = dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
            min_dist = jnp.min(dist_mat)
            print(f'step: {_}, min dist: {min_dist}, step size: {step_size}')
            

            a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
            ps_curves.update_node_positions(a_list_of_curves.reshape(-1,3))
            # ps_curves.get_color_quantity("edge_colors").update_values(edge_colors)
            pth = f"{MOVIE_DIR}/step-{k:04d}.png"
            ps.screenshot(str(pth))
            
            k += 1
    # save qq
    onp.save(f"{output_folder}/qq.npy", onp.array(qq))
# %%    


# %%
