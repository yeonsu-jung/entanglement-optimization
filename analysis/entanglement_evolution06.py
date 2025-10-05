# %%
from protocols import create_nonintersecting_random_rods, create_nonintersecting_random_rods_contained
from protocols import create_nonintersecting_random_rods_contained_centroids, create_nonintersecting_random_rods_contained_pbc
from transforms import q_to_x
import numpy as onp
import jax.numpy as jnp

import matplotlib.pyplot as plt
from visualizations import plot_many_rods

from protocols import create_entangled_rods
import jax
from potentials import total_effective_potential, total_harmonic_line, dist_lin_seg_over_ij
import polyscope as ps
from transforms import q_to_x
from visualizations import prep_for_polyscope
from potentials import dist_lin_seg_vector, acn_over_ij
import os
from pathlib import Path

# get current folder and make output folder named the current folder's name
# current filename
current_file = os.path.basename(__file__)
filename = os.path.splitext(current_file)[0]
output_folder = f'{filename}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f'Output folder: {output_folder}')



output_folder = Path(output_folder)
MOVIE_DIR = f"{output_folder}/movie"
if not os.path.exists(MOVIE_DIR):
    os.makedirs(MOVIE_DIR)
    print(f'Movie folder: {MOVIE_DIR}')

if __name__ == "__main__":
    # working()
    # projection_every_step() # too slow...
    save_folder = f'{output_folder}'

    num_rods = 100
    alpha = 100
    rod_diameter = 1/alpha
    container_size = 1

    pathlist = []
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-05_03_EntangledRelaxedPacking-N0100-AR0050-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-04_18_EntangledRelaxedPacking-N0100-AR0075-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-04_18_EntangledRelaxedPacking-N0100-AR0100-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-04_18_EntangledRelaxedPacking-N0100-AR0200-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-04_18_EntangledRelaxedPacking-N0100-AR0300-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-04_18_EntangledRelaxedPacking-N0100-AR0500-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-05_03_EntangledRelaxedPacking-N0100-AR0010-Scale1')
    pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-05_03_EntangledRelaxedPacking-N0100-AR0020-Scale1')

    # sort by AR
    pathlist = sorted(pathlist, key=lambda x: int(x.split('AR')[1].split('-')[0]))

    ii,jj=jnp.triu_indices(num_rods,k=1)

    ent_list = []
    for pth in pathlist:
        file_path = f'{pth}/qq.npy'
        q = onp.load(file_path).reshape(-1,num_rods,5)        
        xf = q_to_x(q[-1]).reshape(num_rods,2,3)

        r1 = xf[:,0,:]
        r2 = xf[:,1,:]

        dij = dist_lin_seg_over_ij(r1,r2,ii,jj)
        dij = jnp.array(dij)
        print(f"min distance: {dij.min()}")

        aij = acn_over_ij(r1,r2,ii,jj)
        aij = jnp.array(aij)

        entanglement = jnp.sum( jnp.abs(aij) )
        print(f"entanglement: {entanglement}")

        ent_list.append(entanglement)


# %%
AR_list = [10,20,50,75,100,200,300,500]
AR_list = jnp.array(AR_list)
plt.loglog(AR_list,ent_list,'o-')

# %%


# %%

