# %%
from protocols import create_random_rods,create_nonintersecting_random_rods
from protocols import create_intersecting_random_rods_contained_in_noncube
from protocols import create_intersecting_random_rods_uncontained

import jax.numpy as jnp
import matplotlib.pyplot as plt
from transforms import q_to_x
from potentials import acn_over_ij
# %%
if __name__ == "__main__":
    num_rods_list = jnp.geomspace(10,2000,20).astype(int)

    q_list = []
    acn_list = []
    for num_rods in num_rods_list:
        # num_rods = 2000
        random_keys = [0,0,0]
        # centroid within a box
        # uniformly.
        container_size = jnp.array([1.,1.,1.])
        # q0 = create_nonintersecting_random_rods(num_rods, rod_diameter=1e-20, max_attempts=10000)
        # q0 = create_intersecting_random_rods_contained_in_noncube(num_rods, None, container_size, max_attempts=1000000)
        q0 = create_intersecting_random_rods_uncontained(num_rods, max_attempts=1000000)
        q_list.append(q0)
        
        x0 = q_to_x(q0)
        x0 = x0.reshape((-1,6))
        plt.figure()
        plt.subplot(111, projection='3d')
        for i in range(x0.shape[0]):
            plt.plot([x0[i,0], x0[i,3]], [x0[i,1], x0[i,4]], [x0[i,2], x0[i,5]])        
        plt.title(f'Random rods, num_rods={num_rods}')
        plt.savefig(f'random_rods_entanglement/random_rods_{num_rods}.png')
        plt.show()

        
            
        r1 = x0[:,0:3]
        r2 = x0[:,3:6]
        i_indices,j_indices = jnp.triu_indices(x0.shape[0],k=1)
        acn_ij = acn_over_ij(r1, r2, i_indices, j_indices)
        acn_list.append(acn_ij)

# %%
    for acn_ij in acn_list:
        plt.hist(jnp.abs(acn_ij),bins=50, alpha=0.5, density=True)
        # plt.yscale('log')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e-1,1)
    plt.xlabel('Absolute ACN values')
    plt.ylabel('Density')
    plt.title('Histogram of absolute ACN values for different numbers of rods')
    
# %%
    sum_acn_list = []
    normalized_acn_list = []
    max_acn_list = []
    normalized_by_num_rods = []
    for acn_ij,num_rods in zip(acn_list,num_rods_list):
        sum_acn = jnp.sum(jnp.abs(acn_ij))
        sum_acn_list.append(sum_acn)
        normalized_by_num_rods.append(sum_acn/num_rods)
        normalized_acn_list.append(sum_acn/(num_rods*(num_rods-1)/2))
        max_acn_list.append(jnp.max(jnp.abs(acn_ij)))
# %%
    plt.figure()
    plt.loglog(num_rods_list, sum_acn_list, marker='o')
    plt.xlabel('Number of rods')
    plt.ylabel('Pairwise sum of ACN')
    
# %%
    plt.figure()
    plt.plot(num_rods_list,normalized_acn_list, marker='o')
    plt.xlabel('Number of rods')
    plt.ylabel('Sum ACN / No. Rods^2 ')
# %%
    plt.figure()
    plt.plot(num_rods_list,normalized_by_num_rods, marker='o')
    plt.xlabel('Number of rods')
    plt.ylabel('Sum ACN / No. Rods')        
# %%
    plt.figure()
    plt.plot(num_rods_list,max_acn_list, marker='o')
    plt.xlabel('Number of rods')
    plt.ylabel('Max ACN')
# %%
