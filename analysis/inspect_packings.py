import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from potentials import acn_over_ij, dist_lin_seg_over_ij
from matplotlib import pyplot as plt


# pth = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/MaxEnt3')
# for f in pth.iterdir():
#     print(f)

def inspect_folder(folder_path):
    return 0

def inspect_file(file_path):
    rr = np.loadtxt(file_path, delimiter=' ')
    num_rods = rr.shape[0]
    r1 = rr[:,:3]
    r2 = rr[:,3:]    

    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    pairwise_acn = acn_over_ij(r1, r2, i_indices, j_indices)
    pairwise_dist = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
    dist = np.min( pairwise_dist )
    print(dist)

    entanglement = np.sum(np.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
    print(entanglement)

    fig,ax=plt.subplots(1,1,subplot_kw= {'projection':'3d'})
    for i in range(num_rods):
        ax.plot([r1[i,0], r2[i,0]], [r1[i,1], r2[i,1]], [r1[i,2], r2[i,2]], 'k')
    plt.show()
    
    return 0

if __name__ == '__main__':
    
    # file_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/MaxEnt3/29,19,70/MaxEnt29,19,70-N200-AR0500-Scale1.txt'
    # file_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/MaxEnt3/46,15,99/MaxEnt46,15,99-N200-AR0500-Scale1.txt'
    # file_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/MaxEnt3/52,33,20/MaxEnt52,33,20-N200-AR0500-Scale1.txt'

    pathlist = []
    pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/37,178,56')
    pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/6,7,8')
    pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/919,461,568')

    for pth in pathlist:
        for f in os.listdir(pth):
            if f.endswith('.txt'):
                file_path = os.path.join(pth, f)
                print(file_path)
                # inspect_file(file_path)


    # inspect_file(file_path)

    # folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/MaxEnt3/33,31,94'
    # inspect_folder(folder_path)