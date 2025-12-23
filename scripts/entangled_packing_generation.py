import sys
import os
from pathlib import Path
import numpy as onp
from jax import numpy as jnp



sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

from potentials import total_effective_potential
from transforms import q_to_x

output_dir = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/scripts/outputs'
    

from protocols import create_entangled_rods

# q_entangled = create_entangled_rods(num_rods=200, rod_length=1.0, rod_diameter=0.01, box_size=1.5, entanglement_factor=5, max_attempts=10000)
# np.savetxt('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/python/results/q_entangled_packing.txt', q_entangled)


import numpy as np
import datetime
N_outer = 1
Nmax = 100000
scale_factor = 1
num_rods = 500
dt = 1.e-2
amp = 100

random_keys = [56,321,194]
# random_keys = [6,7,8]
# random_keys = [37,178,56]
# random_keys = [919,461,568]


output_folder = f'{output_dir}/entangled_packings/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results_per_random_keys = f'{output_folder}/{random_keys[0]},{random_keys[1]},{random_keys[2]}'

if not os.path.exists(results_per_random_keys):
    os.makedirs(results_per_random_keys)

now = datetime.datetime.now()

for num_rods in [500]:

    save_dir_name = f'{results_per_random_keys}/N{num_rods}'
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name,exist_ok=True)

    # rod_diameter = 1/AR
    # params = {"col_rad": rod_diameter/2, "amp": 1., "sigma": 0.025, AR: AR}

    dt_string = now.strftime("%Y-%m-%d_%H")  
    
    # dt_string = "2024-10-16_00"
    scale_factor = 1
    packing_id = f'{dt_string}_EntangledPacking-N{num_rods:04d}-Scale{scale_factor}'

    if not os.path.exists(f"{results_per_random_keys}/{packing_id}"):
        os.makedirs(f"{results_per_random_keys}/{packing_id}",exist_ok=True)

    filename = f"{results_per_random_keys}/{packing_id}/qq.npy"
    if os.path.exists(filename):
        qq = onp.load(filename)
    else:
        onp.save(filename,[])
        qq = []

    # if os.path.exists(f'{results_per_random_keys}/N{num_rods}/q_entangled.npy'):
    #     q_entangled = np.load(f'{results_per_random_keys}/N{num_rods}/q_entangled.npy')
    #     print("Loaded existing entangled configuration")

    if 0:
        print("Loading existing entangled configuration")

    else:

        os.makedirs(f'{results_per_random_keys}/N{num_rods}',exist_ok=True)
        # q_entangled = create_entangled_rods(num_rods,total_effective_potential,random_keys,rod_diameter=(1/AR),Nmax=300,N_outer=5,atol=1e-8,dt=dt,initial_q="non-intersecting",callback=_callback)
        # q_entangled = create_entangled_rods(num_rods,total_effective_potential,random_keys,rod_diameter=(1/AR),Nmax=300,N_outer=5,atol=1e-8,dt=dt,initial_q=None,callback=_callback)
        # todo: remove rod diameter from the argument
        q_entangled = create_entangled_rods(num_rods,total_effective_potential,random_keys,rod_diameter=-1,Nmax=300,N_outer=5,atol=1e-8,dt=dt,initial_q="gathered",callback=None)

        # initial_q == "gathered"
        # q_entangled = create_entangled_rods(num_rods,total_effective_potential,random_keys,rod_diameter=(1/AR),Nmax=300,N_outer=5,atol=1e-8,dt=dt,initial_q="gathered",callback=_callback)
        np.save(f'{results_per_random_keys}/N{num_rods}/q_entangled.npy',q_entangled)
        
        x_entangled = q_to_x(jnp.array(q_entangled))
        np.savetxt(f'{results_per_random_keys}/N{num_rods}/x_entangled_packing.txt', x_entangled.reshape(-1,6))


    if len(qq) == 0:
        qq = np.array([q_entangled])

    q0 = qq.reshape(-1,num_rods,5)
    
    from transforms import q_to_x
    x0 = q_to_x(jnp.array(q0))
    np.savetxt(f'{results_per_random_keys}/{packing_id}/x_entangled_packing.txt', x0.reshape(-1,6))
    
    
