# %%
import numpy as np
from protocols import create_nonintersecting_random_rods_contained_in_noncube

num_rods = 200
rod_diameter = 0.01
container_size = np.array([1,1,2])

# (num_rods, rod_diameter, container_size, max_attempts=1000000):
q = create_nonintersecting_random_rods_contained_in_noncube(num_rods, rod_diameter, container_size, max_attempts=1000000)

# %%
from transforms import q_to_x

x = q_to_x(q)
# %%
from matplotlib import pyplot as plt
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in range(num_rods):
    x1 = x[i,0:3]
    x2 = x[i,3:6]
    ax.plot([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]])
# %%
np.savetxt(f'random_nonintersecting_numRods{num_rods}.txt',x)
