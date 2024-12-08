# %%
import numpy as np
import protocols
# %%
num_rods = 9
AR = 25
rod_diameter = 1/AR
container_size = 0.5


container_size = np.array([0.6,0.6,1])
q0 = protocols.create_nonintersecting_random_rods_contained_in_noncube(num_rods=num_rods,
                                                                rod_diameter=rod_diameter,
                                                                container_size=container_size,
                                                                max_attempts=1000000)
    
# %%
from transforms import q_to_x
x0 = q_to_x(q0)

# %%
from matplotlib import pyplot as plt

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in range(num_rods):
    r1 = x0[i,:3]
    r2 = x0[i,3:]
    ax.plot([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]])

np.savetxt(f'nine_rods-N{num_rods:02d}-AR{AR:03d}-Scale1.txt',x0,delimiter=' ')

# %%

