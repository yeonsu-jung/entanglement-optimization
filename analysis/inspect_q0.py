# %%
import numpy as np
q0 = np.load('/Users/yeonsu/GitHub/entanglement-optimization/q0.npy')

# %%
from potentials import total_effective_potential
e0 = total_effective_potential(q0)
print(e0)
# %%
from visualizations import plot_many_rods

plot_many_rods(q0.reshape(-1,5))