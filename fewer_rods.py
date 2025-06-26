# %%
from protocols import create_entangled_rods, create_aligned_rods
from potentials import total_effective_potential, total_angle_repulsion
num_rods = 10
random_keys = [0,1,2]
Nmax = 3000
# %%
q = create_entangled_rods(num_rods,total_effective_potential,random_keys,Nmax=Nmax,initial_q=None)
# %%
total_effective_potential(q)
# %%
from visualizations import plot_many_rods,set_3d_plot
fig,ax=set_3d_plot()
plot_many_rods(q.reshape(-1,5),ax=ax)
# %%
