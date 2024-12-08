# %%
import protocols
import datetime
import numpy as np
num_rods = 3
AR = 25
dt_string = '2024-07-24'
N_outer = 100
Nmax = 100
scale_factor = 1
x = protocols.create_entrel_packing(num_rods,AR,dt_string,N_outer,Nmax,scale_factor)
# %%
