# %%
from data_io import import_all_log
data_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1500_AR300_freq10/NonIntersectingBox-N1500-AR300-Scale1-mu0.50-visc0.00-amp0.00_allLog_20240722-033112.csv'
dta = import_all_log(data_path,max_rows=10000000)
# %%
time_points = dta[0]
nodes_list = dta[1]
# %%
import numpy as np

max_len_list = []
min_len_list = []
for nodes in nodes_list:
    rr = nodes.reshape(-1,6)
    r1 = rr[:,:3]
    r2 = rr[:,3:]
    length_list = np.linalg.norm(r1-r2,axis=1)
    max_len_list.append(np.max(length_list))
    min_len_list.append(np.min(length_list))
    
# %%
from matplotlib import pyplot as plt
plt.plot(time_points,max_len_list)
plt.plot(time_points,min_len_list)
plt.xlabel('time')
plt.ylabel('length')

