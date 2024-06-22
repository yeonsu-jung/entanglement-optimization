# %%
from pathlib import Path
import numpy as np
parent_folder = '/Users/yeonsu/GitHub/dismech-rods-main/data/CalmModelo1'
output_root = '/Users/yeonsu/GitHub/dismech-rods-main/data/FlippedCalmModelo1'

import os

if not os.path.exists(output_root):
    os.makedirs(output_root)
# %%
for each_file in Path(parent_folder).rglob('*.csv'):
    print(each_file)
    
    dta = np.loadtxt(each_file,delimiter=' ')
    
    xyz = dta.reshape((-1,10,3))
    
    xzy = np.zeros_like(xyz)
    xzy[:,:,0] = xyz[:,:,0]
    xzy[:,:,1] = xyz[:,:,2]
    xzy[:,:,2] = xyz[:,:,1]
    
    np.savetxt(f'{output_root}/{each_file.stem}.csv',xzy.reshape((-1,30)),delimiter=' ')
# %%
from matplotlib import pyplot as plt
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for _i in range(len(xzy)):
    ax.plot(xzy[_i,:,0],xzy[_i,:,1],xzy[_i,:,2])
    
ax.view_init(0,0)
    
# %%

    
# %%
