# %%
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from pathlib import Path
from matplotlib.patches import Polygon
from data_io import import_all_log            
# %
def find_folders(data_root,which_AR):
    # find subdirs in a directory which contains the data of a specific AR
    for subdirs in data_root.iterdir():
        search_result = re.search(r'AR(\d+)',subdirs.stem)
        if search_result is None:
            continue
        
        AR = int(search_result.group(1))
        
        if AR == which_AR:
            print(subdirs)
            a_data_container = subdirs
            break
    
def get_crop_range(img_path):
    img = plt.imread(img_path)
    gray_img = np.sum(img,axis=2)

    rows = np.sum(gray_img,axis=0)
    cols = np.sum(gray_img,axis=1)
    rows = rows.max() - rows
    cols = cols.max() - cols
    
    # get crop range
    row_start = np.argmax(rows > 0)
    row_end = len(rows) - np.argmax(rows[::-1] > 0)
    col_start = np.argmax(cols > 0)
    col_end = len(cols) - np.argmax(cols[::-1] > 0)

    cropped = img[col_start:col_end,row_start:row_end,:]
    # plt.imsave(img_path,cropped)
    
    return (col_start,col_end,row_start,row_end)

def _draw(a_data_container,savepath):
    for files in a_data_container.iterdir():
        if files.suffix == '.csv':
            if 'last' not in files.stem:
                a_csv = files
    
    time_line, node_list, contact_list = import_all_log(a_csv,max_rows=100000)
    nodes_in_matrix = node_list[0].reshape((-1,30))
    packing_center = np.mean(np.mean(nodes_in_matrix.reshape((-1,10,3)),axis=1),axis=0)
    
    AR = re.search(r'AR(\d+)',a_data_container.stem).group(1)
    rod_diameter = 1/float(AR)*100
    line_thickness = rod_diameter
        
    for i_time in [0,-1]:
        timestamp = time_line[i_time]
        nodes_in_matrix = node_list[i_time].reshape((-1,30))

        locked_nodes = []
        for rr in nodes_in_matrix.reshape((-1,10,3)):
            I = np.linalg.norm(rr - packing_center,axis=1) < 0.1
            # ax.plot(rr[I,0],rr[I,1],rr[I,2],'k-')
            locked_nodes.append(rr[I,:])
                
                
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        for node in nodes_in_matrix:
            rr = node.reshape((-1,3))
            ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=line_thickness)
            
        # for rr in locked_nodes:
        #     ax.plot(rr[:,0],rr[:,1],rr[:,2],'k-',linewidth=line_thickness*2)

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1.5,0.5)
        # no numbers
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(0,0)
        plt.savefig(f'{savepath}/packing_{timestamp}.png',dpi=300)
        plt.close()
    
    

# %%
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo2'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo3'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo2_FineExcitation'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo3_FineExcitation'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo2_SlowExcitation'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo3_SlowExcitation'

'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation'
'/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation'
# %%


 # %%
data_containers = []
data_containers.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0250_AR050'))
data_containers.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0500_AR100'))
data_containers.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1755_RUN_HangEEModelo1_N1500_AR300'))

output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/prunedData/rod-sim-pnas-revision/Visuals'
if not os.path.exists(output_root):
    os.makedirs(output_root)

for data_container in data_containers:
    savepath = f'{output_root}/{data_container.stem}'
    os.makedirs(savepath,exist_ok=True)
    _draw(Path(data_container),savepath)
    
# %%
parent_folders = []
parent_folders.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1')
parent_folders.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation')
parent_folders.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation')

for parent_folder in parent_folders:
    pth = Path(parent_folder)
    
    wanted_ARs =  [50, 100, 300]
    for data_container in pth.rglob('*'):
        search_result = re.search('N(\d+)_AR(\d+)',str(data_container.stem))
        if search_result is None:
            continue
        num_rods = int(search_result.group(1))
        AR = int(search_result.group(2))

        if AR in wanted_ARs:
            savepath = f'{output_root}/{data_container.stem}'
            os.makedirs(savepath,exist_ok=True)
            _draw(data_container,savepath)



# %%










# %%
imgpath = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0250_AR050/packing_0.0.png'
crop_range = get_crop_range(imgpath)
img = plt.imread(imgpath)
col_start,col_end,row_start,row_end = crop_range
cropped = img[col_start:col_end,row_start:row_end,:]
plt.imshow(cropped)

# %%
imgpaths = []
imgpaths.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0500_AR100/packing_5.0.png')
imgpaths.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0500_AR100/packing_0.0.png')
imgpaths.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0250_AR050/packing_0.0.png')
imgpaths.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0250_AR050/packing_5.0.png')
imgpaths.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1755_RUN_HangEEModelo1_N1500_AR300/packing_0.0.png')
imgpaths.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1755_RUN_HangEEModelo1_N1500_AR300/packing_5.0.png')

col_start,col_end,row_start,row_end = crop_range
for imgpath in imgpaths:
    img = plt.imread(imgpath)    
    cropped = img[col_start:col_end,row_start:row_end,:]
    plt.imsave(imgpath,cropped)

# %%
image_containers = [] 
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240609-1052_RUN_PerturbEEModelo1_N0250_AR050')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240609-1052_RUN_PerturbEEModelo1_N0500_AR100')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1755_RUN_HangEEModelo1_N1500_AR300')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0250_AR050')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240610-1802_RUN_HangEEModelo1_N0500_AR100')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240611-1241_RUN_WeakPerturbEEModelo1_N0250_AR050')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240611-1241_RUN_WeakPerturbEEModelo1_N0500_AR100')
image_containers.append('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Visuals/20240611-1241_RUN_WeakPerturbEEModelo1_N1500_AR300')

# %%
for image_container in image_containers:
    imgpaths = []
    for files in Path(image_container).iterdir():
        if files.suffix == '.png':
            imgpaths.append(files)   
    
    for imgpath in imgpaths:
        img = plt.imread(imgpath)    
        cropped = img[col_start:col_end,row_start:row_end,:]
        plt.imsave(imgpath,cropped)
