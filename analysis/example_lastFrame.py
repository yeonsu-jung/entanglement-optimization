import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os
import re

def parse_path_string(pth):
    filename = pth.split('/')[-1]        
    file_id = filename.split('-mu')[0]
    
    surfix_match = re.search(r'\d{8}-\d{6}', filename)
    surfix = surfix_match.group(0) if surfix_match else None
    
    num_rods_match = re.search(r'-N(\d+)-', filename)
    num_rods = int(num_rods_match.group(1)) if num_rods_match else None
    
    return file_id, surfix, num_rods


root_path = Path('/Users/yeonsu/Data/from_cluster/')

folders = list(root_path.glob('*Ted,*'))
    
pathlist = []
for folder in folders:
    csv_files = list(folder.glob('*lastFrame.csv'))
    pathlist.append(csv_files[0])

outpth_root = Path('/Users/yeonsu/GitHub/dismech-rods-main/data/Ted')
if os.path.exists(outpth_root) == False:
    os.mkdir(outpth_root)
        
for pth in pathlist:
    with open(pth) as f:
        line = f.readline()
        end_time = float(line.split(',')[0])
        
    file_id,surfix,num_rods = parse_path_string(str(pth))

    dta = np.loadtxt(pth, delimiter=',',skiprows=1)
    
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(dta.shape[0]):
        rr = dta[i,:].reshape(-1,3)
        ax.plot(rr[:,0],rr[:,1],rr[:,2])
    plt.savefig(outpth_root / f'{file_id}-{surfix}-{end_time:.2f}.png')
    plt.close()
    
    savepth = outpth_root / f'{file_id}-{surfix}-{end_time:.2f}.txt'
    np.savetxt(savepth, dta, delimiter=' ')

print