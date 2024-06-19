# %%

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle
import networkx as nx

from data_io import import_all_log, parse_path_string    
from scipy.optimize import curve_fit
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1731_RUN_HangModelo1_N0500_AR100_g1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1732_RUN_HangModelo1_N0500_AR100_g2')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1732_RUN_HangModelo1_N0500_AR100_g3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1732_RUN_HangModelo1_N0500_AR100_g4')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1732_RUN_HangModelo1_N0500_AR100_g5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a02')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a05')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a25')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a50')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_HangModelo1_N0125_AR025_g1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_HangModelo1_N0125_AR025_g2')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_HangModelo1_N0125_AR025_g3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_HangModelo1_N0125_AR025_g4')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_HangModelo1_N0125_AR025_g5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a02')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a05')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a25')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a50')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240610-1802_RUN_HangEEModelo1_N0500_AR100_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240610-1802_RUN_HangEEModelo1_N0250_AR050_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240610-1755_RUN_HangEEModelo1_N1500_AR300_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240610-1802_RUN_HangEEModelo1_N0125_AR025_g0.5')

pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240609-1052_RUN_TickleEEModelo1_N1500_AR300_a100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240609-1052_RUN_TickleEEModelo1_N0500_AR100_a100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240609-1051_RUN_TickleEEModelo1_N0125_AR025_a100')

pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1618_RUN_HeavyHangEEModelo1_N1500_AR300_g5')


# %%
class data_container:
    def __init__(self,dataphat,max_rows=100000):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,max_rows=max_rows)
        
max_rows = 1000000
data_container_list = []
for pth in pathlist:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        data_container_list.append(data_container(file,max_rows=max_rows))
        break

# %%
for dc in data_container_list:
    search_result = re.search('Hang',str(dc.path))
    if search_result:
        # dc.model = 'Hang'
        dc.model = 'Hang'
        continue
    
    search_result = re.search('Tickle',str(dc.path))
    if search_result:
        # dc.model = 'Tickle'
        dc.model = 'Tickle'
        continue
    
    dc.model = 'Unknown'
# %%
for dc in data_container_list:
    print(dc.model)
    
# %%
for dc in data_container_list:
    if dc.model == 'Hang':        
        # dc.g = float(re.search('N(\d+)[-_]AR(\d+)_g(\d+)',str(dc.path)).group(3))
        dc.g = float(re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)', str(dc.path)).group(3))
    elif dc.model == 'Tickle':
        dc.a = float(re.search('N(\d+)[-_]AR(\d+)_a(\d+)',str(dc.path)).group(3))
        
# %%
for dc in data_container_list:
    if dc.model == 'Hang':
        print(dc.g)
    elif dc.model == 'Tickle':
        print(dc.a)
        
# %%
fig,ax=plt.subplots()

static_load_data_dict = {'100': [],
                        '25': [],
                        '300': []}

g_data_dict = {'100': [],
             '25': [],
             '300': []}

a_data_dict = {'100': [],
             '25': [],
             '300': []}

for dc in data_container_list:
    if dc.model == 'Hang':
        contact_ij = dc.contact_list[-1].reshape(-1,18)[:,4:6].astype(int)
        contact_graph = nx.Graph()
        contact_graph.add_nodes_from(range(dc.num_rods))
        contact_graph.add_edges_from(contact_ij)
        clusters = list(nx.connected_components(contact_graph))        
        # largest clusters
        largest_cluster = max(clusters,key=len)
        dta = len(largest_cluster)/dc.num_rods
        
        rho = 7e3        
        rod_length = 1
        rod_diameter = 1/dc.AR
        
        if dc.AR == 25:
            g_data_dict['25'].append((dc.g,dta))
            
        elif dc.AR == 100:
            g_data_dict['100'].append((dc.g,dta))
            
        elif dc.AR == 300:
            g_data_dict['300'].append((dc.g,dta))
            
            
    if dc.model == 'Tickle':
        contact_ij = dc.contact_list[-1].reshape(-1,18)[:,4:6].astype(int)
        contact_graph = nx.Graph()
        contact_graph.add_nodes_from(range(dc.num_rods))
        contact_graph.add_edges_from(contact_ij)
        clusters = list(nx.connected_components(contact_graph))        
        # largest clusters
        largest_cluster = max(clusters,key=len)
        dta = len(largest_cluster)/dc.num_rods
        
        if dc.AR == 25:
            a_data_dict['25'].append((dc.a,dta))
        elif dc.AR == 100:
            a_data_dict['100'].append((dc.a,dta))
        elif dc.AR == 300:
            a_data_dict['300'].append((dc.a,dta))

# %%
single_column_size = (2.5,1.75)
output_dir = f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/entanglement'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

denom = rho*rod_diameter**2*rod_length

fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(1/0.5*np.array(g_data_dict['25'])[:,0],np.array(g_data_dict['25'])[:,1],'o',label='AR=25')
plt.plot(1/0.5*np.array(g_data_dict['100'])[:,0],np.array(g_data_dict['100'])[:,1],'o',label='AR=100')
plt.plot(1/0.5*np.array(g_data_dict['300'])[:,0],np.array(g_data_dict['300'])[:,1],'o',label='AR=300')
plt.xlabel(r'$F/(\rho g d^2 l)$')
plt.ylabel(r'$r$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_g.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(np.array(a_data_dict['25'])[:,0],np.array(a_data_dict['25'])[:,1],'o',label='AR=25')
plt.plot(np.array(a_data_dict['100'])[:,0],np.array(a_data_dict['100'])[:,1],'o',label='AR=100')
plt.plot(np.array(a_data_dict['300'])[:,0],np.array(a_data_dict['300'])[:,1],'o',label='AR=300')
plt.xlabel(r'$a$ (m/s$^2$)')
plt.ylabel(r'$r$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_a.png',dpi=300,bbox_inches='tight')


# %%
# excitation acceleration
# is amplitude * frequence squared

exccitation_amplitude = 0.001*0.05

    
        
        
fig,ax=plt.subplots(figsize=single_column_size)
plt.loglog(4*np.pi**2*np.array(a_data_dict['25'] )[:,0]**2*0.001/0.5,np.array(a_data_dict['25'])[:,1],'o',label='AR=25')
plt.loglog(4*np.pi**2*np.array(a_data_dict['100'])[:,0]**2*0.001/0.5,np.array(a_data_dict['100'])[:,1],'o',label='AR=100')
plt.loglog(4*np.pi**2*np.array(a_data_dict['300'])[:,0]**2*0.001/0.5,np.array(a_data_dict['300'])[:,1],'o',label='AR=300')

plt.xlabel(r'$a/g$')
plt.ylabel(r'$F$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_a.png',dpi=300,bbox_inches='tight')


# %%
fig,ax=plt.subplots(figsize=single_column_size)
        
        
        
        
# %%
    


    
    
# %%
