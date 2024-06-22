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
# %%
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


a_data_full = {25: [], 100: [], 300: []}

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
# %%
denom = rho*rod_diameter**2*rod_length

fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(1/0.5*np.array(g_data_dict['25'])[:,0],np.array(g_data_dict['25'])[:,1],'o',label=r'$\alpha=25$')
plt.plot(1/0.5*np.array(g_data_dict['100'])[:,0],np.array(g_data_dict['100'])[:,1],'o',label=r'$\alpha=100$')
plt.plot(1/0.5*np.array(g_data_dict['300'])[:,0],np.array(g_data_dict['300'])[:,1],'o',label=r'$\alpha=300$')
plt.xlabel(r'$F/(\rho g d^2 l)$')
plt.ylabel(r'$r$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_g.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(np.array(a_data_dict['25'])[:,0],np.array(a_data_dict['25'])[:,1],'o',label=r'$\alpha=25$')
plt.plot(np.array(a_data_dict['100'])[:,0],np.array(a_data_dict['100'])[:,1],'o',label=r'$\alpha=100$')
plt.plot(np.array(a_data_dict['300'])[:,0],np.array(a_data_dict['300'])[:,1],'o',label=r'$\alpha=300$')
plt.xlabel(r'$a/g$')
plt.ylabel(r'$r$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_a.png',dpi=300,bbox_inches='tight')


# %%
# excitation acceleration
# is amplitude * frequence squared

exccitation_amplitude = 0.001*0.05

fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(4*np.pi**2*np.array(a_data_dict['25'] )[:,0]**2*0.001/0.5,np.array(a_data_dict['25'])[:,1],'o',label=r'$\alpha=25$')
plt.plot(4*np.pi**2*np.array(a_data_dict['100'])[:,0]**2*0.001/0.5,np.array(a_data_dict['100'])[:,1],'o',label=r'$\alpha=100$')
plt.plot(4*np.pi**2*np.array(a_data_dict['300'])[:,0]**2*0.001/0.5,np.array(a_data_dict['300'])[:,1],'o',label=r'$\alpha=300$')

plt.xlabel(r'$a/g$')
plt.ylabel(r'$F$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_a.png',dpi=300,bbox_inches='tight')


# %%
fig,ax=plt.subplots(figsize=single_column_size)
        
        
        
        
# %%
    
a_data_dict

    
    
# %%
ARs = np.array([25,50,75,100,125,200,300])
cve = np.array([0.9483138 , 1.15187444, 1.21259957, 1.24966556, 1.25542284,1.34128365, 1.41815353])

plt.plot(ARs,cve,'o')
# %%


fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(1/0.5*np.array(g_data_dict['25'])[:,0],np.array(g_data_dict['25'])[:,1],'o',label=r'$\sigma/\mu=0.95$')
plt.plot(1/0.5*np.array(g_data_dict['100'])[:,0],np.array(g_data_dict['100'])[:,1],'o',label=r'$\sigma/\mu=1.25$')
plt.plot(1/0.5*np.array(g_data_dict['300'])[:,0],np.array(g_data_dict['300'])[:,1],'o',label=r'$\sigma/\mu=1.42$')
plt.xlabel(r'$F/(\rho g d^2 l)$')
plt.ylabel(r'$r$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_g-cv.png',dpi=300,bbox_inches='tight')
# %%


fig,ax=plt.subplots(figsize=single_column_size)
plt.plot(np.array(a_data_dict['25'])[:,0],np.array(a_data_dict['25'])[:,1],'o',label=r'$\sigma/\mu=0.95$')
plt.plot(np.array(a_data_dict['100'])[:,0],np.array(a_data_dict['100'])[:,1],'o',label=r'$\sigma/\mu=1.25$')
plt.plot(np.array(a_data_dict['300'])[:,0],np.array(a_data_dict['300'])[:,1],'o',label=r'$\sigma/\mu=1.42$')
plt.xlabel(r'$a/g$')
plt.ylabel(r'$r$')
plt.legend(loc='lower right')
plt.savefig(f'{output_dir}/entanglement_vs_a-cv.png',dpi=300,bbox_inches='tight')

# %%
fig = plt.figure(figsize=np.array(single_column_size)*1.2)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(1/0.5*np.array(g_data_dict['25'])[:,0], 1/0.95, np.array(g_data_dict['25'])[:,1], label=r'$\sigma/\mu=0.95$')
ax.scatter(1/0.5*np.array(g_data_dict['100'])[:,0], 1/1.25, np.array(g_data_dict['100'])[:,1], label=r'$\sigma/\mu=1.25$')
ax.scatter(1/0.5*np.array(g_data_dict['300'])[:,0], 1/1.42, np.array(g_data_dict['300'])[:,1], label=r'$\sigma/\mu=1.42$')
# show foots
# each data points is connected with its projection on xy plane
# for i in range(len(g_data_dict['25'])):
#     ax.plot([1/0.5*np.array(g_data_dict['25'])[i,0],1/0.5*np.array(g_data_dict['25'])[i,0]],[0.95,0.95],[0,np.array(g_data_dict['25'])[i,1]],'k--',linewidth=0.5)
    
# for i in range(len(g_data_dict['100'])):
#     ax.plot([1/0.5*np.array(g_data_dict['100'])[i,0],1/0.5*np.array(g_data_dict['100'])[i,0]],[1.25,1.25],[0,np.array(g_data_dict['100'])[i,1]],'k--',linewidth=0.5)
    
# for i in range(len(g_data_dict['300'])):
#     ax.plot([1/0.5*np.array(g_data_dict['300'])[i,0],1/0.5*np.array(g_data_dict['300'])[i,0]],[1.42,1.42],[0,np.array(g_data_dict['300'])[i,1]],'k--',linewidth=0.5)
    
# show surface at y = 1.6
for i in range(len(g_data_dict['25'])):
    ax.plot([1/0.5*np.array(g_data_dict['25'])[i,0],1/0.5*np.array(g_data_dict['25'])[i,0]],[1/0.95,1/1.6],[np.array(g_data_dict['25'])[i,1],np.array(g_data_dict['25'])[i,1]],'k--',linewidth=0.5)
    
for i in range(len(g_data_dict['100'])):
    ax.plot([1/0.5*np.array(g_data_dict['100'])[i,0],1/0.5*np.array(g_data_dict['100'])[i,0]],[1/1.25,1/1.6],[np.array(g_data_dict['100'])[i,1],np.array(g_data_dict['100'])[i,1]],'k--',linewidth=0.5)
    
for i in range(len(g_data_dict['300'])):
    ax.plot([1/0.5*np.array(g_data_dict['300'])[i,0],1/0.5*np.array(g_data_dict['300'])[i,0]],[1/1.42,1/1.6],[np.array(g_data_dict['300'])[i,1],np.array(g_data_dict['300'])[i,1]],'k--',linewidth=0.5)

# surface at z = 0.9

# ax.plot([0,10],[1.6,1.6],[0.9,0.9],'k--',linewidth=0.5)

ax.scatter(1/0.5*np.array(g_data_dict['25'])[:,0], 1/1.6, np.array(g_data_dict['25'])[:,1], label=r'$\sigma/\mu=0.95$',color='k',s=1)
ax.scatter(1/0.5*np.array(g_data_dict['100'])[:,0], 1/1.6, np.array(g_data_dict['100'])[:,1], label=r'$\sigma/\mu=1.25$',color='k',s=1)
ax.scatter(1/0.5*np.array(g_data_dict['300'])[:,0], 1/1.6, np.array(g_data_dict['300'])[:,1], label=r'$\sigma/\mu=1.42$',color='k',s=1)

# flip y axis
ax.invert_yaxis()

ax.set_xlabel(r'$F/(\rho g d^2 l)$')
ax.set_ylabel(r'$1/(\sigma/\mu$)')
ax.zaxis.labelpad=-3 # <- change the value here

ax.set_zlabel(r'$f$')
# plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{output_dir}/entanglement_vs_g-cv-3d.png',dpi=300,bbox_inches='tight')




fig = plt.figure(figsize=np.array(single_column_size)*1.2)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(np.array(a_data_dict['25'])[:,0], 1/0.95, np.array(a_data_dict['25'])[:,1], label=r'$\sigma/\mu=0.95$')
ax.scatter(np.array(a_data_dict['100'])[:,0], 1/1.25, np.array(a_data_dict['100'])[:,1], label=r'$\sigma/\mu=1.25$')
ax.scatter(np.array(a_data_dict['300'])[:,0], 1/1.42, np.array(a_data_dict['300'])[:,1], label=r'$\sigma/\mu=1.42$')

# show foots
# each data points is connected with its projection on xy plane


# for i in range(len(a_data_dict['25'])):
#     ax.plot([np.array(a_data_dict['25'])[i,0],np.array(a_data_dict['25'])[i,0]],[0.95,0.95],[0,np.array(a_data_dict['25'])[i,1]],'k--',linewidth=0.5)
    
# for i in range(len(a_data_dict['100'])):
#     ax.plot([np.array(a_data_dict['100'])[i,0],np.array(a_data_dict['100'])[i,0]],[1.25,1.25],[0,np.array(a_data_dict['100'])[i,1]],'k--',linewidth=0.5)
    
# for i in range(len(a_data_dict['300'])):
#     ax.plot([np.array(a_data_dict['300'])[i,0],np.array(a_data_dict['300'])[i,0]],[1.42,1.42],[0,np.array(a_data_dict['300'])[i,1]],'k--',linewidth=0.5)
    
# show surface at y = 1.6
for i in range(len(a_data_dict['25'])):
    ax.plot([np.array(a_data_dict['25'])[i,0],np.array(a_data_dict['25'])[i,0]],[1/0.95,1/1.6],[np.array(a_data_dict['25'])[i,1],np.array(a_data_dict['25'])[i,1]],'k--',linewidth=0.5)
    
for i in range(len(a_data_dict['100'])):
    ax.plot([np.array(a_data_dict['100'])[i,0],np.array(a_data_dict['100'])[i,0]],[1/1.25,1/1.6],[np.array(a_data_dict['100'])[i,1],np.array(a_data_dict['100'])[i,1]],'k--',linewidth=0.5)
    
for i in range(len(a_data_dict['300'])):
    ax.plot([np.array(a_data_dict['300'])[i,0],np.array(a_data_dict['300'])[i,0]],[1/1.42,1/1.6],[np.array(a_data_dict['300'])[i,1],np.array(a_data_dict['300'])[i,1]],'k--',linewidth=0.5)
    
    
ax.scatter(np.array(a_data_dict['25'])[:,0], 1/1.6, np.array(a_data_dict['25'])[:,1], label=r'$\sigma/\mu=0.95$',color='k',s=1)
ax.scatter(np.array(a_data_dict['100'])[:,0], 1/1.6, np.array(a_data_dict['100'])[:,1], label=r'$\sigma/\mu=1.25$',color='k',s=1)
ax.scatter(np.array(a_data_dict['300'])[:,0], 1/1.6, np.array(a_data_dict['300'])[:,1], label=r'$\sigma/\mu=1.42$',color='k',s=1)

ax.invert_yaxis()

ax.set_xlabel(r'$a/g$')
ax.set_ylabel(r'$1/(\sigma/\mu$)')
ax.zaxis.labelpad=-3 # <- change the value here

ax.set_zlabel(r'$f$')
# plt.legend(loc='upper left')
plt.tight_layout()


plt.savefig(f'{output_dir}/entanglement_vs_a-cv-3d.png',dpi=300,bbox_inches='tight')
# %%
fig = plt.figure(figsize=np.array(single_column_size)*1.2)
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(1/0.5*np.array(g_data_dict['25'])[:,0], 1/0.95, 0, label=r'$\sigma/\mu=0.95$')
# ax.scatter(1/0.5*np.array(g_data_dict['100'])[:,0], 1/1.25, 0, label=r'$\sigma/\mu=1.25$')
# ax.scatter(1/0.5*np.array(g_data_dict['300'])[:,0], 1/1.42, 0, label=r'$\sigma/\mu=1.42$')
# ax.set_zlim([0,0.01])

# ax.scatter(1, 1/0.95, np.array(a_data_dict['25'])[:,0] , label=r'$\sigma/\mu=0.95$')


# loop over keys and values
mu_over_sigma_dict = {'25': 1/0.95, '100': 1/1.25, '300': 1/1.42}

for k,_list in a_data_dict.items():
    
    for i_ in range(len(_list)):
        dta = _list[i_][0]
        f = _list[i_][1]
        
        if f > 0.9:
            ax.scatter(1, mu_over_sigma_dict[k], dta, color='b', s=20)
        else:
            ax.scatter(1, mu_over_sigma_dict[k], dta, color='k', alpha = 0.1)

ax.invert_yaxis()
ax.set_xlabel(r'$F/(\rho_s g d^2 l)$')
ax.set_ylabel(r'$1/(\sigma/\mu)$')
ax.zaxis.labelpad=-4 # <- change the value here

ax.set_zlabel(r'$a/g$')
plt.savefig(f'{output_dir}/entanglement_vs_a-g-cv-3d.png',dpi=300,bbox_inches='tight')

# %%
class data_container:
    def __init__(self,dataphat,start_row=0,max_rows=100000,skip_rows=1):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,start_row=start_row,max_rows=max_rows,skip_rows=skip_rows)



pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1343_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq0.1')

# class data_container:
#     def __init__(self,dataphat,max_rows=100000):
#         self.path = Path(dataphat)
#         out = parse_path_string(self.path)
#         # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
#         self.file_id = out[0]
#         self.surfix = out[1]
#         self.num_rods = out[2]
#         self.AR = out[3]
#         self.datetime_string = out[4]
#         self.time_line, self.node_list, self.contact_list = import_all_log(self.path,max_rows=max_rows)
# %%        
max_rows = 1000000
data_container_list = []

# (g, freq)
data_container_dict = {}

for pth in pathlist:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        
        # data_container_list.append(data_container(file,max_rows=max_rows))
        
        exp_id = pth.split('/')[-1]
        search_result = re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)_freq(\d+(\.\d+)?)', exp_id)
        
        N = int(search_result.group(1))
        AR = int(search_result.group(2))
        g = float(search_result.group(3))
        freq = float(search_result.group(5))
        
        data_container_list.append(data_container(file,start_row=480,max_rows=10000))
        
        print(N,AR,g,freq)
        
        
# %%


# %%
i_ = 2
data_entry = data_container_list[i_]

num_rods = data_entry.num_rods
tt = data_entry.time_line
vv = data_entry.contact_list[i_].reshape(-1,18)

contact_ij = vv[:,4:6].astype(int)


# contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
curr_nodes = data_entry.node_list[i_]
graph = nx.Graph()
graph.add_nodes_from(range(len(curr_nodes)))
graph.add_edges_from(contact_ij)
clusters = list(nx.connected_components(graph))
len(clusters)

# largest clusters
largest_cluster = max(clusters,key=len)
f = len(largest_cluster)/num_rods

print(f'f = {f}')
print(f'AR = {data_entry.AR}')
print(f'{len(contact_ij)}')
# %%
nodes_in_shape = curr_nodes.reshape(-1,10,3)

fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
for rr in nodes_in_shape:
    ax.plot(rr[:,0],rr[:,1],rr[:,2],alpha=0.2)
# # %%
# fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
for i_ in largest_cluster:
    ax.plot(nodes_in_shape[i_][:,0],nodes_in_shape[i_][:,1],nodes_in_shape[i_][:,2],'k')
# %%


# %%
import pandas as pd
# initialize df
df = pd.DataFrame(columns=['static','dynamic','f','AR','sigma_over_mu'])

ARs = np.array([25,50,75,100,125,200,300])
cve = np.array([0.9483138 , 1.15187444, 1.21259957, 1.24966556, 1.25542284,1.34128365, 1.41815353])
cve_dict = {25: 0.9483138 , 50: 1.15187444, 75: 1.21259957, 100: 1.24966556, 125: 1.25542284, 200: 1.34128365, 300: 1.41815353}
mu_over_sigma_dict = {'25': 1/0.95, '100': 1/1.25, '300': 1/1.42}

fig,ax=plt.subplots(subplot_kw={'projection': '3d'},figsize=np.array(single_column_size)*1.5)
data_table = []
for dc in data_container_list:
    num_rods = dc.num_rods
    contact_ij = dc.contact_list[-1].reshape(-1,18)[:,4:6].astype(int)
    curr_nodes = dc.node_list[-1]
    
    contact_graph = nx.Graph()
    contact_graph.add_nodes_from(range(len(curr_nodes)))
    contact_graph.add_edges_from(contact_ij)
    clusters = list(nx.connected_components(contact_graph))
    # largest clusters
    largest_cluster = max(clusters,key=len)
    f = len(largest_cluster)/num_rods
    
    exp_id = str(dc.path).split('/')[-2]
    search_result = re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)_freq(\d+(\.\d+)?)', exp_id)        
    N = int(search_result.group(1))
    AR = int(search_result.group(2))
    g = float(search_result.group(3))
    freq = float(search_result.group(5))
    
    # if freq == 100:
    #     continue
    
    if AR == 150:
        AR = 125
    
    sigma_over_mu = cve_dict[AR]
    mu_over_sigma = 1/sigma_over_mu
    
    static = g/0.5
    dynamic = 4*np.pi*freq**2*0.001/0.5
    # data_table.append(np.array([static,dynamic,f,AR,sigma_over_mu]))
    df.loc[len(df)] = [static,dynamic,f,AR,sigma_over_mu]
    
    # ax.scatter(static,dynamic,f)
    # foots
    # ax.plot([static,static],[dynamic,dynamic],[0,f],'k--',linewidth=0.5)
    
    if f > 0.9:
        ax.scatter(static,mu_over_sigma,dynamic,color='g',edgecolors='gray')
    else:
        ax.scatter(static,mu_over_sigma,dynamic,color='k',alpha=0.5,marker='x')


ax.set_xlabel(r'$F/(\rho_s g d^2 l)$',labelpad=-3)
ax.set_ylabel(r'$\mu/\sigma$',labelpad=-3)
ax.set_zlabel(r'$a/g$',labelpad=-13)
# ax.set_xlim([0,8])
# ax.set_ylim([0,1.6])
ax.set_zlim([0,3])
ax.invert_yaxis()
ax.zaxis.loc='top'

# ax.zaxis.labelpad=-6 # <- change the value here

ax.tick_params(axis='x', pad=-3)
ax.tick_params(axis='y', pad=-3)
ax.tick_params(axis='z', pad=-3)
# plt.zticks(rotation=90)

plt.tight_layout()
plt.savefig(f'{output_dir}/entanglement_vs_a-g-cv-3d-with-data.png',dpi=300,bbox_inches='tight')

# %%


# %%    
# ax.set_xlabel(r'$F/(\rho_s g d^2 l)$')
# ax.set_ylabel(r'$a/g$')
# ax.set_zlabel(r'$f$')
    
    
# %%
data_container_list[2].AR
data_container_list[2].path.parent.stem
    
    
    # for i_ in range(len(_list)):
    #     dta = _list[i_][0]
    #     f = _list[i_][1]
        
    #     if f > 0.9:
    #         ax.scatter(1, mu_over_sigma_dict[k], dta, color='b', s=20)
    #     else:
    #         ax.scatter(1, mu_over_sigma_dict[k], dta, color='k', alpha = 0.1)
# %%
# plot f for static = 1 and AR = 100
default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
single_column_size = (1.5,1.3)
fig,ax=plt.subplots(figsize=single_column_size)
#fontsize
plt.rcParams.update({'font.size': 8})
df_ = df[(df['static'] == 1) & (df['AR'] == 25)]
ax.plot(df_['dynamic'],df_['f'],'o-',label=r'$\alpha=25, \tilde{F} = 1$',markersize=3,color=default_color_list[0],alpha=0.5)

df_ = df[(df['static'] == 1) & (df['AR'] == 100)]
xx = df_['dynamic'].to_numpy()
i_sort = np.argsort(xx)
xx = xx[i_sort]
yy = df_['f'].to_numpy()
yy = yy[i_sort]

ax.plot(xx,yy,'o-',label=r'$\alpha=100, \tilde{F} = 1$',markersize=3,color=default_color_list[1],alpha=0.5)

df_ = df[(df['static'] == 1) & (df['AR'] == 125)]
ax.plot(df_['dynamic'],df_['f'],'o-',label=r'$\alpha=125, \tilde{F} = 1$',markersize=3,color=default_color_list[2],alpha=0.5)

# small legend
# legend = plt.legend(loc='lower right',fontsize=6)
# plt.xlabel(r'$a/g$',labelpad=-3)
# plt.ylabel(r'$f$',labelpad=2,rotation=90)
# # plt.tight_layout()
# plt.savefig(f'{output_dir}/f_vs_a-g.png',dpi=300,bbox_inches='tight')


# single_column_size = (1.5,1.3)
# fig,ax=plt.subplots(figsize=single_column_size)
#fontsize

plt.rcParams.update({'font.size': 8})
df_ = df[(df['static'] == 20) & (df['AR'] == 25)]
ax.plot(df_['dynamic'],df_['f'],'s--',label=r'$\alpha=25, \tilde{F} = 20$',markersize=3,color=default_color_list[0],alpha=0.5)

df_ = df[(df['static'] == 20) & (df['AR'] == 100)]
xx = df_['dynamic'].to_numpy()
i_sort = np.argsort(xx)
xx = xx[i_sort]
yy = df_['f'].to_numpy()
yy = yy[i_sort]

ax.plot(xx,yy,'s--',label=r'$\alpha=100, \tilde{F} = 20$',markersize=3,color=default_color_list[1],alpha=0.5)

df_ = df[(df['static'] == 20) & (df['AR'] == 125)]
ax.plot(df_['dynamic'],df_['f'],'s--',label=r'$\alpha=125, \tilde{F} = 20$',markersize=3,color=default_color_list[2],alpha=0.5)

ax.set_xscale('log')
ax.set_yscale('log')


plt.yticks([0.1,1],rotation=90)

plt.xlabel(r'$a/g$',labelpad=-3)
plt.ylabel(r'$f$',labelpad=2)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/f_vs_a-g.png',dpi=300,bbox_inches='tight')
plt.legend()
plt.savefig(f'{output_dir}/f_vs_a-g.eps',bbox_inches='tight')





