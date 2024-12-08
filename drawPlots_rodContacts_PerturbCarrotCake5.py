# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from data_io import import_all_log, parse_path_string    
from analysis import get_curr_force_essentials
import re
import networkx as nx
    
# %%
output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# %%
log_string = ''
data_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0600_AR120/NonIntersectingBox-N0600-AR120-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240609-105239.csv'
file_id,surfix,num_rods,AR,datetime_string = parse_path_string(data_path)
time_line, node_list, contact_list = import_all_log(data_path,max_rows=100000)

time_line = np.array(time_line)
time_line = time_line[time_line <= 10]
node_list = node_list[:len(time_line)]
contact_list = contact_list[:len(time_line)]

time_line = time_line[1:]
node_list = node_list[1:]
contact_list = contact_list[1:]

print(f'Size of time_line: {len(time_line)}')
print(f'Number of rods: {num_rods}')

log_string = log_string + f'Number of rods: {num_rods}\n'
log_string = log_string + f'Number of time points: {len(time_line)}\n'

total_number_of_contacts = np.zeros(len(time_line))
total_force_sum = np.zeros(len(time_line))
size_of_largest_cluster = np.zeros(len(time_line))
avg_num_contacts_in_the_largest_cluster = np.zeros(len(time_line))

last_frame = len(time_line)-1
print(f'Last frame: {last_frame}')
for frame in range(0,len(time_line),1):
    curr_nodes = node_list[frame].reshape((-1,10,3))
    curr_force_all_info = contact_list[frame].reshape(-1,18)
    curr_force_essentials = get_curr_force_essentials(curr_force_all_info,curr_nodes)
    
    contact_ij = curr_force_all_info[:,4:6].astype(int)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(curr_nodes)))
    graph.add_edges_from(contact_ij)
    
    clusters = list(nx.connected_components(graph))
    largest_cluster = max(clusters,key=len)
    size_of_largest_cluster[frame] = len(largest_cluster)
    # number of edges in the largest cluster
    avg_num_contacts_in_the_largest_cluster[frame] = len(graph.subgraph(largest_cluster).edges)/len(largest_cluster)
    
    total_number_of_contacts[frame] = len(curr_force_essentials)
    total_force_sum[frame] = np.sum(np.linalg.norm(curr_force_essentials[:,3:6],axis=1))


# %%

# Entangle
protocol_id = 'Modelo1_fine'

pathlist = []

pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1051_RUN_PerturbEEModelo1_N0125_AR025')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0250_AR050')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0375_AR075')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0500_AR100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0525_AR105')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0550_AR110')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0575_AR115')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0600_AR120')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0625_AR125')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0750_AR150')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0875_AR175')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1000_AR200')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1250_AR250')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300')

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1808_RUN_PerturbEECarrotCake5_N125_AR25_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1809_RUN_PerturbEECarrotCake5_N250_AR50_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1809_RUN_PerturbEECarrotCake5_N375_AR75_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1809_RUN_PerturbEECarrotCake5_N500_AR100_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1809_RUN_PerturbEECarrotCake5_N625_AR125_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1823_RUN_PerturbEECarrotCake5_N1000_AR300_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1823_RUN_PerturbEECarrotCake5_N1500_AR300_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-0229_RUN_PerturbEECarrotCake5_N600_AR120_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-0245_RUN_PerturbEECarrotCake5_N525_AR105_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-0245_RUN_PerturbEECarrotCake5_N550_AR110_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-0245_RUN_PerturbEECarrotCake5_N575_AR115_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-0245_RUN_PerturbEECarrotCake5_N600_AR120_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-1858_RUN_PerturbEECarrotCake5_N620_AR124_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-1901_RUN_PerturbEECarrotCake5_N620_AR124_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-1903_RUN_PerturbEECarrotCake5_N605_AR121_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-1903_RUN_PerturbEECarrotCake5_N610_AR122_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-1903_RUN_PerturbEECarrotCake5_N615_AR123_g0.5')

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N125_AR25_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N250_AR50_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N375_AR75_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N500_AR100_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N625_AR125_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240603-2227_RUN_PerturbEECarrotCake5_N1000_AR2000_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240603-1639_RUN_PerturbEECarrotCake5_N1500_AR300_g0.5')
    
output_path = f'{output_root}/{protocol_id}'
if not os.path.exists(output_path):
    os.makedirs(output_path)
# %%
if os.path.exists(f'{output_path}/total_contact_over_time_data.pkl'):
    exit()

if not os.path.exists(f'{output_path}/total_contact_over_time_data.pkl'):
    
    fig1,ax1=plt.subplots(1,1,figsize=(10,5))
    fig2,ax2=plt.subplots(1,1,figsize=(10,5))

    total_contact_over_time_data = []
    for pth in pathlist:
        
        # find csv file
        data_path = None
        for file in Path(pth).rglob('*.csv'):
            if str(file.stem).endswith('lastFrame'):
                continue
            
            data_path = file
            break
        
        log_string = ''
        
        file_id,surfix,num_rods,AR,datetime_string = parse_path_string(data_path)
        time_line, node_list, contact_list = import_all_log(data_path,max_rows=100000)
        
        time_line = np.array(time_line)
        time_line = time_line[time_line <= 10]
        node_list = node_list[:len(time_line)]
        contact_list = contact_list[:len(time_line)]
        
        time_line = time_line[1:]
        node_list = node_list[1:]
        contact_list = contact_list[1:]
        
        print(f'Size of time_line: {len(time_line)}')
        print(f'Number of rods: {num_rods}')
        
        log_string = log_string + f'Number of rods: {num_rods}\n'
        log_string = log_string + f'Number of time points: {len(time_line)}\n'
        
        total_number_of_contacts = np.zeros(len(time_line))
        total_force_sum = np.zeros(len(time_line))
        size_of_largest_cluster = np.zeros(len(time_line))
        avg_num_contacts_in_the_largest_cluster = np.zeros(len(time_line))
        
        last_frame = len(time_line)-1
        print(f'Last frame: {last_frame}')
        for frame in range(0,len(time_line),1):
            curr_nodes = node_list[frame].reshape((-1,10,3))
            curr_force_all_info = contact_list[frame].reshape(-1,18)
            curr_force_essentials = get_curr_force_essentials(curr_force_all_info,curr_nodes)
            
            contact_ij = curr_force_all_info[:,4:6].astype(int)
            graph = nx.Graph()
            graph.add_nodes_from(range(len(curr_nodes)))
            graph.add_edges_from(contact_ij)
            
            clusters = list(nx.connected_components(graph))
            largest_cluster = max(clusters,key=len)
            size_of_largest_cluster[frame] = len(largest_cluster)
            # number of edges in the largest cluster
            avg_num_contacts_in_the_largest_cluster[frame] = len(graph.subgraph(largest_cluster).edges)/len(largest_cluster)
            
            total_number_of_contacts[frame] = len(curr_force_essentials)
            total_force_sum[frame] = np.sum(np.linalg.norm(curr_force_essentials[:,3:6],axis=1))
            
        ax1.plot(time_line,total_number_of_contacts)
        ax2.plot(time_line,total_force_sum)
        
        local_data_dict = {}
        local_data_dict['file_id'] = file_id
        local_data_dict['time_line'] = time_line
        local_data_dict['total_number_of_contacts'] = total_number_of_contacts
        local_data_dict['total_force_sum'] = total_force_sum
        local_data_dict['size_of_largest_cluster'] = size_of_largest_cluster
        local_data_dict['avg_num_contacts_in_the_largest_cluster'] = avg_num_contacts_in_the_largest_cluster
        
        total_contact_over_time_data.append(local_data_dict)
            
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Number of contacts')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total force sum (N)')
# %%        
# fig1.savefig(f'{output_path}/number_of_contacts_total.png',dpi=300)
# fig2.savefig(f'{output_path}/total_force_sum_total.png',dpi=300)

for dta in total_contact_over_time_data:
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(dta['time_line'],dta['total_number_of_contacts'])
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Number of contacts')
    fig.savefig(f'{output_path}/{dta["file_id"]}_number_of_contacts.png',dpi=300)
    
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(dta['time_line'],dta['total_force_sum'])
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Total force sum (N)')
    fig.savefig(f'{output_path}/{dta["file_id"]}_total_force_sum.png',dpi=300)
    
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(dta['time_line'],dta['size_of_largest_cluster'])
    ax.set_xlabel('Time (sec)')

# %%
import pickle

if not os.path.exists(f'{output_path}/total_contact_over_time_data.pkl'):
    with open(f'{output_path}/total_contact_over_time_data.pkl','wb') as f:
        pickle.dump(total_contact_over_time_data,f)
    
# %%
with open(f'{output_path}/total_contact_over_time_data.pkl','rb') as f:
    total_contact_over_time_data = pickle.load(f)

# %%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for dta in total_contact_over_time_data:
    file_id = dta['file_id']
    match = re.search(r'N(\d+)-AR(\d+)', file_id)
    if match:
        N = match.group(1)
        AR = match.group(2)
        N = int(N)
        AR = int(AR)
        print(f'N: {N}, AR: {AR}')
    else:
        print('No match found')
    ax.plot(dta['time_line'],dta['total_number_of_contacts']/N*2,label=f'N={N}, AR={AR}')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Avg. no. of contacts per rod')
fig.legend(loc='lower right')
fig.savefig(f'{output_path}/avg_number_of_contacts_per_rod.png',dpi=300)

# %%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for dta in total_contact_over_time_data:
    file_id = dta['file_id']
    match = re.search(r'N(\d+)-AR(\d+)', file_id)
    if match:
        N = match.group(1)
        AR = match.group(2)
        N = int(N)
        AR = int(AR)
        print(f'N: {N}, AR: {AR}')
    else:
        print('No match found')
    ax.plot(dta['time_line'],dta['total_number_of_contacts'],label=f'N={N}, AR={AR}')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Total no. of contacts')
fig.legend(loc='lower right')
fig.savefig(f'{output_path}/total_number_of_contacts.png',dpi=300)


# %%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for dta in total_contact_over_time_data:
    file_id = dta['file_id']
    match = re.search(r'N(\d+)-AR(\d+)', file_id)
    if match:
        N = match.group(1)
        AR = match.group(2)
        N = int(N)
        AR = int(AR)
        print(f'N: {N}, AR: {AR}')
    else:
        print('No match found')
    ax.plot(dta['time_line'],dta['size_of_largest_cluster'],label=f'N={N}, AR={AR}')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Size of largest cluster')
fig.legend(loc='lower right')
fig.savefig(f'{output_path}/size_of_largest_cluster.png',dpi=300)

# %%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for dta in total_contact_over_time_data:
    file_id = dta['file_id']
    match = re.search(r'N(\d+)-AR(\d+)', file_id)
    if match:
        N = match.group(1)
        AR = match.group(2)
        N = int(N)
        AR = int(AR)
        print(f'N: {N}, AR: {AR}')
    else:
        print('No match found')

    ax.plot(dta['time_line'],dta['avg_num_contacts_in_the_largest_cluster'],label=f'N={N}, AR={AR}')
    
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Avg. no. of contacts in the largest cluster')
fig.legend(loc='lower right')
fig.savefig(f'{output_path}/avg_num_contacts_in_the_largest_cluster.png',dpi=300)

