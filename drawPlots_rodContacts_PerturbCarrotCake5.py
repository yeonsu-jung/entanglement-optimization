# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from data_io import import_all_log, parse_path_string    
from analysis import get_curr_force_essentials
import re
    
# %%
output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# Entangle
protocol_id = 'PerturbCarrotCake5'

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N125_AR25_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N250_AR50_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N375_AR75_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N500_AR100_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N625_AR125_g0.5')
    
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
        
        last_frame = len(time_line)-1
        print(f'Last frame: {last_frame}')
        for frame in range(0,len(time_line),1):
            curr_nodes = node_list[frame].reshape((-1,10,3))
            curr_force_all_info = contact_list[frame].reshape(-1,18)
            curr_force_essentials = get_curr_force_essentials(curr_force_all_info,curr_nodes)
            total_number_of_contacts[frame] = len(curr_force_essentials)
            total_force_sum[frame] = np.sum(np.linalg.norm(curr_force_essentials[:,3:6],axis=1))
            
        ax1.plot(time_line,total_number_of_contacts)
        ax2.plot(time_line,total_force_sum)
        
        local_data_dict = {}
        local_data_dict['file_id'] = file_id
        local_data_dict['time_line'] = time_line
        local_data_dict['total_number_of_contacts'] = total_number_of_contacts
        local_data_dict['total_force_sum'] = total_force_sum
        
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


