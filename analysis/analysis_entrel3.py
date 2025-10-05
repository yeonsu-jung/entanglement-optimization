# %%
import re
from pathlib import Path
import numpy as np
from data_io import import_all_log, save_as_mat
from scipy.io import loadmat
from matplotlib import pyplot as plt

# meta_folder = Path("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick_3,1,2,720")
meta_folder = Path("")


def plot_rods(xyz,ax=None):
    if ax is None:
        _,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(0,len(xyz),1):
        r1 = xyz[i,:3]
        r2 = xyz[i,3:]
        ax.plot([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]])
    return ax

# %%
# get directory list
pathlist = list(meta_folder.glob("*RUN*"))
# pathlist = [str(p) for p in pathlist if p.is_dir()]
def find_csv_file(folder_path):
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            continue
        else:
            possible_paths.append(pth)    
    if len(possible_paths) == 0:
        print('No csv files found in the folder')
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        # find heaviest file
        max_size = 0
        for pth in possible_paths:
            size = os.path.getsize(pth)
            if size > max_size:
                max_size = size
                heaviest_file = pth
        possible_paths = [heaviest_file]
        
    pth = str(possible_paths[0])
    return pth

# %%
# data class
class SingleKickData:
    def __init__(self, dt_string, AR, num_rods, kick_amplitude, friction_coefficient, data_path):
        self.dt_string = dt_string
        self.AR = AR
        self.num_rods = num_rods
        self.kick_amplitude = kick_amplitude
        self.friction_coefficient = friction_coefficient
        self.data_path = data_path

    # print function
    def __repr__(self):
        return f"dt_string: {self.dt_string}, AR: {self.AR}, num_rods: {self.num_rods}, kick_amplitude: {self.kick_amplitude}, friction_coefficient: {self.friction_coefficient}\n"

def parse_pathname(pathname):
    dt_string = re.search(r'(\d{8}-\d{4})',pathname).group(1)
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))    
    kick_amplitude = float(re.search('Kick(\d+.\d+)',pathname).group(1))
    friction_info = re.search('Friction(\d+.\d+)',pathname)

    key_info = re.search('RandomKeys_(\d+),(\d+),(\d+),(\d+)',pathname)
    if key_info:
        random_keys = key_info.group(1)
    else:
        random_keys = '3,1,2,N/A'

    if friction_info:
        friction_coefficient = float(friction_info.group(1))
    else:
        friction_coefficient = 0.4

    # if a string contains a substring
    if "NoFriction" in str(pathname):
        friction_coefficient = 0

    return dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient

data_list = []
for pth in pathlist:
    dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient = parse_pathname(str(pth))
    csv_file_path = find_csv_file(pth)
    mat_file_path = csv_file_path.replace('.csv','.mat')
    if not Path(mat_file_path).exists():
        save_as_mat(csv_file_path,max_rows=100000)
    else:
        print(f'{mat_file_path} already exists')

    data_list.append(SingleKickData(dt_string, AR, num_rods, kick_amplitude, friction_coefficient, mat_file_path))
print(data_list)

# %%
# for dta in data_list:
#     data = loadmat(dta.data_path)
#     print(data.keys())
# %%
# pick up AR = 300, friction_coefficient = 0.4, kick_amplitude = 0.1
# find this

AR = 500
chosen_data = []
for dta in data_list:
    if dta.AR == AR and (0.1 <= dta.friction_coefficient < 0.2) and dta.kick_amplitude == 1.0:
        # data = loadmat(dta.data_path,simplify_cells=True)
        chosen_data.append(dta)

# %%
global_data_list = []
for dta in chosen_data:
    data = loadmat(dta.data_path,simplify_cells=True)
    
    time_line = data['time_line']
    node_list = data['node_list']
    velocity_list = data['velocity_list']
    contacts_list = data['contact_list']

    nodes_at_last_frame = node_list[-1]
    xyz = nodes_at_last_frame.reshape(-1,6)    

    rad_gyr_list = []
    num_contacts_list = []
    for i in range(len(node_list)):
        nodes_at_frame = node_list[i]
        contacts_at_frame = contacts_list[i]
        xyz = nodes_at_frame.reshape(-1,6)
        centroids = (xyz[:,:3]+xyz[:,3:])/2

        global_centroid = np.mean(centroids,axis=0)
        moment_arm = centroids - global_centroid
        radius_of_gyration = np.mean(np.linalg.norm(moment_arm,axis=1))
        rad_gyr_list.append(radius_of_gyration)
        num_contacts_list.append(len(contacts_at_frame)//8/500)

    local_dataset = {
        'time_line': time_line,
        'data_obj': dta,
        'node_list': node_list,
        'velocity_list': velocity_list,
        'rad_gyr_list': rad_gyr_list,
        'num_contacts_list': num_contacts_list,
    }
    global_data_list.append(local_dataset)

# sort global_data_list by friction_coefficient
global_data_list = sorted(global_data_list,key=lambda x: x['data_obj'].friction_coefficient)
friction_coefficient_list = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
# %%
_trimmed = []
_trimmed = global_data_list
# to_remove = [11,12,13]
# for i in range(len(global_data_list)):
#     if i in to_remove:
#         continue
#     _trimmed.append(global_data_list[i])
    
    
# %%
from transforms import x_to_q
from potentials import total_effective_potential
import time

global_entanglement_over_time = {}

for local_dataset in _trimmed:
    time_line = local_dataset['time_line']
    nodes_list = local_dataset['node_list']
    velocity_list = local_dataset['velocity_list']
    num_contacts_list = local_dataset['num_contacts_list']

    kinetic_energy_over_time = []
    mean_velocity_over_time = []
    entanglement_over_time = []
    start = time.time()
    for i in range(0,len(nodes_list),30):
        x = nodes_list[i].reshape(-1,6)
        q = x_to_q(x)
        e = -total_effective_potential(q)
        entanglement_over_time.append(e)

        v = velocity_list[i]
        v = v.reshape(-1,6)

        centroid_velocity = np.mean(v,axis=0)
        non_rigid_body_velocity = v - centroid_velocity
        mean_velocity = np.mean(np.linalg.norm(non_rigid_body_velocity,axis=1))
        mean_velocity_over_time.append(mean_velocity)

        kinetic_energy = 0.5*np.sum(np.linalg.norm(v,axis=1)**2)
        kinetic_energy_over_time.append(kinetic_energy)

    
    print(f'{local_dataset["data_obj"].friction_coefficient} took {time.time()-start:.2f} seconds')
    skipped_time_line = time_line[::30]
    
    local_data = {
        'time_line': skipped_time_line,
        'entanglement': entanglement_over_time,
        'mean_velocity': mean_velocity_over_time,
        'kinetic_energy': kinetic_energy_over_time,
    }

    global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient] = local_data
# %%
fig,ax=plt.subplots(figsize=(2.5,2))
for local_dataset in _trimmed:
    time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
    kinetic_energy = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['kinetic_energy']
    ax.plot(time_line,kinetic_energy,'o-',label=f'{local_dataset["data_obj"].friction_coefficient}')
    
# %%




# %%
fig,ax=plt.subplots(figsize=(2.5,2))
for local_dataset in _trimmed:
    time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
    entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
    entanglement = np.array(entanglement)/(500*499/2)
    ax.loglog(time_line,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')

ax.set_xlabel('Time, $t$ (sec)')
ax.set_ylabel('Normalized entanglement, $E$')
# ax.legend(title='$\\mu$',loc='upper right')
plt.savefig(f'/Users/yeonsu/Figures/Random3,1,2/EntanglementOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')


# %%

    
# %%

    



# %%
subfolder_name = 'Random3,1,2'
figure_output_folder = f"/Users/yeonsu/Figures/{subfolder_name}"

fig,axs=plt.subplots(2,1,figsize=(4,5))
final_radius_of_gyration = []
for local_dataset in _trimmed:
    time_line = local_dataset['time_line']
    rad_gyr_list = local_dataset['rad_gyr_list']
    num_contacts_list = local_dataset['num_contacts_list']
    
    axs[0].semilogy(time_line,rad_gyr_list,'-',label=f'{local_dataset["data_obj"].friction_coefficient}')
    axs[1].plot(time_line,num_contacts_list)

    final_radius_of_gyration.append(rad_gyr_list[-1])
axs[1].set_xlabel('Time, $t$ (sec)')
axs[1].set_ylabel('Number of contacts')
axs[0].set_ylabel('Radius of gyration, $R_g$')
# small legend
axs[0].legend(title='$\\mu$',loc='upper right',fontsize=7)
plt.savefig(f'{figure_output_folder}/RadiusOfGyrationOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
# %%


fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
fig,ax=plt.subplots(figsize=(2.5,2))
plt.plot(fric_coeff,final_radius_of_gyration,'o-')
plt.xlabel('Friction coefficient, $\\mu$')
plt.ylabel('Radius of gyration, $R_g$')


if not os.path.exists(figure_output_folder):
    os.makedirs(figure_output_folder)

plt.savefig(f'{figure_output_folder}/RadiusOfGyration_AR{AR}.png',dpi=300,bbox_inches='tight')
# %%
fig,ax=plt.subplots(figsize=(2.5,2))
for local_dataset in _trimmed:
    time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
    mean_velocity = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['mean_velocity']
    ax.plot(time_line,mean_velocity,'o-',label=f'{local_dataset["data_obj"].friction_coefficient}')
plt.xlabel('Time, $t$ (sec)')
plt.ylabel('Mean relative velocity, $v$ (L/sec)')
plt.savefig(f'{figure_output_folder}/MeanVelocityOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
# %%
# %%
file_id = f'N{num_rods}-AR{int(AR):04d}-Kick{1.0}'
output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}'

import os
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %%
import polyscope as ps
from visualizations import prep_for_polyscope

rod_diameter = 1/AR
ps.init()



_t = 0
a_list_of_curves = node_list[_t].reshape(num_rods,-1,3)
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)

min_z = np.min(nodes[:,2])
# ps.set_ground_plane_height_factor(-min_z)
               
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps_curves.set_radius(rod_diameter/2,relative=False)

ps.set_up_dir("z_up")
ps.screenshot('temp.png',transparent_bg=False)
# ps.show()
# exit()
# %%
num_files_already = len(list(Path(output_path).glob('frame_*')))
print(f'Number of frames: {num_files_already}')

# %%
skip_factor = 500
for i,a_list_of_curves in enumerate(node_list[num_files_already::skip_factor]):
    a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
    # num_rods = len(a_list_of_curves)
    nodes = np.vstack(a_list_of_curves)
    ps_curves.update_node_positions(nodes)
    file_path = f'{output_path}/frame_{i+num_files_already:04d}.png'
    ps.screenshot(file_path)

import subprocess
subprocess.run(['ffmpeg', '-framerate', '10', '-i', f'{output_path}/frame_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p', f'{output_path}/output.mp4'])
