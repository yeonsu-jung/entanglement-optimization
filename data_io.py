# %%
from numpy import loadtxt, savetxt
import jax.numpy as jnp
import numpy as np
import datetime
import re

from visualizations import set_3d_plot, plot_many_rods
from matplotlib import pyplot as plt

from utils import parse_id_string
import glob
import scipy.io
import os
from pathlib import Path
from scipy.io import loadmat


def parse_path_string(pth):    
    pth = str(pth)
    filename = pth.split('/')[-1]        
    file_id = filename.split('-mu')[0]
    
    surfix_match = re.search(r'\d{8}-\d{6}', filename)
    surfix = surfix_match.group(0) if surfix_match else None
    
    num_rods_match = re.search(r'-N(\d+)-', filename)
    num_rods = int(num_rods_match.group(1)) if num_rods_match else None
    
    AR_match = re.search(r'-AR(\d+)-', filename)
    AR = int(AR_match.group(1)) if AR_match else None
    
    datetime_match = re.search(r'\d{8}-\d{6}', filename)
    datetime_str = datetime_match.group(0) if datetime_match else None
    
    return file_id, surfix, num_rods, AR, datetime_str


def load_xray_data(pth):
    pth = Path(pth)
    dta = loadmat(pth)
    cl = dta["centerlines"]
    
    N = cl.shape[0]
    centerlines = []
    for i in range(N):
        centerlines.append(np.array(cl[i][0],dtype=np.float64))

    data_rearranged = []
    for rr in centerlines:
        # rr = centerlines[i]
        # interpolate to have 10 points.
        rr = np.array(rr)
        N = rr.shape[0]
        t = np.linspace(0,1,N)
        t_new = np.linspace(0,1,10)
        rr_new = np.zeros((10,3))
        rr_new = np.array([np.interp(t_new,t,rr[:,0]),
                        np.interp(t_new,t,rr[:,1]),
                        np.interp(t_new,t,rr[:,2])]).T

        data_rearranged.append(rr_new)
    
    pixel_size_in_um = 1
    num_rods = len(data_rearranged)
    data_rearranged = np.array(data_rearranged)
    data_rearranged = np.array(data_rearranged)*pixel_size_in_um
    data_rearranged = data_rearranged.reshape(num_rods,-1)
    return centerlines,data_rearranged


def read_data(pth):
    q = loadtxt(pth)
    # q = jnp.array(q,dtype=jnp.float64)
    return q

def reshape_txt(pth):    
    data = loadtxt(pth)
    data = data.reshape((-1,5))
    
    # filepart
    filepart = pth.split('/')[-1].split('.')[0]
    # new file name
    newfile = f'/Users/yeonsu/Data/{filepart}_reshaped.txt'
    
    savetxt(newfile, data)
    return 1    

def xyzform(filename):
    data = loadtxt(filename)
    data = data.reshape((-1,5))
    new_data = np.zeros((data.shape[0],6))
    new_data[:,:3] = data[:,:3]
    
    N = data.shape[0]
    for i in range(N):
        new_data[i,3:6] = sph2cart(data[i,3],data[i,4])
        
    # filepart
    filepart = filename.split('/')[-1].split('.')[0]
    # new file name
    newfile = f'/Users/yeonsu/Data/{filepart}_xyz.txt'
    
    savetxt(newfile, new_data)
    return 1
    
def export_start_last_edges(data):
    data = data.reshape((-1,5))
    new_data = np.zeros((data.shape[0],6))
    new_data[:,:3] = data[:,:3]
    
    N = data.shape[0]
    for i in range(N):
        new_data[i,3:6] = sph2cart(data[i,3],data[i,4])
        
    start_edges = new_data[:,0:3]
    last_edges = new_data[:,-3:]    
    
    num_rods = data.shape[0]
    # YYYY-MM-DD_HH-MM-SS
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    newfile = f'/Users/yeonsu/Data/{dt_string}_start_edges_N{num_rods}.txt'
    savetxt(newfile, start_edges)
    
    newfile = f'/Users/yeonsu/Data/{dt_string}_last_edges_N{num_rods}.txt'
    savetxt(newfile, last_edges)
    
def import_from_dismech(pth,num_rods):    
    dta = loadtxt(pth,delimiter=',',dtype=np.float64)
    timepoints = dta[:,0]
    spatial_data = dta[:,1:]
    num_vertices = spatial_data.shape[1]//(3*num_rods)
    # print(spatial_data.shape[1])
    # print(num_vertices)    
    # print(spatial_data[0,0:10])    
    
    # spatial_data = spatial_data.reshape((-1,num_rods,num_vertices,3))
    # print(spatial_data[0,0,0,:])
    # print(spatial_data[0,0,1,:])
    
    return spatial_data, timepoints

def import_from_dismech_hook(pth,num_rods,start_col = 1,max_rows = 1000, row_skip = 100):
    dta = loadtxt(pth,delimiter=',',dtype=np.float64, max_rows=max_rows)
    timepoints = dta[::row_skip,0]
    spatial_data = dta[::row_skip,start_col:]
    num_vertices = spatial_data.shape[1]//(3*num_rods)
    
    return spatial_data, timepoints, num_vertices
    
    
def example_import_and_plot():
    upper_dir = '/Users/yeonsu/Data/from-cluster/'
    pth = f'{upper_dir}20240422-161737_node_20240425-181017.csv'
    
    id_string = (pth.split('/')[-1].split('.')[0]).split('_')[0]
    
    cache_dir = f'/Users/yeonsu/Data/cache/{id_string}'    
    filename = glob.glob(f'{cache_dir}/*.txt')[0]    
    print(filename)    
    # check if the filename contains N and AR
    if 'N' in filename:
        splitted = parse_id_string(filename)
        for s in splitted:
            if 'N' in s:
                num_rods = int(float(s[1:]))
            if 'AR' in s:
                AR = int(float(s[2:]))
        
        print(f"num_rods: {num_rods}")
        print(f"AR: {AR}")
    
        
    spatial_data,timepoints = import_from_dismech(pth,num_rods)
    # spatial_data = spatial_data.reshape((-1,num_rods,num_vertices,3))
    
    # from visualizations import set_3d_plot, plot_many_curves
    # fig,ax=set_3d_plot()
    # params = {}
    # plot_many_curves(spatial_data[-1,:,:,:],params=params,ax=ax)
    # plt.show()
    
    spatial_data = jnp.array(spatial_data)
    from potentials import distance_between_two_curves, all_distnaces_between_curves
    
    num_vertices = spatial_data.shape[1]//(3*num_rods)
    
    import time
    
    start = time.time()
    d = all_distnaces_between_curves(spatial_data[1,:])
    now = time.time()
    
    print(d)
    print(jnp.min(d))
    print(f"Elapsed time: {now-start} seconds")
    
    nnz = jnp.count_nonzero(d<0.5)
    print(nnz)
    
    fig = plt.figure()
    plt.hist(d, bins=100)    
    filename = f"/Users/yeonsu/Figures/{id_string}_histogram.png"
    plt.savefig(filename)
    # plt.show()
    
    from visualizations import set_3d_plot, plot_many_curves
    import matplotlib.animation as animation
    
    print(f"num timepoints: {spatial_data.shape[0]}")
    fig,ax=set_3d_plot()
    params = {}
    spatial_data = spatial_data.reshape((-1,num_rods,num_vertices,3))
    plot_many_curves(spatial_data[0,:,:,:],params=params,ax=ax)
    
    def update(frame):
        ax.clear()
        print(f"frame: {frame}")        
        plot_many_curves(spatial_data[frame,:,:,:],params=params,ax=ax)
        return ax
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=spatial_data.shape[0], interval=30)
    
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(f'/Users/yeonsu/Videos/{id_string}.mp4', writer = FFwriter)
  
# moved to 
# def q_to_x(q):
#     # q = jnp.array(q)
#     q = q.reshape((-1,5))
#     x = jnp.zeros((q.shape[0],6))
#     x = x.at[:,:3].set(q[:,:3])
#     x = x.at[:,3:6].set(sph2cart(q[:,3],q[:,4]) + x[:,0:3])
#     return x

def export_from_q_to_x(pth):
    q = loadtxt(pth)
    q = jnp.array(q,dtype=jnp.float64)
    print(q.shape)
    
    id_string = pth.split('/')[-2]
    filepart = pth.split('/')[-1].split('.')[0]
    num_rods = filepart.split('N')[1]
    q_in_matrix = jnp.reshape(q,(-1,5))
    
    from transforms import q_to_x
    x = q_to_x(q)
    savetxt(f'/Users/yeonsu/Data/{id_string}_entangled_edges_N{num_rods}.txt',x)
       
    # sanity check
    # d = jnp.linalg.norm(x[:,:3] - x[:,3:6],axis=1)
    # print(d)
    x = np.array(x)    
    from visualizations import plot_edges, set_3d_plot
    fig,ax=set_3d_plot()
    plot_edges(x,ax)
    plt.show()
    return 1

def scale_and_export(pth):
    from data_io import import_from_dismech    
    from transforms import sph2cart
    data,num_rods,AR = import_from_dismech(pth)    
    data = data.reshape((-1,5))
    new_data = np.zeros((data.shape[0],6))
    new_data[:,:3] = data[:,:3]    
    N = data.shape[0]
    for i in range(N):
        new_data[i,3:6] = data[i,:3] + sph2cart(data[i,3],data[i,4])
        
    # export path
    export_dir = f'/Users/yeonsu/Data/export/'
    # os.makedirs(export_dir,exist_ok=False)
    
    scale_factor = 100
    length = 1*scale_factor
    center = np.concatenate([np.mean(new_data[:,:3],axis=0),np.mean(new_data[:,:3],axis=0)])
    new_data = (new_data-center)*scale_factor
    newfile = f'{export_dir}/{dt_string}_edges_N{num_rods}_AR{AR}_length{length}.txt'
    np.savetxt(newfile, new_data)
    
def export_nodes_at_final_time(pth):    
    filepart = pth.split('/')[-1].split('.')[0]    
    num_rods = 100
    curves, timepoints = import_from_dismech(pth,num_rods)    
    last_curve = curves[-1,:]
    last_curve = last_curve.reshape((-1,30))
    export_dir = '/Users/yeonsu/Data/export'
    np.savetxt(f'{export_dir}/{filepart}_last_nodes.txt',last_curve)
    print(f"Exported {filepart}_last_nodes.txt")
    return 1


def foo():
    # sim_id = '20240426-215217_node_20240427-014524'
    sim_id = '20240426-215217_node_20240427-160317'    
    root_dir = '/Users/yeonsu/Data/from-cluster'
    pth = f'{root_dir}/{sim_id}.csv'
    num_rods = 100
    
    export_nodes_at_final_time(pth)
    
    from visualizations import plot_many_curves,set_3d_plot    
    from data_io import import_from_dismech
    curves,timepoints = import_from_dismech(pth,num_rods)
    nodes_at_a_time = curves[-1,:]
    
    fig,ax = set_3d_plot()
    plot_many_curves(nodes_at_a_time,num_rods,ax)
    plt.show()
    
def import_all_log(alllog_pth, max_rows = 10):
    with open(alllog_pth) as f:
        lines = f.readlines()
    lines = lines[:max_rows]
        
    time_line = []
    node_list = []
    contact_list = []
    for i,line in enumerate(lines):
        if 'Time' in line:
            time_line.append(float(line.split('Time: ')[-1].rstrip('\n')))
            
        if 'Node' in line:
            next_line = lines[i+1]                       
            node_list.append(np.array([float(x) for x in next_line.split(',')]))
            
        if 'Force' in line:
            next_line = lines[i+1]
            if next_line == "\n":
                contact_list.append(np.array([]))
            else:
                contact_list.append(np.array([float(x) for x in next_line.split(',')]))
                
    return time_line, node_list, contact_list
# %%

def pullout_video_frames_single_file(alllog_pth):
    log_string = ''
    
    file_id,surfix,num_rods,AR,datetime_string = parse_path_string(alllog_pth)
    time_line, node_list, contact_list = import_all_log(alllog_pth,max_rows=100000)
    output_path = f'/Users/yeonsu/Videos/{protocol_id}/{file_id}_{surfix}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        start_point = 0
    
    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')
    
    log_string = log_string + f'Number of rods: {num_rods}\n'
    log_string = log_string + f'Number of time points: {len(time_line)}\n'
    
    start_point = 0
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(start_point,len(node_list),1):
        nodes_in_matrix = node_list[i].reshape((-1,30))
        for node in nodes_in_matrix:
            rr = node.reshape((-1,3))
            ax.plot(rr[:,0],rr[:,1],rr[:,2])
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.view_init(elev=0,azim=0)
        ax.text(1,1,1,f'time: {time_line[i]}')
        plt.tight_layout(pad=0)
        
        plt.savefig(f'{output_path}/frames_{i:04d}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        ax.clear()
        
    with open(f'{output_path}/log.txt','w') as f:
        f.write(log_string)
        
def batch_pullout_video(pathlist):
    for folder_path in pathlist:
        folder_path = Path(folder_path)

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
        data_path = possible_paths[0]
        protocol_id = 'EatEntangledCarrotCake5'
        
        print(f'Processing {folder_path}')
        print(f'Protocol ID: {protocol_id}')
        print(f'Data paths: {str(data_path)}')
        
        pullout_video_frames_single_file(data_path)
    
# %%
if __name__ == '__main__':
    pathlist = []
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0125-AR025')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0250-AR050')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0375-AR075')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0500-AR100')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0625-AR125')
    
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240602-0259_RUN_PerturbEECarrotCake5_N125_AR25_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240602-0259_RUN_PerturbEECarrotCake5_N250_AR50_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240602-0259_RUN_PerturbEECarrotCake5_N375_AR75_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240602-0259_RUN_PerturbEECarrotCake5_N500_AR100_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240602-0259_RUN_PerturbEECarrotCake5_N625_AR125_g0.5')
    
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0125_AR025_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0250_AR050_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0375_AR075_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0500_AR100_g0.5')
    # pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0625_AR125_g0.5')
    # protocol_id = 'EatJostledCarrotCake5'
    
    
    
    # folder_path = pathlist[4]
    # folder_path = Path(folder_path)
    # # python data_io.py
    # possible_paths = []
    # for pth in folder_path.glob('**/*.csv'):
    #     if 'lastFrame' in str(pth):
    #         continue
    #     else:
    #         possible_paths.append(pth)    
    # if len(possible_paths) == 0:
    #     print('No csv files found in the folder')
    #     exit()
    # elif len(possible_paths) > 1:
    #     print('Multiple csv files found in the folder')
    #     # find heaviest file
    #     max_size = 0
    #     for pth in possible_paths:
    #         size = os.path.getsize(pth)
    #         if size > max_size:
    #             max_size = size
    #             heaviest_file = pth
    #     possible_paths = [heaviest_file]
    # data_path = possible_paths[0]
        
    data_path = '/Users/yeonsu/Data/from_cluster/NonIntersectingBox-N1000-AR200-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240602-211541.csv'
    folder_path = Path(data_path).parent
    protocol_id = 'EatEntangledCarrotCake5'
    
    
    print(f'Processing {folder_path}')
    print(f'Protocol ID: {protocol_id}')
    print(f'Data paths: {str(data_path)}')
    
    pullout_video_frames_single_file(data_path)
    
