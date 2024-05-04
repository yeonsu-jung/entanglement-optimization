from numpy import loadtxt, savetxt
import jax.numpy as jnp
import numpy as np
import datetime

from visualizations import set_3d_plot, plot_many_rods
from matplotlib import pyplot as plt

from utils import parse_id_string
import glob

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



if __name__ == '__main__':
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