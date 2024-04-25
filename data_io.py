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

def sph2cart(theta,phi):
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.array([x,y,z]).transpose()

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
    
    

if __name__ == '__main__':
    # pth = '/Users/yeonsu/Data/from-cluster/20240422-161737_node_20240424-155848.csv'
    upper_dir = '/Users/yeonsu/Data/from-cluster/'
    pth = f'{upper_dir}20240422-161737_node_20240425-010509.csv'
    
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
    
    # plt.show()