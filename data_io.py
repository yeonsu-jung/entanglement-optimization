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
    
    spatial_data = spatial_data.reshape((-1,num_rods,num_vertices,3))
    # print(spatial_data[0,0,0,:])
    # print(spatial_data[0,0,1,:])
    
    return spatial_data, timepoints
    

if __name__ == '__main__':
    # pth = '/Users/yeonsu/Data/from-cluster/20240422-161737_node_20240424-155848.csv'
    upper_dir = '/Users/yeonsu/Data/from-cluster/'
    pth = f'{upper_dir}20240422-161737_node_20240424-160715.csv'
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
    
    from visualizations import set_3d_plot, plot_many_curves
    fig,ax=set_3d_plot()
    params = {}
    plot_many_curves(spatial_data[-1,:,:,:],params=params,ax=ax)
    plt.show()
    
    # set_3d_plot()
    # plot_many_rods(jnp.reshape(q,(-1,5)))
    
    # plt.show()