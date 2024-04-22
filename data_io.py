from numpy import loadtxt, savetxt
import jax.numpy as jnp
import numpy as np
import datetime

from visualizations import set_3d_plot, plot_many_rods
from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    pth = '/Users/yeonsu/Data/entangled_rods_N100_relaxed_21-04-2024_01-10-46.txt'
    q = read_data(pth)
    export_start_last_edges(q)
    
    set_3d_plot()
    plot_many_rods(jnp.reshape(q,(-1,5)))
    
    plt.show()