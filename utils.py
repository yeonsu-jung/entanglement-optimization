from numpy import loadtxt, savetxt
import numpy as np

def reshape_txt(pth):    
    data = loadtxt(pth)
    data = data.reshape((-1,5))
    
    # filepart
    filepart = pth.split('/')[-1].split('.')[0]
    # new file name
    newfile = f'/Users/yeonsu/Data/{filepart}_reshaped.txt'
    
    savetxt(newfile, data)
    

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
    filepart = pth.split('/')[-1].split('.')[0]
    # new file name
    newfile = f'/Users/yeonsu/Data/{filepart}_xyz.txt'
    
    savetxt(newfile, new_data)
    
def start_last_edges(filename):
    data = loadtxt(filename)
    data = data.reshape((-1,5))
    new_data = np.zeros((data.shape[0],6))
    new_data[:,:3] = data[:,:3]
    
    N = data.shape[0]
    for i in range(N):
        new_data[i,3:6] = sph2cart(data[i,3],data[i,4])
        
    start_edges = new_data[:,0:3]
    last_edges = new_data[:,-3:]
    
    filepart = pth.split('/')[-1].split('.')[0]
    newfile = f'/Users/yeonsu/Data/{filepart}_start_edges.txt'    
    savetxt(newfile, start_edges)
    
    newfile = f'/Users/yeonsu/Data/{filepart}_last_edges.txt'    
    savetxt(newfile, last_edges)
    
    
if __name__ == '__main__':
    
    tmp = np.array([1,2,3,4,5,6])
    print(tmp[:3])
    
    print(tmp[-3:])
    
    
    pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_21-04-2024_15-35-59.txt'
    # xyzform(pth)
    
    start_last_edges(pth)
    
    