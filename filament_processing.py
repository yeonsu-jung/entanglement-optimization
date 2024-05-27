import numpy as np
import filamentprocessing
from pathlib import Path
from scipy.io import loadmat

def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))


def curvature_of_polygonal_curve(nodes):
    tan2 = nodes[2:,:] - nodes[1:-1,:]    
    tan1 = nodes[1:-1,:] - nodes[:-2,:]
    
    nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
    den = np.sum(tan1*tan2,axis=1)
    curvature = np.sum(nom/den)
    return curvature

def calculate_curvature(seg):
    # Compute first and second derivatives
    dx_dt = np.gradient(seg[:, 0])
    dy_dt = np.gradient(seg[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Compute curvature
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**(3/2)
    
    return curvature

def prune_filaments():
    return 0

if __name__ is '__main__':
    
    data_root = Path('/Users/yeonsu/Code/contact_table')
    data_obj = loadmat(data_root / 'centerlines.mat')
    
    print(data_obj.keys())
    
    

