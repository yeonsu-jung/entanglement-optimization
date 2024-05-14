# %%
import numpy as np
from utils import parse_filename
import glob
from pathlib import Path
from scipy.io import loadmat
import os
from matplotlib import pyplot as plt
from visualizations import set_3d_plot
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time
import networkx as nx

class logger:
    def __init__(self):
        self.log = []
    def add(self,txt):
        self.log.append(txt)
    def print(self):
        for txt in self.log:
            print(txt)
            
def get_centerlines(pth,logger):
    
    dta = loadmat(pth)
    cl = dta["centerlines"]        
    N = cl.shape[0]
    centerlines = []
    for i in range(N):
        centerlines.append(cl[i][0])
        
    return centerlines

def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(-height_z, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid

def set_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotate_grid(X, Y, Z, rotation_matrix):
    shape = X.shape
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    rotated_grid = rotation_matrix @ grid
    X_rot, Y_rot, Z_rot = rotated_grid.reshape(3, *shape)
    return X_rot, Y_rot, Z_rot

def main():
    root_pth = Path('./xray_raw_data')
    for pth in (Path.glob(root_pth, '**/centerlines.mat')):
        dta = loadmat(pth)
        # get folder from pth
        prnt_fldr = str(pth.parent)
        exp_id = prnt_fldr.split('/')[-1]
        prnt_fldr = Path(prnt_fldr)
        
        cl = dta["centerlines"]
        N = cl.shape[0]
        centerlines = []
        for i in range(N):
            centerlines.append(cl[i][0])

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
        
        visualize = False
        if visualize:
            from visualizations import set_3d_plot
            fig,ax = set_3d_plot()
            for d in data_rearranged:
                ax.plot(d[:,0]/650,d[:,1]/650,d[:,2]/650)
            plt.savefig(prnt_fldr / 'rendering.png', dpi=300)
            
        # pixel_size_in_um = 78.22*1e-6;
        pixel_size_in_um = 1
        num_rods = len(data_rearranged)
        data_rearranged = np.array(data_rearranged)
        data_rearranged = np.array(data_rearranged)*pixel_size_in_um
        data_rearranged = data_rearranged.reshape(num_rods,-1)
        
        N = data_rearranged.shape[0]
        tokens = exp_id.split('_')
        for token in tokens:
            if token.startswith('alpha'):
                alpha = int(token[5:])
            if token.startswith('epsilon'):
                epsilon = int(token[7:])
        
        radius = 1/alpha/2
        print(f'alpha: {alpha}, epsilon: {epsilon}, radius: {radius}')
        
        for txt_file in Path.glob(prnt_fldr, '*.txt'):
            os.remove(txt_file)
            
        data_rearranged = data_rearranged[-1000:,:]
        np.savetxt(prnt_fldr / f'centerlines-N{N}-AR{alpha}-Scale100.txt',data_rearranged)
    
    return 0

def prep_svd_cylinder(cl):
    N = len(cl)
    svd_cylinders = np.zeros((N,7))
    centroids = np.zeros((N,3))
    orientations = np.zeros((N,3))
    for i,c in enumerate(cl):
        center = c.mean(axis=0)
        u,s,v = np.linalg.svd(c-center)
        orientation = v[0,:]
        
        e1 = center - s[0]*(v[0,:])/np.sqrt(2)/2
        e2 = center + s[0]*(v[0,:])/np.sqrt(2)/2
        r1 = s[1]/np.sqrt(2)
        
        svd_cylinders[i,:] = np.hstack((e1,e2,r1))
        centroids[i,:] = center
        orientations[i,:] = orientation
        
    return svd_cylinders,centroids,orientations

@jit(nopython=True)
def fixbound(x):
    if x < 0.:
        return 0.
    elif x > 1:
        return 1.
    else:
        return x
    
    

@jit(nopython=True)
def lumelsky_dist(point1s, point1e, point2s, point2e):
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)
    R = np.dot(d1, d2)

    den = D1 * D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:  # line1 is a segment and line2 is a point
            u = 0
            t = fixbound(S1 / D1)
        elif D2 != 0:  # line2 is a segment and line1 is a point
            t = 0
            u = fixbound(-S2 / D2)
        else:  # both segments are points
            t = u = 0
    elif den == 0:  # lines are parallel
        t = 0
        u = fixbound(-S2 / D2)
        uf = fixbound(u)
        if uf != u:
            t = fixbound((uf * R + S1) / D1)
            u = uf
    else:  # general case
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = fixbound((t * R - S2) / D2)
        uf = fixbound(u)
        if uf != u:
            t = fixbound((uf * R + S1) / D1)
            u = uf

    # Compute distance
    dist = np.linalg.norm(d1 * t - d2 * u - d12)
    return dist


@jit(nopython=True)
def compute_cylinder_distance_matrix(svd_cylinders):
    N = svd_cylinders.shape[0]
    distances = np.zeros((N, N))
    for i in range(N):
        p1 = svd_cylinders[i, 0:3]
        q1 = svd_cylinders[i, 3:6]
        for j in range(i+1, N):
            p2 = svd_cylinders[j, 0:3]
            q2 = svd_cylinders[j, 3:6]
            distances[i, j] = lumelsky_dist(p1, q1, p2, q2)
    
    return distances

def plot_centerline_with_container(centerlines,svd_cylinders,i,ax):
    cl = centerlines[i]
    cyl = svd_cylinders[i,:]

    cyl_diam = cyl[6]
    cyl_e1 = cyl[0:3]
    cyl_e2 = cyl[3:6]
    cyl_cen = (cyl_e1+cyl_e2)/2
    cyl_len = np.linalg.norm(cyl_e1-cyl_e2)
    cyl_axis = (cyl_e2-cyl_e1)/cyl_len

    Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, cyl_diam, cyl_len/2)
    # Compute the rotation matrix
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), cyl_axis) 
    # Rotate the cylinder
    Xc_rot, Yc_rot, Zc_rot = rotate_grid(Xc, Yc, Zc, rotation_matrix)
    Xc_rot = Xc_rot + cyl_cen[0]
    Yc_rot = Yc_rot + cyl_cen[1]
    Zc_rot = Zc_rot + cyl_cen[2]
    
    bounding_box = np.array([np.min(cl, axis=0), np.max(cl, axis=0)])
    ax.plot_surface(Xc_rot, Yc_rot, Zc_rot, alpha=0.5)
    ax.plot(cl[:,0], cl[:,1], cl[:,2], color='r')
    # ax.scatter(cyl_e1[0], cyl_e1[1], cyl_e1[2], color='g')
    # zoom in
    # ax.set_xlim(bounding_box[:,0])
    # ax.set_ylim(bounding_box[:,1])
    # ax.set_zlim(bounding_box[:,2])

# %%
def foo():
    pth = '/Users/yeonsu/Documents/GitHub/entanglement-optimization/xray_raw_data/alpha38_epsilon00/centerlines.mat'
    centerlines = get_centerlines(pth,logger)
    # svd_cylinders,_,_ = prep_svd_cylinder(centerlines)
    
def load_xray_data(pth):
    pth = Path(pth)
    dta = loadmat(pth)
    cl = dta["centerlines"]
    
    N = cl.shape[0]
    centerlines = []
    for i in range(N):
        centerlines.append(cl[i][0])

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

import numpy as np

# Function to check if a point is inside a cylinder
def is_inside_cylinder(point, cylinder_start, cylinder_end, radius):
    v = cylinder_end - cylinder_start
    w = point - cylinder_start
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    b = c1 / c2
    pb = cylinder_start + b * v
    distance = np.linalg.norm(pb - point)
    return distance <= radius and 0 <= b <= 1

# Function to estimate the volume overlap
def estimate_overlap(cylinder1_start, cylinder1_end, radius1, cylinder2_start, cylinder2_end, radius2, num_samples=100000):
    min_x = min(cylinder1_start[0], cylinder1_end[0], cylinder2_start[0], cylinder2_end[0]) - max(radius1, radius2)
    max_x = max(cylinder1_start[0], cylinder1_end[0], cylinder2_start[0], cylinder2_end[0]) + max(radius1, radius2)
    min_y = min(cylinder1_start[1], cylinder1_end[1], cylinder2_start[1], cylinder2_end[1]) - max(radius1, radius2)
    max_y = max(cylinder1_start[1], cylinder1_end[1], cylinder2_start[1], cylinder2_end[1]) + max(radius1, radius2)
    min_z = min(cylinder1_start[2], cylinder1_end[2], cylinder2_start[2], cylinder2_end[2]) - max(radius1, radius2)
    max_z = max(cylinder1_start[2], cylinder1_end[2], cylinder2_start[2], cylinder2_end[2]) + max(radius1, radius2)
    
    count_inside_both = 0
    
    for _ in range(num_samples):
        point = np.array([
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
            np.random.uniform(min_z, max_z)
        ])
        
        if is_inside_cylinder(point, cylinder1_start, cylinder1_end, radius1) and is_inside_cylinder(point, cylinder2_start, cylinder2_end, radius2):
            count_inside_both += 1
    
    volume_box = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    volume_overlap = (count_inside_both / num_samples) * volume_box
    
    return volume_overlap
# %%    
if __name__ == '__main__':
    # main()
    pth = Path('/Users/yeonsu/Documents/GitHub/entanglement-optimization/xray_raw_data/alpha38_epsilon00/centerlines.mat')
    centerlines,_ = load_xray_data(pth)
    centerlines = centerlines[-1000:]
    # data_rearranged = data_rearranged[-1000:,:]
    svd_cylinders,_,_ = prep_svd_cylinder(centerlines)
    
    i_rod = 69
    j_rod = 1 # 69,597
    # Define the cylinders
    cylinder1_start = svd_cylinders[i_rod,0:3]
    cylinder1_end = svd_cylinders[j_rod,3:6]
    radius1 = svd_cylinders[i_rod,6]

    cylinder2_start = svd_cylinders[j_rod,0:3]
    cylinder2_end = svd_cylinders[j_rod,3:6]
    radius2 = svd_cylinders[j_rod,6]
    
    dist_matrix = compute_cylinder_distance_matrix(svd_cylinders)
    dist_matrix[np.diag_indices_from(dist_matrix)] = np.inf
    dist_matrix = dist_matrix + dist_matrix.T
    
    collision_limit_matrix = svd_cylinders[:,6,None] + svd_cylinders[:,6]
    collision_limit_matrix *= 2
    
    # find pairs of cylinders that are close    
    pairs = np.where(dist_matrix < collision_limit_matrix)
    
    colliding_cylinder_indices = []
    
    for i in range(len(pairs[0])):
        i_rod = pairs[0][i]
        j_rod = pairs[1][i]
        
        p1 = svd_cylinders[i_rod, 0:3]
        q1 = svd_cylinders[i_rod, 3:6]
        p2 = svd_cylinders[j_rod, 0:3]
        q2 = svd_cylinders[j_rod, 3:6]
        
        dist = lumelsky_dist(p1, q1, p2, q2)
        if dist < 1e-6:
            # get rid of j_rod
            colliding_cylinder_indices.append((i_rod,j_rod))
            
        visualize = False
        if visualize:
            print(f'Collision between rods {i_rod} and {j_rod}')
            
            fig,ax = set_3d_plot()            
            plot_centerline_with_container(centerlines,svd_cylinders,i_rod,ax)
            plot_centerline_with_container(centerlines,svd_cylinders,j_rod,ax)
            plt.savefig(f'colliding_cylinders/collision_{i_rod}_{j_rod}.png', dpi=300)
            plt.close()
    
    visualize = False
    if visualize:
        fig,ax = set_3d_plot()
        i = 1000
        i_rod = pairs[0][i]
        j_rod = pairs[1][i]
        plot_centerline_with_container(centerlines,svd_cylinders,i_rod,ax)
        plot_centerline_with_container(centerlines,svd_cylinders,j_rod,ax)
        plt.show()
    
    if visualize:
        fig,ax = set_3d_plot()
        for i_rod in colliding_cylinder_indices:
            plot_centerline_with_container(centerlines,svd_cylinders,i_rod,ax)
        plt.show()
    
    G = nx.Graph()
    G.add_edges_from(colliding_cylinder_indices)
    connected_components = list(nx.connected_components(G))
    # Print the connected components
    for i, component in enumerate(connected_components):
        print(f"Component {i+1}: {component}")

    items = connected_components[1]
    fig,ax = set_3d_plot()
    
    if visualize:
        for i in items:
            print(i)
            plot_centerline_with_container(centerlines,svd_cylinders,i,ax)
        plt.show()
    
    
    
    # Estimate the overlap
    # overlap_volume = estimate_overlap(cylinder1_start, cylinder1_end, radius1, cylinder2_start, cylinder2_end, radius2)
    # print(f"Estimated Overlap Volume: {overlap_volume}")
            
    print()
    # main()  
