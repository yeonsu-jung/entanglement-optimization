# %%
import numpy as np
from matplotlib import pyplot as plt
from numba import jit as njit
import time

def get_local_fields_at_a_point(centerlines, point, R, rod_diameter, visualize=False):
    _,labels,edges_all_in_one = get_edges_labels_from_centerlines(centerlines)
    
    I_local = sample_edges_locally_and_return_indices(edges_all_in_one,point,R)
    unique_labels_in_sphere = np.unique(labels[I_local])    
    ee_list = collect_local_edges(edges_all_in_one,labels,unique_labels_in_sphere)        
    
    if len(ee_list) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    total_edges = np.vstack(ee_list)
    edge_length = np.linalg.norm(total_edges[:,3:6] - total_edges[:,:3],axis=1)                
    
    number_of_local_curves = len(ee_list)
    local_volume_fraction = np.sum(edge_length)*(np.pi*rod_diameter**2/4)/(4/3*np.pi*R**3)                    
    
    local_orientational_order = compute_local_orientational_order(ee_list)
    
    lk_mat = compute_local_lk(ee_list)    
    local_average_crossing_number = np.sum(np.abs(lk_mat[np.triu_indices(lk_mat.shape[0],k=1)]))
    
    if visualize:
        fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
        for ee in ee_list:
            ax.plot(ee[:,0],ee[:,1],ee[:,2])
    
    # print(f'Number of local curves: {number_of_local_curves}')
    # print(f'Local volume fraction: {local_volume_fraction}')
    # print(f'Local orientational order: {local_orientational_order}')
    # print(f'Local average crossing number: {local_average_crossing_number}')
    
    return number_of_local_curves, local_volume_fraction, local_orientational_order, local_average_crossing_number
    
def get_edges_labels_from_centerlines(centerlines):
    edges = []
    for i in range(len(centerlines)):
        rr = centerlines[i]
        edges.append(np.hstack([rr[:-1],rr[1:]]))
    labels = label_edges(edges)
    edges_vcat = np.vstack(edges)
    return edges,labels,edges_vcat
    
def sample_edges_locally_and_return_indices(edges,center,R):
    I1 = np.linalg.norm((edges[:,:3] - center), axis=1) < R
    I2 = np.linalg.norm((edges[:,3:6] - center), axis=1) < R
    I_sphere_segments = I1 & I2
    return I_sphere_segments    

def collect_local_edges(edges,labels,unique_labels_in_sphere):
    ee_list = []
    for lb in unique_labels_in_sphere:                    
        I = labels == lb
        ee_list.append(edges[I,:])
    return ee_list
    
def compute_local_orientational_order(ee_list):                    
    S = np.zeros((3,3))                    
    total_edges = np.vstack(ee_list)
    for i in range(len(total_edges)):
        orientation_i = total_edges[i,3:6] - total_edges[i,:3]
        orientation_i /= np.linalg.norm(orientation_i)
        S += np.outer(orientation_i,orientation_i)

    eigenvalues,eigenvectors = np.linalg.eig( 3/2*(S/len(total_edges) - 1/3*np.eye(3)) )
    I = np.argmax(np.abs(eigenvalues))
    
    return eigenvalues[I]

def label_edges(edges):
    labels = []
    for i,ee in enumerate(edges):
        n = ee.shape[0]
        lb = np.ones(n,dtype=np.int64)*i        
        labels.append(lb)
        
    return np.hstack(np.array(labels))

def get_local_fields_over_domain(centerlines, R, h, rod_diameter):
    # R: radius of bounding spheres
    # h: grid spacing
    edges = []
    for i in range(len(centerlines)):
        rr = centerlines[i]
        edges.append(np.hstack([rr[:-1],rr[1:]]))
    labels = label_edges(edges)
    edges_vcat = np.vstack(edges)
    
    centerlines_vcat = np.vstack(centerlines)

    a = np.max(centerlines_vcat[:,0]) - np.min(centerlines_vcat[:, 0])
    b = np.max(centerlines_vcat[:,1]) - np.min(centerlines_vcat[:, 1])
    c = np.max(centerlines_vcat[:,2]) - np.min(centerlines_vcat[:, 2])
    x_min = np.min(centerlines_vcat[:, 0])
    y_min = np.min(centerlines_vcat[:, 1])
    z_min = np.min(centerlines_vcat[:, 2])    
    
    center_x = np.arange(R, a-R, h) + x_min
    center_y = np.arange(R, b-R, h) + y_min
    center_z = np.arange(R, c-R, h) + z_min

    num_x = center_x.size
    num_y = center_y.size
    num_z = center_z.size
    num_rods = len(centerlines)

    print(f'Number of rods: {num_rods}')
    print(f'Size of the map: {num_x}, {num_y}, {num_z}')
    t_start = time.time()
    
    n_field = np.full((num_x,num_y,num_z),np.nan)
    phi_field = np.full((num_x,num_y,num_z),np.nan)
    e_field = np.full((num_x,num_y,num_z),np.nan)
    S_field = np.full((num_x,num_y,num_z),np.nan)
    
    # local_fields = []
    for k in range(num_z):
        I_slab_segments = (np.abs(edges_vcat[:, 2] - center_z[k]) < 1.1 * R) & (np.abs(edges_vcat[:, 5] - center_z[k]) < 1.1 * R)
        # unique_labels = np.unique(labels[I_slab_segments])
        labeled_edges_in_slab = edges_vcat[I_slab_segments,:]
        labels_in_slab = labels[I_slab_segments]
        
        if labeled_edges_in_slab.shape[0] == 0:
            continue
        
        for i in range(num_x):
            for j in range(num_y):
                
                center = np.array([center_x[i], center_y[j], center_z[k]])
                
                I1 = np.linalg.norm((labeled_edges_in_slab[:,:3] - center), axis=1) < R
                I2 = np.linalg.norm((labeled_edges_in_slab[:,3:6] - center), axis=1) < R
                I_sphere_segments = I1 & I2
                if np.count_nonzero(I_sphere_segments) == 0:
                    continue
                
                # cf. for a box
                # I1 = np.abs(labeled_edges_in_slab[:,0] - center[0]) < R
                # I2 = np.abs(labeled_edges_in_slab[:,1] - center[1]) < R
                # I3 = np.abs(labeled_edges_in_slab[:,2] - center[2]) < R
                # I4 = np.abs(labeled_edges_in_slab[:,3] - center[0]) < R
                # I5 = np.abs(labeled_edges_in_slab[:,4] - center[1]) < R
                # I6 = np.abs(labeled_edges_in_slab[:,5] - center[2]) < R
                # I_box_segments = I1 & I2 & I3 & I4 & I5 & I6
                
                labeled_segments_in_sphere = labeled_edges_in_slab[I_sphere_segments]
                labels_in_sphere = labels_in_slab[I_sphere_segments]
                unique_labels_in_sphere = np.unique(labels_in_slab[I_sphere_segments])
                
                ee_list = []
                for lb in unique_labels_in_sphere:                    
                    I = labels_in_sphere == lb
                    ee_list.append(labeled_segments_in_sphere[I,:])
                    
                # we got edges in the sphere now
                
                
                total_edges = np.vstack(ee_list)
                edge_length = np.linalg.norm(total_edges[:,3:6] - total_edges[:,:3],axis=1)                
                    
                lk_mat = compute_local_lk(ee_list)
                number_of_local_curves = len(ee_list)
                local_volume_fraction = np.sum(edge_length)*(np.pi*rod_diameter**2/4)/(4/3*np.pi*R**3)                
                local_average_crossing_number = np.sum(np.abs(lk_mat[np.triu_indices(lk_mat.shape[0],k=1)]))
                local_orientational_order = compute_local_orientational_order(ee_list)
                
                n_field[i,j,k] = number_of_local_curves
                phi_field[i,j,k] = local_volume_fraction
                e_field[i,j,k] = local_average_crossing_number
                S_field[i,j,k] = local_orientational_order                

        print(f'Z-Layer: {k+1}/{num_z} \t Loop time: {time.time() - t_start:.2f} \t Elapsed time: {time.time() - t_start:.2f}')

    return n_field, phi_field, e_field, S_field, center_x, center_y, center_z

@njit(nopython=True)
def compute_linking_number_for_edges(e_i,e_j):
    r_ij = e_i[0:3] - e_j[0:3]
    r_ijj = e_i[0:3] - e_j[3:6]
    r_iij = e_i[3:6] - e_j[0:3]
    r_iijj = e_i[3:6] - e_j[3:6]    

    tol = 1e-6
    n1 = np.cross(r_ij, r_ijj)
    n1 = n1/(np.linalg.norm(n1)+tol)
    n2 = np.cross(r_ijj, r_iijj)
    n2 = n2/(np.linalg.norm(n2)+tol)
    n3 = np.cross(r_iijj, r_iij)
    n3 = n3/(np.linalg.norm(n3)+tol)
    n4 = np.cross(r_iij, r_ij)
    n4 = n4/(np.linalg.norm(n4)+tol)
    
    tol = 1e-6
    return -1/4/np.pi*np.abs(np.arcsin(  my_clip(my_dot(n1,n2),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n2,n3),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n3,n4),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n4,n1),-1.+tol,1.-tol)))    
    
@njit(nopython=True)
def compute_curves_lk(ee_i,ee_j):
    lk = 0
    num_edges_i = ee_i.shape[0]
    num_edges_j = ee_j.shape[0]
    for i in range(num_edges_i):
        e_i = ee_i[i]                        
        for j in range(num_edges_j):                            
            e_j = ee_j[j]
            lk += compute_linking_number_for_edges(e_i, e_j)
    return lk
    
# @njit(nopython=True)
def compute_local_lk(ee_list):
    # u and k are arrays of shape (N,3)
    num_distinct_labels = len(ee_list)
    lk_mat = np.full((num_distinct_labels,num_distinct_labels),np.nan)
    for i in range(num_distinct_labels):
        e_i = ee_list[i]
        for j in range(i+1,num_distinct_labels):
            e_j = ee_list[j]
            lk_mat[i,j] = compute_curves_lk(e_i, e_j)
            
    return lk_mat

@njit(nopython=True)
def compute_linking_number(p_i,p_ii,p_j,p_jj):
    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = np.cross(r_ij, r_ijj)
    n1 = n1/(np.linalg.norm(n1)+tol)
    n2 = np.cross(r_ijj, r_iijj)
    n2 = n2/(np.linalg.norm(n2)+tol)
    n3 = np.cross(r_iijj, r_iij)
    n3 = n3/(np.linalg.norm(n3)+tol)
    n4 = np.cross(r_iij, r_ij)
    n4 = n4/(np.linalg.norm(n4)+tol)
    
    tol = 1e-6
    return -1/4/np.pi*np.abs(np.arcsin(  my_clip(my_dot(n1,n2),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n2,n3),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n3,n4),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n4,n1),-1.+tol,1.-tol)))
    

@njit(nopython=True)
def my_clip(x, xmin, xmax):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x
    
@njit(nopython=True)
def my_dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def compute3DFields(centerlines):
    
    R_omega = (1*1/50)**0.5
    h_omega = R_omega/10
    n_field,phi_field,e_field,S_field,cx,cy,cz = get_local_fields_over_domain(centerlines, R_omega,h_omega, 1, 1/50.)
    
    import polyscope as ps
    ps.init()

    dims = (n_field.shape[0], n_field.shape[1], n_field.shape[2])
    bound_low = (cx[0], cz[0], cy[0])
    bound_high = (cx[-1], cz[-1], cy[-1])

    ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)
    ps_grid.add_scalar_quantity("n_field", e_field, defined_on='nodes', enabled=True)
    
    ps.show()
    
def testLk():
    p1 = np.array([-100.,0,0])
    p2 = np.array([100.,0,0])
    p3 = np.array([0,-100.,1.])
    p4 = np.array([0,100.,1.])
    
    lk = compute_linking_number(p1,p2,p3,p4)
    print(f'Linking number: {lk}')
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'k')
    ax.plot([p3[0],p4[0]],[p3[1],p4[1]],[p3[2],p4[2]],'r')
    ax.axis('equal')
    
# %%
def main():
    
    centerlines = []
    for i in range(10):
        x = np.cumsum(np.random.randn(100))
        y = np.cumsum(np.random.randn(100))
        z = np.cumsum(np.random.randn(100))
        x = np.convolve(x, np.ones(5)/5, mode='valid')
        y = np.convolve(y, np.ones(5)/5, mode='valid')
        z = np.convolve(z, np.ones(5)/5, mode='valid')        
        centerlines.append(np.vstack([x,y,z]).T)
        
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for r in centerlines:
        ax.plot(r[:,0],r[:,1],r[:,2])
    
    point = np.array([0,0,0])
    R = 3
    rod_diameter = 0.01
    
    get_local_fields_at_a_point(centerlines, point, R, rod_diameter,visualize=True)
    
    h = R/2
    n_field, phi_field, e_field, S_field, center_x, center_y, center_z = get_local_fields_over_domain(centerlines, R, h, rod_diameter)
    
    mid_section = n_field.shape[2]//2
    
    fig,ax=plt.subplots(2,2)
    ax[0,0].imshow(n_field[:,:,mid_section])
    ax[0,0].set_title('Number of local curves')
    ax[0,1].imshow(phi_field[:,:,mid_section])
    ax[0,1].set_title('Local volume fraction')
    ax[1,0].imshow(e_field[:,:,mid_section])
    ax[1,0].set_title('Local average crossing number')
    ax[1,1].imshow(S_field[:,:,mid_section])
    ax[1,1].set_title('Local orientational order')
    # Try use Polyscope to visualize 3D graphics
    # https://polyscope.run/
    
if __name__ == '__main__':
    main()