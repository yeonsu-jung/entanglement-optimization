import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from visualizations import set_3d_plot    
from scipy.spatial.transform import Rotation as R


def prep_svd_cylinder(cl,scale_factor = 1.5):
    N = len(cl)
    svd_cylinders = np.zeros((N,7))
    centroids = np.zeros((N,3))
    orientations = np.zeros((N,3))
    for i,c in enumerate(cl):
        center = c.mean(axis=0)
        centered = c - center
        
        u,s,v = np.linalg.svd(centered,full_matrices=False)
        orientation = v[0,:]
        slist = np.dot(centered, orientation)
        max_s = np.max(slist)
        min_s = np.min(slist)
        
        e1 = center + min_s*(v[0,:])*scale_factor 
        e2 = center + max_s*(v[0,:])*scale_factor 
        r1 = s[1]*max_s/s[0]*2*scale_factor
        
        dlist = (centered - slist[:,None]*orientation)
        dlist = np.linalg.norm(dlist,axis=1)
        r1 = np.max(dlist)
        
        # Eigen::VectorXd distances = (filament - projection * principal_orientation.transpose()).rowwise().norm();
        
        
        svd_cylinders[i,:] = np.hstack((e1,e2,r1))
        centroids[i,:] = center
        orientations[i,:] = orientation
        
    return svd_cylinders,centroids,orientations

def circlefit_rod(rr,linearity_threshold, radius_curvature_threshold):
    n = rr.shape[0]  # number of data points
    if n <= 2:
        return rr

    # Main fitting process for n > 2
    centroid = np.mean(rr, axis=0) ###
    rr_centered = rr - centroid

    U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    rr_sorted = rr_centered[sorted_indices]
    svd_line = slist[:, None] * orientation
    line_fit_error = rr_sorted - svd_line
    
    # Circle fit using Taubin's method for n > 2
    # Project points onto v1 and v2
    rr_projected = rr_sorted @ np.column_stack([v1, v2])
    local_cen = np.mean(rr_projected, axis=0)
    xm,ym,r0 = CircleFitByTaubin(rr_projected,local_cen) # numba jit make it 10 times faster.
    
    if np.isnan(r0):
        s1, s2 = np.min(slist), np.max(slist)
        best_estimation = centroid + np.outer(slist, v1)
        return best_estimation
    
    x_projected = rr_sorted @ v1 - xm
    y_projected = rr_sorted @ v2 - ym
    phi_list = np.arctan2(y_projected, x_projected)
    phi_list = np.unwrap(phi_list)
    xpt = xm + r0 * np.cos(phi_list)
    ypt = ym + r0 * np.sin(phi_list)
    
    # if (np.sqrt(np.mean(line_fit_error**2)) < linearity_threshold) or (r0 > radius_curvature_threshold):
    #     s1, s2 = np.min(slist), np.max(slist)
    #     best_estimation = centroid + np.outer(slist, v1)
    #     return best_estimation
    
    best_estimation = centroid + np.outer(xpt, v1) + np.outer(ypt, v2)
    return best_estimation

@jit(nopython=True)
def CircleFitByTaubin(XY,centroid):
    # utilizing 'frenet_robust' code
    # https://www.mathworks.com/matlabcentral/fileexchange/47885-frenet_robust-zip
    # G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
    #             Space Curves Defined By Implicit Equations, With 
    #             Applications To Edge And Range Image Segmentation",
    # IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    n = len(XY)  # number of data points
    # centroid = np.mean(XY, axis=0)  # the centroid of the data set

    # Computing moments (normalized by n)
    Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0

    for i in range(n):
        Xi = XY[i, 0] - centroid[0]  # centering data
        Yi = XY[i, 1] - centroid[1]  # centering data
        Zi = Xi**2 + Yi**2
        Mxy += Xi * Yi
        Mxx += Xi**2
        Myy += Yi**2
        Mxz += Xi * Zi
        Myz += Yi * Zi
        Mzz += Zi**2

    Mxx /= n
    Myy /= n
    Mxy /= n
    Mxz /= n
    Myz /= n
    Mzz /= n

    # Computing the coefficients of the characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy**2
    A3 = 4 * Mz
    A2 = -3 * Mz**2 - Mzz
    A1 = Mzz * Mz + 4 * Cov_xy * Mz - Mxz**2 - Myz**2 - Mz**3
    A0 = (Mxz**2 * Myy + Myz**2 * Mxx - Mzz * Cov_xy -
          2 * Mxz * Myz * Mxy + Mz**2 * Cov_xy)
    A22 = A2 + A2
    A33 = A3 + A3 + A3

    xnew = 0
    ynew = 1e+20
    epsilon = 1e-12
    IterMax = 20

    # Newton's method starting at x=0
    for iter in range(IterMax):
        yold = ynew
        ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3))
        if abs(ynew) > abs(yold):
            xnew = 0
            break
        Dy = A1 + xnew * (A22 + xnew * A33)
        xold = xnew
        xnew = xold - ynew / Dy
        if xnew < 1e-15:
            break
        if abs((xnew - xold) / xnew) < epsilon:
            break
        if iter >= IterMax and abs(xnew) > epsilon:
            # print('Newton-Taubin will not converge')
            xnew = 0
        if xnew < 0 and abs(xnew) > epsilon:
            # print(f'Newton-Taubin negative root: x={xnew}')
            xnew = 0

    # Computing the circle parameters
    DET = xnew**2 - xnew * Mz + Cov_xy
    if DET == 0:
        # print('Too flat')
        return np.nan,np.nan,np.nan
        
        
    Center = np.array([Mxz * (Myy - xnew) - Myz * Mxy,
                       Myz * (Mxx - xnew) - Mxz * Mxy]) / DET / 2
    
    
    
    # Par = np.hstack((Center + centroid, np.sqrt(np.dot(Center, Center) + Mz)))

    return Center[0] + centroid[0], Center[1] + centroid[1], np.sqrt(np.dot(Center, Center) + Mz)

# @jit(nopython=True,fastmath=True)
def fit_rod_light(rr_centered,centroid,linearity_threshold, radius_curvature_threshold):
    n = rr.shape[0]  # number of data points
    if n <= 2:
        return rr_centered

    # Main fitting process for n > 2
    # SVD for least-squares fit of coalescing plane
    U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr_centered), orientation)
    sorted_indices = np.argsort(slist)
    rr_sorted = rr_centered[sorted_indices]
    svd_line = slist[:, None] * orientation    
    # Circle fit using Taubin's method for n > 2
    # Project points onto v1 and v2
    xx_projected = rr_sorted @ v1
    yy_projected = rr_sorted @ v2
    x1 = np.mean(xx_projected)
    x2 = np.mean(yy_projected)
    # local_cen = np.mean(rr_projected, axis=0)
    local_cen = 1
    
    
    xm,ym,r0 = CircleFitByTaubin(rr_projected,local_cen) # numba jit make it 10 times faster.
    
    if np.isnan(r0):
        s1, s2 = np.min(slist), np.max(slist)
        r1 = centroid + s1 * v1
        r2 = centroid + s2 * v1
        best_estimation = centroid + np.outer(slist, v1)
        return best_estimation
    
    x_projected = rr_sorted @ v1 - xm
    y_projected = rr_sorted @ v2 - ym
    # Compute atan2 for these projections and unwrap phi_list to prevent discontinuities
    phi_list = np.arctan2(y_projected, x_projected)
    phi_list = np.unwrap(phi_list)
    xpt = xm + r0 * np.cos(phi_list)
    ypt = ym + r0 * np.sin(phi_list)    
    best_estimation = centroid + np.outer(xpt, v1) + np.outer(ypt, v2)    
    return best_estimation

def fit_rod(rr,linearity_threshold, radius_curvature_threshold):
    n = rr.shape[0]  # number of data points
    if rr.size == 0:
        return create_output(np.nan,[], np.inf, np.nan, [], [], 0, rr, 0, 0)
    if n == 1:
        temp = rr / np.linalg.norm(rr, axis=1, keepdims=True)
        return create_output(np.nan,rr, np.inf, np.tile(temp, (1, 3)), [], [], 0, rr, 0, 0)
    if n == 2:
        cen = np.mean(rr, axis=0)
        ori = (rr[1, :] - rr[0, :]) / np.linalg.norm(rr[1, :] - rr[0, :])
        len_rod = np.linalg.norm(rr[1, :] - rr[0, :])
        slopes = np.vstack((-ori,ori))
        return create_output(slopes,cen, np.inf, np.column_stack([ori]*3), linspace_vector(rr[0], rr[1], 1000), [], len_rod, rr, 0, 0)

    # Main fitting process for n > 2
    centroid = np.mean(rr, axis=0) ###
    rr_centered = rr - centroid

    # SVD for least-squares fit of coalescing plane
    U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    rr_sorted = rr_centered[sorted_indices]
    svd_line = slist[:, None] * orientation
    line_fit_error = rr_sorted - svd_line
    
    # Circle fit using Taubin's method for n > 2
    # Project points onto v1 and v2
    rr_projected = rr_sorted @ np.column_stack([v1, v2])
    local_cen = np.mean(rr_projected, axis=0)
    xm,ym,r0 = CircleFitByTaubin(rr_projected,local_cen) # numba jit make it 10 times faster.
    
    if np.isnan(r0):
        slist = np.dot((rr - centroid), orientation)
        s1, s2 = np.min(slist), np.max(slist)
        r1 = centroid + s1 * v1
        r2 = centroid + s2 * v1
        best_estimation = centroid + np.outer(slist, orientation)
        err = np.mean( np.sqrt(np.sum((rr - best_estimation)**2,axis=1)) )
        
        slopes = np.vstack((-orientation,orientation))
        
        return create_output(slopes,centroid, np.inf, np.column_stack([v1, v2, v3]), linspace_vector(r1, r2, 1000), slist, s2 - s1, best_estimation, np.linalg.norm(err), 0)
    
    x_projected = rr_sorted @ v1 - xm
    y_projected = rr_sorted @ v2 - ym
    # Compute atan2 for these projections and unwrap phi_list to prevent discontinuities
    phi_list = np.arctan2(y_projected, x_projected)
    phi_list = np.unwrap(phi_list)

    min_th = np.min(phi_list)
    max_th = np.max(phi_list)
    xpt = xm + r0 * np.cos(phi_list)
    ypt = ym + r0 * np.sin(phi_list)
    

    circle_fit_error = np.vstack((x_projected - r0*np.cos(phi_list),y_projected - r0*np.sin(phi_list))).T
    th = np.linspace(min_th,max_th,phi_list.size//3)
    xpt2 = xm+r0*np.cos(th)
    ypt2 = ym+r0*np.sin(th)
    
    # end slopes
    end_slope1_proj = -r0*np.sin(phi_list[-1])
    end_slope2_proj = r0*np.cos(phi_list[-1])
    
    # start slopes
    start_slope1_proj = -r0*np.sin(phi_list[0])
    start_slope2_proj = r0*np.cos(phi_list[0])
    
    start_slope_3d = start_slope1_proj*v1 + start_slope2_proj*v2
    start_slope_3d = start_slope_3d/np.linalg.norm(start_slope_3d)
    start_slope_3d *= -np.sign(np.sum(start_slope_3d * (rr_centered[-1, :] - rr_centered[0, :])))    
    
    end_slope_3d = end_slope1_proj*v1 + end_slope2_proj*v2
    end_slope_3d = end_slope_3d/np.linalg.norm(end_slope_3d)
    end_slope_3d *= +np.sign(np.sum(end_slope_3d * (rr_centered[-1, :] - rr_centered[0, :])))
    slopes = np.vstack((start_slope_3d,end_slope_3d))
    
    if  (np.sqrt(np.mean(line_fit_error**2)) < linearity_threshold) or (r0 > radius_curvature_threshold) or (np.sqrt(np.mean(line_fit_error**2)) < np.sqrt(np.mean(circle_fit_error**2))):
        # print(f'Linearity error: {np.sqrt(np.mean(line_fit_error**2))} < {linearity_threshold}')
        # print(f'Curvature error: {np.sqrt(np.mean(circle_fit_error**2))}')
        
        s1, s2 = np.min(slist), np.max(slist)
        r1 = centroid + s1 * v1
        r2 = centroid + s2 * v1
        best_estimation = centroid + np.outer(slist, orientation)
        err = np.mean( np.sqrt(np.sum((rr - best_estimation)**2,axis=1)) )
        planarity = np.mean((rr_sorted @ v3)**2)
        return create_output(slopes,centroid, np.inf, np.column_stack([v1, v2, v3]), linspace_vector(r1, r2, 1000), slist, s2 - s1, best_estimation, np.linalg.norm(err), planarity)
    
    even_reconstruction =  centroid + np.outer(xpt2, v1) + np.outer(ypt2, v2)
    best_estimation = centroid + np.outer(xpt, v1) + np.outer(ypt, v2)
    err = np.mean( np.sqrt(np.sum((centroid + rr_sorted - best_estimation)**2,axis=1)) )
    
    # r0 * (np.max(phi_list) - np.min(phi_list))
    rod_length = np.sum( np.diff(rr_sorted,axis=0)**2, axis=1).sum()**0.5
    
    z_projected = rr_sorted @ v3
    planarity = np.mean(z_projected**2)
    
    return create_output(slopes,centroid, r0, np.column_stack([v1, v2, v3]), even_reconstruction, phi_list, rod_length, even_reconstruction, err, planarity)

# @jit(nopython=True)
def fit_rod_error(rr_centered, DEBUG_FLAG=False):
    n = rr_centered.shape[0]  # number of data points

    # SVD for least-squares fit of coalescing plane
    _, S, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1, v2, v3 = V[:,0], V[:,1], V[:,2]

    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr_centered), orientation)
    sorted_indices = np.argsort(slist)
    rr_sorted = rr_centered[sorted_indices]
    
    # if DEBUG_FLAG:
        # plt.scatter(np.dot(rr_sorted, v1), np.dot(rr_sorted, v2), marker='o')
        # plt.axis('equal')
        # plt.show()

    # Check if points are co-linear
    if np.max(np.abs(np.cross(rr_sorted[1:] - rr_sorted[:-1], v3))) < 1e-10:
        s1, s2 = np.min(slist), np.max(slist)
        r1 = s1 * v1
        r2 = s2 * v1
        best_estimation = np.outer(slist, v1)
        err = np.sum((rr_centered - best_estimation)**2, axis=1)
        return err

    # Circle fit using Taubin's method for n > 2
    # Project points onto v1 and v2
    
    rr_projected = rr_sorted @ np.column_stack([v1, v2])
    local_cen = np.mean(rr_projected, axis=0)
    
    xm,ym,r0 = CircleFitByTaubin(rr_projected,local_cen) # numba jit make it 10 times faster.
    
    x_projected = rr_sorted @ v1 - xm
    y_projected = rr_sorted @ v2 - ym
    # Compute atan2 for these projections
    phi_list = np.arctan2(y_projected, x_projected)
    # Unwrap phi_list to prevent discontinuities
    phi_list = np.unwrap(phi_list)

    # plt.plot(x_projected, y_projected, 'o')
    # plt.plot(r0 * np.cos(phi_list), r0 * np.sin(phi_list), 'r')
    
    xpt = xm + r0 * np.cos(phi_list)
    ypt = ym + r0 * np.sin(phi_list)
    circle_points3D = np.outer(xpt, v1) + np.outer(ypt, v2)
    
    # best_estimation = x_projected[:,None] * v1 + y_projected[:,None] * v2
    best_estimation = np.outer(xpt, v1) + np.outer(ypt, v2)
    err = np.mean( np.sqrt(np.sum((centroid + rr_sorted - best_estimation)**2,axis=1)) )
    
    # fig,ax = set_3d_plot()
    # plot_3d(rr_sorted,ax, 'Original points', {'marker': '.', 'color': 'b'})
    # plot_3d(circle_points3D, ax,'Rod fitting visualization', {'color': 'r', 'marker': '.'})
    
    return err



def create_output(slopes,cen, r, u, pts, philist, len_rod, rec, err, planarity):
    return {
        'cen': cen,  # Center of the fitted model
        'r': r,      # Radius of the circle (or inf for line)
        'u': u,      # Principal directions (vectors)
        'pts': pts,  # Best estimation of points
        'philist': philist,  # Angular coordinates of points
        'len': len_rod,      # Length of the rod or circumference
        'rec': rec,  # Reconstructed points
        'err': err,   # Errors of fit
        'slopes': slopes,
        'planarity': planarity
    }

def linspace_vector(start, end, num):
    return np.linspace(start, end, num)

def plot_3d(points, ax, title, params={}):
    ax.plot(points[:, 0], points[:, 1], points[:, 2],**params,label=title)

def test():
    # Test the function with some random data
    np.random.seed(0)
    # Generate smooth curve
    t = np.linspace(0, np.pi/4, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    rr = np.vstack([x, y, z]).T
    
    from visualizations import set_3d_plot
    # fig,ax = set_3d_plot()
    # ax.plot(rr[:, 0], rr[:, 1], rr[:, 2],'o',color='b')
    # plt.show()
    
    result = fit_rod(rr,DEBUG_FLAG=True)
    print(result)
    
    fig,ax = set_3d_plot()
    ax.plot(result['ref'][:, 0], result['ref'][:, 1], result['ref'][:, 2],'o',color='r')
    ax.plot(rr[:, 0], rr[:, 1], rr[:, 2],'o',color='b')
    plt.show()
    
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_principal_axis_length(rr):
    if rr.size == 0:
        return {}

    if rr.shape[0] == 1:
        return {
            'Centroid': rr,
            'PrincipalAxisLength': np.array([0, 0, 0]),
            'Orientation': np.array([0, 0, 0]),
            'EigenValues': np.array([0, 0, 0]),
            'EigenVectors': np.array([0, 0, 0]),
        }
    
    centroid = np.mean(rr, axis=0)
    centered = rr - centroid
    mu000 = np.sum(centered ** 0)
    mu200 = np.sum(centered[:, 0] ** 2) / mu000 + 1/12
    mu020 = np.sum(centered[:, 1] ** 2) / mu000 + 1/12
    mu002 = np.sum(centered[:, 2] ** 2) / mu000 + 1/12
    mu110 = np.sum(centered[:, 0] * centered[:, 1]) / mu000
    mu011 = np.sum(centered[:, 1] * centered[:, 2]) / mu000
    mu101 = np.sum(centered[:, 2] * centered[:, 0]) / mu000
    
    num_points = rr.shape[0]
    cov_mat = np.array([
        [mu200, mu110, mu101],
        [mu110, mu020, mu011],
        [mu101, mu011, mu002]
    ]) / num_points
    
    U, S, _ = np.linalg.svd(cov_mat)
    indices = np.argsort(-S)
    S = S[indices]
    U = U[:, indices]
    
    if U[0, 0] < 0:
        U = -U
        U[:, 2] = -U[:, 2]
    
    V, D = np.linalg.eig(cov_mat)
    indices = np.argsort(-D)
    V = V[:, indices]
    D = D[indices]
    
    return {
        'Centroid': centroid,
        'PrincipalAxisLength': 4 * np.sqrt(S * num_points),
        'Orientation': R.from_matrix(U).as_euler('xyz', degrees=True),
        'EigenValues': D * num_points,
        'EigenVectors': V
    }

# Example usage:
# points = np.random.rand(10, 3)  # Replace with your actual data
# stats = get_principal_axis_length(points)
# print(stats)

    
if __name__ == '__main__':
    # test()    
    
    # 3d line
    noise_level = 0.0001
    
    
    x = np.linspace(0,1,30)
    y = np.linspace(0,1,30)
    z = np.linspace(0,1,30)
    rr = np.vstack([x,y,z]).T
    
    noise = np.random.randn(*rr.shape)*noise_level
    np.max(np.sqrt(np.sum(noise**2,axis=1)))
    
    rr = rr + noise
    
    # y = np.cos(x)
    # z = np.sin(x)
    # rr = np.vstack([x,y,z]).T
    # rr = rr + noise
    
    # np.max(np.sqrt(np.sum(noise**2,axis=1)))
    # np.median(np.sqrt(np.sum(noise**2,axis=1)))
    # np.mean(np.sqrt(np.sum(noise**2,axis=1)))
    
    # np.mean(noise)
    # np.median(noise)
    # np.max(noise)
    
    # 3d plot
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'o')
    
    linearity_threshold = 0.01
    radius_curvature_threshold = 100
    
    fr1 = fit_rod(rr,linearity_threshold, radius_curvature_threshold)
    fr1['err']
    fr1['len']

    
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid
    # fr2 = fit_rod_light(rr_centered,centroid,linearity_threshold, radius_curvature_threshold)
    
    # fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    # ax.plot(rr[:,0],rr[:,1],rr[:,2],'o')
    # ax.plot(fr1['rec'][:,0],fr1['rec'][:,1],fr1['rec'][:,2],'o')
    # ax.plot(fr2[:,0],fr2[:,1],fr2[:,2],'o')
    
    
    
    
    # alignment_error_matrix
    # fitting_error_matrix
    
    # 3 seconds
    
    
    
    print()
    

