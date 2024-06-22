import numpy as np
import matplotlib.pyplot as plt
from cluster_of_filaments_in_unit_sphere import *

def get_bbox(curves):
    """
    Get bounding box of the curves.
    """
    bbox = np.array([np.min( np.vstack(curves),axis=0 ),np.max( np.vstack(curves),axis=0 )])
    return bbox

def get_curve_image(curves,image_size=500,thickness=3):
    """
    Get 3d array representing the volume image of the curves.
    """
    # scale curves
    
    
    volume_image = np.zeros((image_size, image_size, image_size))
    for curve in curves:
        for point in curve:
            x, y, z = point
            x = int((x + 1) * 50)
            y = int((y + 1) * 50)
            z = int((z + 1) * 50)
            volume_image[x, y, z] = 1
    
    return volume_image


# Parameters
M = 20  # Number of chains
N = 500  # Number of segments per chain
Lp = .20  # Persistence length
segment_length = 1  # Length of each segment

# Plotting the 3D curves
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Worm-Like Chains in 3D within Unit Sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Generate and plot M WLC chains in 3D
curves = []
for _ in range(M):
    wlc_3d = generate_wlc_3d(N, Lp, segment_length)
    # Smooth out with Gaussian kernel
    sigma = 10
    wlc_3d_smoothed = gaussian_filter1d(wlc_3d, sigma=sigma, axis=0)
    ax.plot(wlc_3d_smoothed[:, 0], wlc_3d_smoothed[:, 1], wlc_3d_smoothed[:, 2], lw=1)
    
    curves.append(wlc_3d_smoothed)

image_size = 300
bbox = get_bbox(curves)

scale_factor = np.floor(image_size/np.max(bbox[1] - bbox[0]))/1.5
# make curves fit in the image, scale and translate

all_curve_vertices = np.vstack(curves)
all_curve_vertices = all_curve_vertices - np.min(all_curve_vertices, axis=0)
all_curve_vertices *= scale_factor
all_curve_vertices = np.unique(all_curve_vertices, axis=0)

volume_image = np.zeros((image_size, image_size, image_size))
for each_point in all_curve_vertices:
    x, y, z = each_point
    x = int((x + 1))
    y = int((y + 1))
    z = int((z + 1))
    volume_image[x, y, z] = 1
    
# save it to matlab file
import scipy.io
scipy.io.savemat('volume_image.mat', {'volume_image': volume_image})


