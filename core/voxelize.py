import numpy as np
from typing import Tuple

def _distance_point_to_segment(point, seg_start, seg_end):
    """Calculates the shortest distance from a point to a line segment."""
    line_vec = seg_end - seg_start
    point_vec = point - seg_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0:
        return np.linalg.norm(point_vec)
        
    # Project point_vec onto line_vec
    t = np.dot(point_vec, line_vec) / line_len_sq
    
    # Clamp t to [0, 1] to stay on the segment
    t_clamped = np.clip(t, 0, 1)
    
    # Find the closest point on the segment
    closest_point = seg_start + t_clamped * line_vec
    
    return np.linalg.norm(point - closest_point)


def voxelize_rods(
    q: np.ndarray,
    rod_diameter: float,
    volume_shape: Tuple[int, int, int],
    box_bounds: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Converts a set of rod configurations into a 3D volume image.

    Args:
        q: Rod configurations from `create_nonintersecting_rods`, shape (N, 5).
        rod_diameter: The diameter of the rods.
        volume_shape: The desired output resolution (nx, ny, nz) of the volume.
        box_bounds: A tuple (min_coords, max_coords) defining the physical
                    space the volume represents.

    Returns:
        A 3D NumPy array (volume) with 1s for voxels occupied by rods and 0s otherwise.
    """
    num_rods = q.shape[0]
    rod_radius = rod_diameter / 2.0
    
    box_min, box_max = box_bounds
    voxel_size = (box_max - box_min) / np.array(volume_shape)
    
    # Create an empty volume
    volume = np.zeros(volume_shape, dtype=np.int8)
    
    print(f"Voxelizing {num_rods} rods into a {volume_shape} grid...")
    
    # Iterate through each rod
    for i in range(num_rods):
        # 1. Get rod start and end points
        p_start = q[i, :3]
        phi, theta = q[i, 3], q[i, 4]
        orientation = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        p_end = p_start + orientation
        
        # 2. Define a bounding box around the rod to check only relevant voxels
        min_bb_world = np.minimum(p_start, p_end) - rod_radius
        max_bb_world = np.maximum(p_start, p_end) + rod_radius

        # 3. Convert world bounding box to voxel index ranges
        min_idx = np.floor((min_bb_world - box_min) / voxel_size).astype(int)
        max_idx = np.ceil((max_bb_world - box_min) / voxel_size).astype(int)
        
        # Clip indices to be within the volume's bounds
        min_idx = np.maximum(0, min_idx)
        max_idx = np.minimum(np.array(volume_shape), max_idx)

        # 4. Iterate over voxels within the rod's bounding box
        for vz in range(min_idx[2], max_idx[2]):
            for vy in range(min_idx[1], max_idx[1]):
                for vx in range(min_idx[0], max_idx[0]):
                    # Calculate the center of the current voxel
                    voxel_center = box_min + np.array([vx + 0.5, vy + 0.5, vz + 0.5]) * voxel_size
                    
                    # 5. If voxel center is within the rod's radius, mark it as filled
                    dist = _distance_point_to_segment(voxel_center, p_start, p_end)
                    if dist <= rod_radius:
                        volume[vx, vy, vz] = 1

    print("Voxelization complete.")
    return volume

