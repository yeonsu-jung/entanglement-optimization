# %%
import numpy as np
import polyscope as ps

# Initialize polyscope
ps.init()
ps.set_ground_plane_mode("none")
ps.set_SSAA_factor(3)
# Function to create a cube
def create_cube(center, side_length=0.05):
    half_side = side_length / 2
    vertices = np.array([
        [center[0] - half_side, center[1] - half_side, center[2] - half_side],
        [center[0] + half_side, center[1] - half_side, center[2] - half_side],
        [center[0] + half_side, center[1] + half_side, center[2] - half_side],
        [center[0] - half_side, center[1] + half_side, center[2] - half_side],
        [center[0] - half_side, center[1] - half_side, center[2] + half_side],
        [center[0] + half_side, center[1] - half_side, center[2] + half_side],
        [center[0] + half_side, center[1] + half_side, center[2] + half_side],
        [center[0] - half_side, center[1] + half_side, center[2] + half_side]
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        [1, 2, 6], [1, 6, 5],  # Right
        [0, 3, 7], [0, 7, 4]   # Left
    ])
    
    return vertices, faces

# Function to create a square for shadow
def create_square(center, plane, side_length=0.05):
    half_side = side_length / 2
    if plane == "xy":
        vertices = np.array([
            [center[0] - half_side, center[1] - half_side, 0],
            [center[0] + half_side, center[1] - half_side, 0],
            [center[0] + half_side, center[1] + half_side, 0],
            [center[0] - half_side, center[1] + half_side, 0]
        ])
    elif plane == "yz":
        vertices = np.array([
            [0, center[1] - half_side, center[2] - half_side],
            [0, center[1] + half_side, center[2] - half_side],
            [0, center[1] + half_side, center[2] + half_side],
            [0, center[1] - half_side, center[2] + half_side]
        ])
    elif plane == "zx":
        vertices = np.array([
            [center[0] - half_side, 0, center[2] - half_side],
            [center[0] + half_side, 0, center[2] - half_side],
            [center[0] + half_side, 0, center[2] + half_side],
            [center[0] - half_side, 0, center[2] + half_side]
        ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    
    return vertices, faces

# Function to create cubes and their shadows
def create_cubes_and_shadows(points, cube_side_length=0.05):
    all_vertices = []
    all_faces = []
    shadow_vertices_xy = []
    shadow_faces_xy = []
    shadow_vertices_yz = []
    shadow_faces_yz = []
    shadow_vertices_zx = []
    shadow_faces_zx = []
    
    offset = 0
    shadow_offset_xy = 0
    shadow_offset_yz = 0
    shadow_offset_zx = 0
    
    for point in points:
        vertices, faces = create_cube(point, cube_side_length)
        faces += offset
        offset += len(vertices)
        all_vertices.append(vertices)
        all_faces.append(faces)
        
        # Create shadows on xy-plane
        v_xy, f_xy = create_square(point, "xy", cube_side_length)
        f_xy += shadow_offset_xy
        shadow_offset_xy += len(v_xy)
        shadow_faces_xy.append(f_xy)
        shadow_vertices_xy.append(v_xy)
        
        # Create shadows on yz-plane
        v_yz, f_yz = create_square(point, "yz", cube_side_length)
        f_yz += shadow_offset_yz
        shadow_offset_yz += len(v_yz)
        shadow_faces_yz.append(f_yz)
        shadow_vertices_yz.append(v_yz)
        
        # Create shadows on zx-plane
        v_zx, f_zx = create_square(point, "zx", cube_side_length)
        f_zx += shadow_offset_zx
        shadow_offset_zx += len(v_zx)
        shadow_faces_zx.append(f_zx)
        shadow_vertices_zx.append(v_zx)
    
    return (
        np.vstack(all_vertices), np.vstack(all_faces),
        np.vstack(shadow_vertices_xy), np.vstack(shadow_faces_xy),
        np.vstack(shadow_vertices_yz), np.vstack(shadow_faces_yz),
        np.vstack(shadow_vertices_zx), np.vstack(shadow_faces_zx)
    )

# random_points = np.random.rand(10, 3)
random_points = np.array([[0,0,0],[0,0,1]])
# Create the cubes and their shadows
(vertices, faces, 
 shadow_vertices_xy, shadow_faces_xy, 
 shadow_vertices_yz, shadow_faces_yz, 
 shadow_vertices_zx, shadow_faces_zx) = create_cubes_and_shadows(random_points,0.03)

# Register the mesh in Polyscope
ps_cubes = ps.register_surface_mesh("cubes", vertices, faces, smooth_shade=False, enabled=True)
ps_cubes.set_color((0.7, 0.2, 0.2))

# Register the shadows in Polyscope
# ps_shadow_xy = ps.register_surface_mesh("shadow_xy", shadow_vertices_xy, shadow_faces_xy, smooth_shade=False, enabled=True)
# ps_shadow_xy.set_color((1, 0, 0))
# ps_shadow_xy.set_transparency(0.5)

# ps_shadow_yz = ps.register_surface_mesh("shadow_yz", shadow_vertices_yz, shadow_faces_yz, smooth_shade=False, enabled=True)
# ps_shadow_yz.set_color((0, 1, 0))
# ps_shadow_yz.set_transparency(0.5)

# ps_shadow_zx = ps.register_surface_mesh("shadow_zx", shadow_vertices_zx, shadow_faces_zx, smooth_shade=False, enabled=True)
# ps_shadow_zx.set_color((0, 0, 1))
# ps_shadow_zx.set_transparency(0.5)

ps_cubes.set_enabled(True)
ps_point = ps.register_point_cloud("points", random_points, enabled=True, radius=0.015, color=(0,0,0))
ps_point.set_enabled(False)
# ps_point.set_color()

ps.look_at([0.5, 0.5, 0.7], [0,0,0])
ps.show()
# %%
