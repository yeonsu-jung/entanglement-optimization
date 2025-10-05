# %%
import numpy as np
import polyscope as ps

# Define bounding box
bounding_box = np.array([[-0.5, 0.5],
                         [-0.5, 0.5],
                         [-1, 1.5]])

# Define floor vertices and faces
floor_vertices = np.array([
    [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]],
    [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 0]],
    [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 0]],
    [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 0]]])

# Each face is defined by the indices of its vertices
floor_faces = np.array([[0, 1, 2, 3]])

# Initialize polyscope
ps.init()
ps.set_SSAA_factor(3)
ps.set_navigation_style("free")

# Set ground plane settings
ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_darkness(0.5)
ps.set_ground_plane_height_factor(0.01)
ps.set_view_projection_mode("perspective")

# Register the surface mesh with polyscope
ps_floor_faces = ps.register_surface_mesh("floor_faces", floor_vertices, floor_faces, enabled=True, transparency=1)
ps_floor_faces.set_color((0.7, 0.7, 0.7))

# Show the polyscope window
ps.show()
