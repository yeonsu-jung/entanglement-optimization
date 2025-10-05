# %%
import numpy as np
import polyscope as ps

# Initialize polyscope
ps.init()

# Create mesh grid
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
mg = np.meshgrid(x, y, indexing='ij')

# Compute surface values; Gaussian
surf = np.exp(-mg[0]**2 - mg[1]**2)

# Flatten the mesh grid and surface values
points = np.vstack([mg[0].ravel(), mg[1].ravel(), surf.ravel()]).T
values = surf.ravel()

# Create surface vertices and faces
def create_surface_mesh(points, values, threshold=0.):
    mask = values > threshold
    vertices = points[mask]
    faces = []

    # Assuming a structured grid, create faces by connecting adjacent vertices
    nx, ny = mg[0].shape
    for i in range(nx - 1):
        for j in range(ny - 1):
            idx = i * ny + j
            if mask[idx] and mask[idx + 1] and mask[idx + ny] and mask[idx + ny + 1]:
                # Calculate the indices in the masked array
                masked_indices = np.where(mask)[0]
                v0 = masked_indices[np.where(masked_indices == idx)[0][0]]
                v1 = masked_indices[np.where(masked_indices == idx + 1)[0][0]]
                v2 = masked_indices[np.where(masked_indices == idx + ny)[0][0]]
                v3 = masked_indices[np.where(masked_indices == idx + ny + 1)[0][0]]
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

    return vertices, np.array(faces)

# Create the surface mesh
surface_vertices, surface_faces = create_surface_mesh(points, values)

# Check if surface_faces is not empty before registering
if surface_faces.size > 0:
    ps_surface = ps.register_surface_mesh("surface", surface_vertices, surface_faces, smooth_shade=False, enabled=True)
    ps_surface.set_color((0.2, 0.7, 0.2))

# Show the polyscope visualization
ps.show()

