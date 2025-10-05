# Function to create intersecting cylinders at each point
def create_crossing_cylinders(points, cylinder_length=0.05, cylinder_radius=0.01):
    all_vertices = []
    all_faces = []
    offset = 0
    
    for point in points:
        # Cylinder along x-axis
        # start_x = point - np.array([cylinder_length / 2, 0, 0])
        # end_x = point + np.array([cylinder_length / 2, 0, 0])
        # vertices_x, faces_x = create_cylinder(start_x, end_x, cylinder_radius)
        # faces_x += offset
        # offset += len(vertices_x)
        # all_vertices.append(vertices_x)
        # all_faces.append(faces_x)
        
        # Cylinder along y-axis
        start_y = point - np.array([0, cylinder_length / 2, 0])
        end_y = point + np.array([0, cylinder_length / 2, 0])
        vertices_y, faces_y = create_cylinder(start_y, end_y, cylinder_radius)
        faces_y += offset
        offset += len(vertices_y)
        all_vertices.append(vertices_y)
        all_faces.append(faces_y)
        
        # Cylinder along z-axis
        start_z = point - np.array([0, 0, cylinder_length / 2])
        end_z = point + np.array([0, 0, cylinder_length / 2])
        vertices_z, faces_z = create_cylinder(start_z, end_z, cylinder_radius)
        faces_z += offset
        offset += len(vertices_z)
        all_vertices.append(vertices_z)
        all_faces.append(faces_z)
        
    return np.vstack(all_vertices), np.vstack(all_faces)
Create the crossing cylinders for each point
vertices, faces = create_crossing_cylinders(points2)
Register the mesh in Polyscope
ps_crossing_cylinders = ps.register_surface_mesh("crossing_cylinders", vertices, faces, smooth_shade=False, enabled=True)
ps_crossing_cylinders.set_color((0.7, 0.2, 0.2))