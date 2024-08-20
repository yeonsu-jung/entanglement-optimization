import numpy as np

def compute_kappa(vertices,m1,m2):
    tangents = np.diff(vertices,axis=0)    
    kb = 2 * np.cross(tangents[:-1], tangents[1:]) / (1 + np.sum(tangents[:-1] * tangents[1:], axis=1)[:, None])
    kappa_1 = np.sum(m1[:-1] * kb, axis=1)
    kappa_2 = np.sum(m2[:-1] * kb, axis=1)
    
    return np.concatenate(kappa_1,kappa_2)

def parallel_transport(d1_1, t1, t2):
    b = np.cross(t1, t2)
    
    if np.linalg.norm(b) == 0:
        return d1_1
    else:
        b = b / np.linalg.norm(b)
        b = b - np.dot(b, t1) * t1
        b = b / np.linalg.norm(b)
        b = b - np.dot(b, t1) * t2
        b = b / np.linalg.norm(b)
        
        n1 = np.cross(t1, b)
        n2 = np.cross(t2, b)
        
        d1_2 = np.dot(d1_1, t1) * t2 + np.dot(d1_1, n1) * n2 + np.dot(d1_1, b) * b
        d1_2 = d1_2 - np.dot(d1_2, t2) * t2
        d1_2 = d1_2 / np.linalg.norm(d1_2)
        
        return d1_2

def compute_space_parallel(tangent, ne):
    d1 = np.zeros_like(tangent)
    d2 = np.zeros_like(tangent)

    t0 = tangent[0]
    t1 = np.array([0.0, 0.0, -1.0])
    d1Tmp = np.cross(t0, t1)

    if np.abs(np.linalg.norm(d1Tmp)) < 1.0e-6:
        t1 = np.array([0.0, 1.0, 0.0])
        d1Tmp = np.cross(t0, t1)

    d1[0] = d1Tmp
    d2[0] = np.cross(t0, d1Tmp)

    for i in range(ne - 1):
        a = d1[i]
        b = tangent[i]
        c = tangent[i + 1]
        d = parallel_transport(a, b, c)
        d1[i + 1] = d
        d2[i + 1] = np.cross(c, d)
    
    return d1, d2


def rotate_axis_angle(v, z, theta):
    if theta != 0:
        cs = np.cos(theta)
        ss = np.sin(theta)
        v = cs * v + ss * np.cross(z, v) + np.dot(z, v) * (1.0 - cs) * z
    return v



def signed_angle(u, v, n):
    w = np.cross(u, v)
    angle = np.arctan2(np.linalg.norm(w), np.dot(u, v))
    if np.dot(n, w) < 0:
        return -angle
    else:
        return angle


def get_ref_twist(tangent, d1, ne, ref_twist_old):
    ref_twist = np.zeros(ne)
    
    for i in range(1, ne):
        u0 = d1[i - 1]
        u1 = d1[i]
        t0 = tangent[i - 1]
        t1 = tangent[i]

        ut = parallel_transport(u0, t0, t1)
        rotate_axis_angle(ut, t1, ref_twist_old[i])

        sgnAngle = signed_angle(ut, u1, t1)
        ref_twist[i] = ref_twist_old[i] + sgnAngle
    
    return ref_twist


def compute_twist_bar(x, ref_twist, ne):
    twist_bar = np.zeros(ne)
    for i in range(1, ne):
        theta_i = x[4 * (i - 1) + 3]
        theta_f = x[4 * i + 3]
        twist_bar[i] = theta_f - theta_i + ref_twist[i]
    return twist_bar


def compute_material_director(vertices,angle_list,m1,m2,d1,d2):
    ne = vertices.shape[0]
    for i in range(ne):
        # angle = x[4 * i + 3]
        angle = angle_list[i]
        cs = np.cos(angle)
        ss = np.sin(angle)
        m1[i, :] = cs * d1[i, :] + ss * d2[i, :]
        m2[i, :] = -ss * d1[i, :] + cs * d2[i, :]
        
        
import numpy as np

def reconstruct_t_curr(kb, t_prev):
    kb_norm = np.linalg.norm(kb)
    if kb_norm == 0:
        # No curvature, t_curr is aligned with t_prev
        return t_prev

    # Compute the rotation angle theta
    theta = 2 * np.arcsin(0.5 * kb_norm)
    
    # Normalize kb to use as rotation axis
    axis = kb / kb_norm
    
    # Use Rodrigues' rotation formula to rotate t_prev by theta around axis
    t_curr = (
        t_prev * np.cos(theta) +
        np.cross(axis, t_prev) * np.sin(theta) +
        axis * np.dot(axis, t_prev) * (1 - np.cos(theta))
    )
    
    return t_curr / np.linalg.norm(t_curr)

def reconstruct_vertices(kappa_1, kappa_2, initial_vertex, initial_tangent, arc_length):
    num_segments = len(kappa_1)
    vertices = np.zeros((num_segments + 1, 3))
    tangents = np.zeros((num_segments + 1, 3))
    
    # Initialize the first vertex and tangent
    vertices[0] = initial_vertex
    tangents[0] = initial_tangent / np.linalg.norm(initial_tangent)
    
    # Initialize material frame
    m1 = np.cross(tangents[0], np.array([0.0, 0.0, -1.0]))
    m1 /= np.linalg.norm(m1)
    m2 = np.cross(tangents[0], m1)
    
    for i in range(num_segments):
        # Compute the curvature binormal vector from kappa_1 and kappa_2
        curvature_binormal = kappa_1[i] * m1 + kappa_2[i] * m2
        
        # Reconstruct the next tangent vector
        tangents[i + 1] = reconstruct_t_curr(curvature_binormal, tangents[i])
        
        # Parallel transport the material frame
        m1 = parallel_transport(m1, tangents[i], tangents[i + 1])
        m2 = np.cross(tangents[i + 1], m1)
        m2 /= np.linalg.norm(m2)
        
        # Update the vertex position
        vertices[i + 1] = vertices[i] + tangents[i] * (arc_length[i+1] - arc_length[i])
    
    return vertices