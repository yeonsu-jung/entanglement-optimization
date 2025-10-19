# x,y,z,theta,phi
# ddg
# distance constraint
from jax import jit, lax, grad, vmap, config
import jax.numpy as jnp

# from jax.config import config
# config.update("jax_enable_x64", True)
# todo: remove pair, triple, ,,, etc..
config.update("jax_enable_x64", True)
print("64-bit enabled:", config.read("jax_enable_x64"))




@jit
def rotate_vector(v, t, theta):

    cs = jnp.cos(theta)
    ss = jnp.sin(theta)
    # return cs * v + ss * t.cross(v) + t.dot(v) * (1.0 - cs) * t

    return jnp.where( theta == 0, v, cs * v + ss * jnp.cross(t, v) + jnp.dot(t, v) * (1.0 - cs) * t )

@jit
def signed_angle(v1, v2, n):
    w = jnp.cross(v1, v2)
    angle = jnp.arctan2(jnp.linalg.norm(w), jnp.dot(v1, v2))
    return jnp.where(jnp.dot(n, w) < 0, -angle, angle)

@jit
def get_lengths_for_a_curve(positions):
    tangents = positions[:-1] - positions[1:]
    return jnp.sqrt( jnp.sum(tangents**2,axis=1) )

@jit
def get_material_frame(curr_bishop_frames,ref_twists):
    ref_twists = ref_twists[:, None]
    m1 =  jnp.cos(ref_twists) * curr_bishop_frames[:,1, :] + jnp.sin(ref_twists) * curr_bishop_frames[:,2, :]
    m2 = -jnp.sin(ref_twists) * curr_bishop_frames[:,1, :] + jnp.cos(ref_twists) * curr_bishop_frames[:,2, :]
    return jnp.stack((m1,m2),axis=1)


# batch_parallel_transport = jit(vmap(parallel_transport))
# batch_rotate_vector = jit(vmap(rotate_vector))
# batch_signed_angle = jit(vmap(signed_angle))

@jit
def get_twists(curr_bishop_frames, previous_ref_twists):

    twists = jnp.zeros( curr_bishop_frames.shape[0] )
    
    t21 = curr_bishop_frames[:-1, 0, :] # t0
    t22 = curr_bishop_frames[1:, 0, :]  # t1

    u21 = curr_bishop_frames[:-1, 1, :] # u0
    u22 = curr_bishop_frames[1:, 1, :]  # u1

    ut = batch_parallel_transport(u21,t21,t22)
    ut = batch_rotate_vector(ut, t22, previous_ref_twists[1:])

    twists = previous_ref_twists[1:] + batch_signed_angle(ut,u22,t22)

    # pad zero at the first entry
    twists = jnp.concatenate([jnp.array([0.0]), twists], axis=0)
    # twists = jnp.concatenate([twists,jnp.array([0.0])], axis=0)
    
    return twists

@jit
def time_parallel(prev_bishop_frames,curr_positions):

    # jdb.print("{x}, {y}",
    #           x=prev_bishop_frames.shape[0],
    #           y=curr_positions.shape[0])

    batch_parallel_transport = jit(vmap(parallel_transport))
    # current tangents
    t_curr = curr_positions[1:] - curr_positions[:-1]
    t_curr = t_curr/jnp.linalg.norm(t_curr,axis=1,keepdims=True)

    # parallel_transport
    u_curr = batch_parallel_transport(prev_bishop_frames[:,1,:],prev_bishop_frames[:,0,:],t_curr)
    # v_curr = batch_parallel_transport(prev_bishop_frames[:,2,:],prev_bishop_frames[:,0,:],t_curr)
    v_curr = jnp.cross(t_curr,u_curr)
    
    return jnp.stack((t_curr,u_curr,v_curr),axis=1)

@jit
def get_curvatures_for_a_curve(bishop_frames):
    """
    Compute curvature vectors along a curve given bishop frames.
    
    Args:
        bishop_frames: Array of shape (N, 3, 3) where each frame is [t, u, v].
                    The tangent is the first row of each frame.
    
    Returns:
        kb: Array of shape (N, 3) containing curvature vectors.
            The last entry is set to zeros (since we have N-1 segments).
    """
    # Extract tangent vectors for each segment.
    # t1 is the tangent for the current segment (from bishop_frames[i])
    # t2 is the tangent for the next segment (from bishop_frames[i+1])
    t1 = bishop_frames[:-1, 0, :]  # shape (N-1, 3)
    t2 = bishop_frames[1:, 0, :]   # shape (N-1, 3)
    
    # Compute the cross product and dot product for each pair of tangents.
    cross_vals = jnp.cross(t1, t2)        # shape (N-1, 3)
    dot_vals = jnp.sum(t1 * t2, axis=1)     # shape (N-1,)
    
    # Compute curvature using the formula:
    #   curvature = 2 * cross(t1, t2) / (1 + dot(t1, t2))
    curvature = 2 * cross_vals / (1.0 + dot_vals)[:, None]  # shape (N-1, 3)
    
    # Append a zero curvature for the last frame (since we have N-1 segments)
    # curvature = jnp.concatenate([curvature, jnp.zeros((1, 3))], axis=0)
    
    return curvature

@jit
def get_curvatures_for_a_curve(bishop_frames):
    
    """
    Compute curvature vectors along a curve given bishop frames.
    
    Args:
        bishop_frames: Array of shape (N, 3, 3) where each frame is [t, u, v].
                    The tangent is the first row of each frame.
    
    Returns:
        kb: Array of shape (N, 3) containing curvature vectors.
            The last entry is set to zeros (since we have N-1 segments).
    """
    # Extract tangent vectors for each segment.
    # t1 is the tangent for the current segment (from bishop_frames[i])
    # t2 is the tangent for the next segment (from bishop_frames[i+1])
    t1 = bishop_frames[:-1, 0, :]  # shape (N-1, 3)
    t2 = bishop_frames[1:, 0, :]   # shape (N-1, 3)
    
    # Compute the cross product and dot product for each pair of tangents.
    cross_vals = jnp.cross(t1, t2)        # shape (N-1, 3)
    dot_vals = jnp.sum(t1 * t2, axis=1)     # shape (N-1,)
    
    # Compute curvature using the formula:
    #   curvature = 2 * cross(t1, t2) / (1 + dot(t1, t2))
    curvature = 2 * cross_vals / (1.0 + dot_vals)[:, None]  # shape (N-1, 3)
    
    # Append a zero curvature for the last frame (since we have N-1 segments)
    # curvature = jnp.concatenate([curvature, jnp.zeros((1, 3))], axis=0)
    
    return curvature

@jit
def get_bishop_frames(positions):
    """
    Compute bishop frames along a linear sequence of positions.
    
    Each segment (from positions[i] to positions[i+1]) defines a tangent, and
    the bishop frame is computed by parallel transporting the frame along the curve.
    
    Args:
        positions: Array of shape (N, 3) representing positions along a curve.
    
    Returns:
        bishop_frames: Array of shape (N-1, 3, 3) where each frame is [t, u, v]
                       for the corresponding segment.
    """
    num_segments = positions.shape[0] - 1  # There are N-1 segments

    # Compute initial tangent (segment 0)
    p0 = positions[0]
    p1 = positions[1]
    t0 = (p1 - p0) / jnp.linalg.norm(p1 - p0)
    
    # Choose an arbitrary reference vector v0
    
    # Compute u0 so that (t0, u0, v0) forms a right-handed frame.
    u0 = jnp.cross(t0, jnp.array([0.0, 0.0, -1.0]))
    u0 = u0 / jnp.linalg.norm(u0)
    v0 = jnp.cross(t0, u0)
    v0 /= jnp.linalg.norm(v0)

    # Create initial frame for the first segment
    frame0 = jnp.stack([t0, u0, v0])
    
    # Prepare pairs of positions for the remaining segments.
    # For i from 1 to num_segments-1, we use (positions[i], positions[i+1]).
    pos_pairs = jnp.stack([positions[1:-1], positions[2:]], axis=1)

    def scan_fn(carry, pos_pair):
        t_prev, u_prev = carry
        p_curr, p_next = pos_pair
        
        # Compute the tangent for the current segment
        t_curr = (p_next - p_curr) / jnp.linalg.norm(p_next - p_curr)
        # Parallel transport u_prev to obtain u_curr
        u_curr = parallel_transport(u_prev, t_prev, t_curr)
        v_curr = jnp.cross(t_curr, u_curr)
        
        new_carry = (t_curr, u_curr)
        frame = jnp.stack([t_curr, u_curr, v_curr])
        return new_carry, frame

    init_carry = (t0, u0)
    # Run the scan over the remaining segments (if any)
    _, frames = lax.scan(scan_fn, init_carry, pos_pairs)
    
    # Combine the initial frame with the frames from the scan
    bishop_frames = jnp.concatenate([frame0[None, ...], frames], axis=0)
    return bishop_frames


@jit
def space_parallel(positions, a_vector):
    # Compute normalized tangents from positions
    tangents = positions[1:] - positions[:-1]
    tangents = tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)
    
    ne = tangents.shape[0]
    d1 = jnp.zeros_like(tangents)
    d2 = jnp.zeros_like(tangents)
    
    # Initialize the first frame using cross products.
    m1 = jnp.cross(tangents[0], a_vector)
    m2 = jnp.cross(tangents[0], m1)
    d1 = d1.at[0].set(m1)
    d2 = d2.at[0].set(m2)
    
    def body(i, val):
        d1, d2 = val
        # Previous frame and tangents for the transport step.
        a = d1[i]
        b = tangents[i]
        c = tangents[i + 1]
        d = parallel_transport(a, b, c)
        # Update the frames at index i+1.
        d1 = d1.at[i + 1].set(d)
        d2 = d2.at[i + 1].set(jnp.cross(c, d))
        return (d1, d2)
    
    d1, d2 = lax.fori_loop(0, ne - 1, body, (d1, d2))
    return jnp.stack([tangents, d1, d2], axis=1)

@jit
def get_ref_twist(ref_position_triple,cur_position_triple, previous_u_vector, previous_twist):
    
    x11,x12,x13 = ref_position_triple
    x21,x22,x23 = cur_position_triple

    t11 = x12 - x11
    t12 = x13 - x12

    t21 = x22 - x21 # t0
    t22 = x23 - x22 # t1

    u12 = parallel_transport(previous_u_vector,t11,t12)
    u21 = parallel_transport(previous_u_vector,t11,t21)   # u0
    u22 = parallel_transport(u12,t12,t22)   # u1    
    ut = parallel_transport(u21,t21,t22)

    # rotate u21 by (t21 -> t22)
    ut = rotate_vector(ut, t22, previous_twist)
    return previous_twist + signed_angle(ut,u22,t22)

@jit
def get_local_curvature(curvature_global,frame):
    t1,u1,v1,t2,u2,v2 = frame
    # assert( jnp.abs(jnp.dot(curvature_global, (t1+))) < 1e-6 )
    u = (u1+u2)/2
    v = (v1+v2)/2
    return jnp.array([jnp.dot(curvature_global,v),-jnp.dot(curvature_global,u)])



@jit
def rotation_matrix_between_vectors(b, a=jnp.array([0.0, 0.0, 1.0]), alpha=None):
    """
    Returns a rotation matrix that rotates unit vector b to unit vector a.
    
    Parameters:
        b (jnp.ndarray): shape (3,), the initial unit vector.
        a (jnp.ndarray, optional): shape (3,), the target unit vector.
                                   Defaults to [0, 0, 1].
        alpha (float, optional): rotation angle in radians.
                                 If None, alpha is computed as arccos(dot(a, b)).
    
    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
    # Normalize the input vector.
    b = b / jnp.linalg.norm(b)
    d = b.shape[0]

    # If alpha is not provided, compute it as the angle between a and b.
    if alpha is None:
        alpha = jnp.arccos(jnp.dot(a, b))
        
    dot_ab = jnp.dot(a, b)
    tol = 1e-20

    # Define the branch for the nearly identical case.
    def case_ident(_):
        return jnp.eye(d)
    
    # Define the branch for the opposite vectors case.
    def case_negident(_):
        return -jnp.eye(d)
    
    # Define the general case.
    def case_general(_):
        c = b - a * dot_ab
        c = c / jnp.linalg.norm(c)
        A = jnp.outer(a, c) - jnp.outer(c, a)
        return jnp.eye(d) + jnp.sin(alpha) * A + (jnp.cos(alpha) - 1) * (jnp.outer(a, a) + jnp.outer(c, c))
    
    # Use lax.cond to choose the appropriate branch.
    rot = lax.cond(jnp.abs(dot_ab - 1) < tol,
                   lambda _: jnp.eye(d),
                   lambda _: lax.cond(jnp.abs(dot_ab + 1) < tol,
                                      lambda _: -jnp.eye(d),
                                      case_general,
                                      operand=None),
                   operand=None)
    return rot

@jit    
def parallel_transport(d1_1, t1, t2):
    """
    Perform parallel transport of vector d1_1 along tangents t1 and t2.
    
    Parameters:
    d1_1: jnp.ndarray, shape (3,)
        Input vector to be transported.
    t1: jnp.ndarray, shape (3,)
        Initial tangent vector.
    t2: jnp.ndarray, shape (3,)
        Final tangent vector.
        
    Returns:
    jnp.ndarray, shape (3,)
        Resultant vector after parallel transport.
    """
    b = jnp.cross(t1, t2)
    b_norm = jnp.linalg.norm(b)

    def true_fun(_):
        return d1_1
    
    def false_fun(_):
        b_unit = b / b_norm
        # … (rest of the computation as before) …
        b_unit = b_unit - jnp.dot(b_unit, t1) * t1
        b_unit = b_unit / jnp.linalg.norm(b_unit)
        b_unit = b_unit - jnp.dot(b_unit, t1) * t2
        b_unit = b_unit / jnp.linalg.norm(b_unit)

        n1 = jnp.cross(t1, b_unit)
        n2 = jnp.cross(t2, b_unit)
        d1_2 = (jnp.dot(d1_1, t1) * t2 +
                jnp.dot(d1_1, n1) * n2 +
                jnp.dot(d1_1, b_unit) * b_unit)
        d1_2 = d1_2 - jnp.dot(d1_2, t2) * t2
        return d1_2 / jnp.linalg.norm(d1_2)
    
    return lax.cond(b_norm == 0, true_fun, false_fun, operand=None)



def get_length_with_edges(positions,edges):
    return jnp.linalg.norm(positions[edges[0]] -  positions[edges[1]])


# @jit
def rotate(u_i, kappa):
    """
    Rotate the unit vector u_i by the curvature vector kappa via the modified Cayley transform.
    
    With the definition of discrete curvature:
        kappa = 2 * (u_i x u_j) / (1 + u_i·u_j)
    the rotation is given by:
    
        u_j = ((4 - ||kappa||^2)*u_i + 4*(kappa x u_i) + 2*kappa*(kappa·u_i)) / (4 + ||kappa||^2)
    
    Parameters:
        u_i (jnp.ndarray): 3D unit vector.
        kappa (jnp.ndarray): Curvature vector.
        
    Returns:
        jnp.ndarray: The rotated 3D unit vector.
    """
    kappa_norm_sq = jnp.dot(kappa, kappa)
    numerator = (4 - kappa_norm_sq) * u_i + 4 * jnp.cross(kappa, u_i) + 2 * kappa * jnp.dot(kappa, u_i)
    return numerator / (4 + kappa_norm_sq)

# @jit
def reconstruct_curve_from_curvature_and_length(first_point, first_edge, curvatures, reference_lengths):
    
    """
    Reconstruct the curve from the initial point, first edge direction,
    curvature vectors, and segment lengths.
    
    Parameters:
        first_point (jnp.ndarray): The starting point of the curve.
        first_edge (jnp.ndarray): The initial edge vector.
        curvatures (jnp.ndarray): Array of curvature vectors (length = num_segments - 1).
        reference_lengths (jnp.ndarray): Array of segment lengths.
        
    Returns:
        (jnp.ndarray, jnp.ndarray): Reconstructed node positions and tangents.
    """
    num_segments = len(curvatures) + 1  # number of segments
    nodes = jnp.zeros((num_segments + 1, 3))
    nodes = nodes.at[0].set(first_point)
    tangents = jnp.zeros((num_segments, 3))
    tangents = tangents.at[0].set(first_edge / jnp.linalg.norm(first_edge))
    
    # Compute subsequent tangents via the rotation formula.
    def tangent_body(i, tangents):
        next_tangent = rotate(tangents[i], curvatures[i])
        tangents = tangents.at[i+1].set(next_tangent)
        return tangents

    tangents = lax.fori_loop(0, num_segments - 1, tangent_body, tangents)
    
    # Integrate the tangents (scaled by segment lengths) to compute node positions.
    def node_body(i, nodes):
        return nodes.at[i+1].set(nodes[i] + reference_lengths[i] * tangents[i])
    
    nodes = lax.fori_loop(0, num_segments, node_body, nodes)
    
    return nodes, tangents

@jit
def setup_frame(x,initial_setup):
    # if x is a tuple
    # if isinstance(x, tuple):
    #     positions = x[0]
    #     twists = x[1]
    # # else if x is a jnp array
    # elif isinstance(x, jnp.ndarray):
    #     N = x.shape[0]
    #     num_nodes = (N-3)//4+1
    #     positions = x[:num_nodes*3].reshape(-1,3)
    #     twists = x[num_nodes*3:]
    # else:
    #     raise ValueError("Invalid input type. Expected tuple or jnp array.")

    
    # don't worry; it must be a tuple
    positions = x[0]
    twists = x[1]
    ref_twists_old = initial_setup['ref_twist'] + twists

    bishop_frames = space_parallel(positions,jnp.array([0.0,0.0,-1.0]))
    natural_curvatures = get_curvatures_for_a_curve(bishop_frames)
    natural_lengths = get_lengths_for_a_curve(positions)
    new_twists = get_twists(bishop_frames,ref_twists_old)
    material_frames = get_material_frame(bishop_frames,new_twists)

    m1_averaged = (material_frames[:-1, 1] + material_frames[1:, 1])/2
    m2_averaged = (material_frames[:-1, 0] + material_frames[1:, 0])/2

    kappa1_bar =  jnp.sum(natural_curvatures * m1_averaged, axis=1)
    kappa2_bar = -jnp.sum(natural_curvatures * m2_averaged, axis=1) # ok so far

    dict = {
        "bishop_frames": bishop_frames,
        "material_frames": material_frames,
        "natural_curvatures": natural_curvatures,
        "natural_lengths": natural_lengths,
        "kappa1_bar": kappa1_bar,
        "kappa2_bar": kappa2_bar,
        "ref_twist": new_twists
    }

    # return bishop_frames,material_frames,natural_curvatures,natural_lengths,kappa1_bar,kappa2_bar
    return dict

def benchmark():
    pth = '/Users/yeonsu/GitHub/elastic-graph/node_edge_data/trefoil_graph_data.npz'
    data = jnp.load(pth)

    # these two are my degrees of freedom
    positions = jnp.array(data['positions']) # 3*N
    twists = jnp.zeros(positions.shape[0]-1) # N - 1

    import numpy as np
    from matplotlib import pyplot as plt
    np.savetxt('/Users/yeonsu/GitHub/elastic-graph/node_edge_data/positions.txt', positions)
    
    # let's use tuple
    # positions and then twists
    # positions and twists interleaved
    # or tuple

    x = jnp.concatenate([positions.flatten(), twists])
    # fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # ax.plot(*positions.T)
    # plt.show()

    N = positions.shape[0] # number of nodes

    q0 = (positions,twists)
    initial_setup = {
        "bishop_frames": jnp.zeros((N-1,3)),
        "material_frames": jnp.zeros((N-2,3)),
        "natural_curvatures": jnp.zeros((N-2,3)),
        "natural_lengths": jnp.zeros((N-1,)),
        "kappa1_bar": jnp.zeros((N-2,)),
        "kappa2_bar": jnp.zeros((N-2,)),
        "ref_twist": jnp.zeros((N-1,))
    }

    setup = setup_frame(q0,initial_setup)

    # benchmark
    import time
    start = time.time()
    for i in range(1000):
        setup = setup_frame(q0,initial_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (setup frame): {(end - start)/1000} seconds")


    start = time.time()
    bishop_frames = space_parallel(positions,jnp.array([0.0,0.0,-1.0]))
    natural_curvatures = get_curvatures_for_a_curve(bishop_frames)
    natural_lengths = get_lengths_for_a_curve(positions)
    new_twists = get_twists(bishop_frames,twists)
    material_frames = get_material_frame(bishop_frames,new_twists)

    print("Time taken for 1000 iterations: ", time.time() - start)
    

    start = time.time()

    for i in range(1000):
        bishop_frames = space_parallel(positions,jnp.array([0.0,0.0,-1.0]))
        natural_curvatures = get_curvatures_for_a_curve(bishop_frames)
        natural_lengths = get_lengths_for_a_curve(positions)
        new_twists = get_twists(bishop_frames,twists)
        material_frames = get_material_frame(bishop_frames,new_twists)

    print("Time taken for 1000 iterations: ", time.time() - start)
    print("Time per iteration (space parallel): ", (time.time() - start)/1000)



def trefoil_knotting():
    # dof
    # [q_0, theta_0, q_1, theta_1, ...]

    pth = '/Users/yeonsu/GitHub/elastic-graph/node_edge_data/trefoil_graph_data.npz'
    data = jnp.load(pth)

    # these two are my degrees of freedom
    positions = jnp.array(data['positions']) # 3*N
    twists = jnp.zeros(positions.shape[0]-1) # N - 1

    import numpy as np
    from matplotlib import pyplot as plt
    np.savetxt('/Users/yeonsu/GitHub/elastic-graph/node_edge_data/positions.txt', positions)
    # let's use tuple

    # positions and then twists
    # positions and twists interleaved
    # or tuple

    x = jnp.concatenate([positions.flatten(), twists])
    
    # fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # ax.plot(*positions.T)
    # plt.show()

    N = positions.shape[0] # number of nodes

    q0 = (positions,twists)
    initial_setup = {
        "bishop_frames": jnp.zeros((N-1,3)),
        "material_frames": jnp.zeros((N-2,3)),
        "natural_curvatures": jnp.zeros((N-2,3)),
        "natural_lengths": jnp.zeros((N-1,)),
        "kappa1_bar": jnp.zeros((N-2,)),
        "kappa2_bar": jnp.zeros((N-2,)),
        "ref_twist": jnp.zeros((N-1,))
    }

    setup = setup_frame(q0,initial_setup)

    # benchmark
    import time
    start = time.time()
    for i in range(1000):
        setup = setup_frame(q0,initial_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (setup frame): {(end - start)/1000} seconds")

    previous_setup = setup.copy()
    
    bishop_frame = setup['bishop_frames']
    initial_lengths = setup['natural_lengths']
    initial_k1 = setup['kappa1_bar']
    initial_k2 = setup['kappa2_bar']



    # from visualizations import plot_with_bishop_frames
    
    # fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # plot_with_bishop_frames(positions,bishop_frame,ax=ax)
    # plt.show()

    # flatten the curve: same edge lengths
    natural_lengths = setup['natural_lengths']
    initial_point = positions[0]
    initial_edge_direction = positions[1] - positions[0]
    initial_edge_direction = initial_edge_direction / jnp.linalg.norm(initial_edge_direction)

    length_cumsum = jnp.concatenate([jnp.array([0]),jnp.cumsum(natural_lengths)])
    # add points from initial point toward the initial edge direction
    flattened_positions = jnp.zeros((N,3))
    flattened_positions = flattened_positions.at[0].set(initial_point)
    for i in range(1,N):
        flattened_positions = flattened_positions.at[i].set(initial_point + initial_edge_direction * length_cumsum[i])

    


    # add jitter
    from jax import random
    key = random.PRNGKey(0)
    jitter = random.normal(key, (N,3), dtype=jnp.float32) * 0.01
    flattened_positions = flattened_positions + jitter    

    # check sanity
    check_sanity = False
    if check_sanity:
        fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
        ax.plot(*flattened_positions.T)
        ax.axis('equal')
        ax.set_box_aspect([1,1,1])  # aspect ratio is 1:1:1
        plt.show()
        positions.shape
        flattened_positions.shape
        _q0 = (flattened_positions,twists)
        _setup = setup_frame(_q0,initial_setup)
        _nl = _setup['natural_lengths']
        plt.plot(natural_lengths,_nl)
        plt.show()
        np.isclose(natural_lengths,_nl)

        tmp = setup['natural_lengths']

    @jit
    def elastic_energy(q, previous_setup):
        positions = q[0]
        twists = q[1]

        new_setup = setup_frame((positions,twists),previous_setup)
        k1 = new_setup['kappa1_bar']
        k2 = new_setup['kappa2_bar']

        natural_lengths = new_setup['natural_lengths']
        voronoi_lengths = (natural_lengths[1:] + natural_lengths[:-1]) / 2

        ref_twist = new_setup['ref_twist']
        stretching_energy = 10000*jnp.sum( (natural_lengths-initial_lengths)**2 )
        bending_energy = 10*jnp.sum( (jnp.abs(k1-initial_k1)**2 + jnp.abs(k2-initial_k2)**2)/voronoi_lengths )
        twisting_energy = jnp.sum(((twists[1:] - twists[:-1] - ref_twist[1:])**2)/voronoi_lengths)

        return stretching_energy + bending_energy + twisting_energy
        # return bending_energy + twisting_energy
    
    e = elastic_energy(q0,previous_setup)


    print(e)

    # benchmark
    start = time.time()
    for i in range(1000):
        e = elastic_energy(q0,previous_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (elastic energy): {(end - start)/1000} seconds")
    
    # grad_e = jit(grad(lambda q: elastic_energy(q, previous_setup)))
    grad_e = jit(grad(elastic_energy,argnums=0))

    f = grad_e(q0,previous_setup)
    # benchmark
    start = time.time()
    for i in range(1000):
        f = grad_e(q0,previous_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (grad): {(end - start)/1000} seconds")
        
    import optax
    # num_steps = 100000
    num_steps = 400000
    learning_rate = 1.e-2
    intercept = 10000
    optimizer = optax.adam(learning_rate)
    
    q_init = (flattened_positions,twists)
    opt_state = optimizer.init(q_init)

    qq = []
    qq.append(q_init)
    q = q_init

    previous_setup = setup_frame(q,initial_setup)
    

    print(elastic_energy(q, previous_setup))

    @jit
    def update(q, previous_setup, opt_state):
        x = q[0]
        t = q[1]
        dqds = grad_e(q, previous_setup)
        updates, opt_state = optimizer.update(dqds, opt_state)
        new_q = optax.apply_updates(q, updates)
        # new_q = q - 1e-5*dqds
        # new_q = (x - 1e-5 * dqds[0], t - 1e-5 * dqds[1])

        previous_setup = setup_frame(new_q,previous_setup)

        return new_q, previous_setup, opt_state
    

    @jit
    def project_positions_to_fixed_lengths(positions, initial_lengths):
        # Loop over segments and rescale to target length
        def body_fn(i, pos):
            p0 = pos[i]
            p1 = pos[i+1]
            dir = p1 - p0
            dir = dir / jnp.linalg.norm(dir)
            new_p1 = p0 + initial_lengths[i] * dir
            return pos.at[i+1].set(new_p1)

        # Fix first point, project all others
        positions = lax.fori_loop(0, positions.shape[0] - 1, body_fn, positions)
        return positions
    
    # @jit
    # def project_curve_scan(first_point, first_direction, lengths):
    #     """
    #     Project positions along the fixed segment directions to match reference lengths.
    #     """
    #     def body_fn(carry, length):
    #         prev_pos, direction = carry
    #         new_pos = prev_pos + length * direction
    #         return (new_pos, direction), new_pos

    #     # Initial tangent should be normalized
    #     direction = first_direction / jnp.linalg.norm(first_direction)
    #     _, positions = lax.scan(body_fn, (first_point, direction), lengths)
    #     # Prepend the first point
    #     positions = jnp.vstack([first_point[None, :], positions])
    #     return positions
    
    import pickle
    qq = []
    k = 0
    for step in range(num_steps):

        q, previous_setup, opt_state  = update(q, previous_setup, opt_state)
        

        # q = (project_positions_to_fixed_lengths(q[0], initial_lengths), q[1])        
        # dqds = grad_e(q, previous_setup)
        # new_q = (q[0] - learning_rate * dqds[0], q[1] - learning_rate * dqds[1])
        # previous_setup = setup_frame(new_q,previous_setup)
        # q = new_q

        if step % intercept == 0:
            qq.append(q)
            print(f'Step {step}, Energy: {elastic_energy(q, previous_setup)}')
            # print(f'Energy: {elastic_energy(curr_positions, prev_bishop_frames, previous_twists, kappa1_bar, kappa2_bar, natural_lengths, natural_curvatures)}')
            # print(f'Step {step}, Energy: {elastic_energy(curr_positions, prev_bishop_frames, previous_twists, kappa1_bar, kappa2_bar, natural_lengths, natural_curvatures)}, Grad Norm: {jnp.linalg.norm(grad(elastic_energy)(curr_positions, prev_bishop_frames, previous_twists, kappa1_bar, kappa2_bar, natural_lengths, natural_curvatures))}')
        if step % 1000 == 0:
            with open('qq.pkl', 'wb') as f:
                pickle.dump(qq, f)

            plt.figure()
            plt.subplot(111, projection='3d')
            positions = q[0]
            plt.plot(*positions.T)
            plt.title(f'Step {step}, Energy: {elastic_energy(q, previous_setup)}')
            plt.savefig(f'trefoil_knotting/frame_{k:05d}', bbox_inches='tight', dpi=300)
            plt.close('all')
            k += 1

    


    initial_positions = q_init[0]
    final_positions = q[0]
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # ax.plot(*initial_positions.T)
    ax.plot(*final_positions.T)
    # plot_with_bishop_frames(final_positions,setup['bishop_frames'],ax=ax)
    plt.show()

    # save q
    with open('qq.pkl', 'wb') as f:
        pickle.dump(qq, f)
    

    # forget this..
    # from flax import struct
    # @struct.dataclass
    # class State:
    #     position: jnp.ndarray
    #     velocity: jnp.ndarray

    # initialization


def benchmark_energy_impl():

    # dof
    # [q_0, theta_0, q_1, theta_1, ...]

    pth = '/Users/yeonsu/GitHub/elastic-graph/node_edge_data/trefoil_graph_data.npz'
    data = jnp.load(pth)

    # these two are my degrees of freedom
    positions = jnp.array(data['positions']) # 3*N
    twists = jnp.zeros(positions.shape[0]-1) # N - 1

    import numpy as np
    from matplotlib import pyplot as plt
    np.savetxt('/Users/yeonsu/GitHub/elastic-graph/node_edge_data/positions.txt', positions)
    # let's use tuple

    x = jnp.concatenate([positions.flatten(), twists])
    N = positions.shape[0] # number of nodes

    q0 = (positions,twists)
    initial_setup = {
        "bishop_frames": jnp.zeros((N-1,3)),
        "material_frames": jnp.zeros((N-2,3)),
        "natural_curvatures": jnp.zeros((N-2,3)),
        "natural_lengths": jnp.zeros((N-1,)),
        "kappa1_bar": jnp.zeros((N-2,)),
        "kappa2_bar": jnp.zeros((N-2,)),
        "ref_twist": jnp.zeros((N-1,))
    }

    setup = setup_frame(q0,initial_setup)

    # benchmark
    import time
    start = time.time()
    for i in range(1000):
        setup = setup_frame(q0,initial_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (setup frame): {(end - start)/1000} seconds")

    previous_setup = setup.copy()
    
    bishop_frame = setup['bishop_frames']
    initial_lengths = setup['natural_lengths']
    initial_k1 = setup['kappa1_bar']
    initial_k2 = setup['kappa2_bar']

    from visualizations import plot_with_bishop_frames

    # flatten the curve: same edge lengths
    natural_lengths = setup['natural_lengths']
    initial_point = positions[0]
    initial_edge_direction = positions[1] - positions[0]
    initial_edge_direction = initial_edge_direction / jnp.linalg.norm(initial_edge_direction)

    length_cumsum = jnp.concatenate([jnp.array([0]),jnp.cumsum(natural_lengths)])
    # add points from initial point toward the initial edge direction
    flattened_positions = jnp.zeros((N,3))
    flattened_positions = flattened_positions.at[0].set(initial_point)
    for i in range(1,N):
        flattened_positions = flattened_positions.at[i].set(initial_point + initial_edge_direction * length_cumsum[i])

    # add jitter
    from jax import random
    key = random.PRNGKey(0)
    jitter = random.normal(key, (N,3), dtype=jnp.float32) * 0.01
    flattened_positions = flattened_positions + jitter    

    @jit
    def elastic_energy(q, previous_setup):
        positions = q[0]
        twists = q[1]

        new_setup = setup_frame((positions,twists),previous_setup)
        k1 = new_setup['kappa1_bar']
        k2 = new_setup['kappa2_bar']

        natural_lengths = new_setup['natural_lengths']
        voronoi_lengths = (natural_lengths[1:] + natural_lengths[:-1]) / 2

        ref_twist = new_setup['ref_twist']
        stretching_energy = 10000*jnp.sum( (natural_lengths-initial_lengths)**2 )
        bending_energy = 10*jnp.sum( (jnp.abs(k1-initial_k1)**2 + jnp.abs(k2-initial_k2)**2)/voronoi_lengths )
        twisting_energy = jnp.sum(((twists[1:] - twists[:-1] - ref_twist[1:])**2)/voronoi_lengths)

        return stretching_energy + bending_energy + twisting_energy
        # return bending_energy + twisting_energy

    @jit
    def elastic_energy_all_in_one(q,ref_twists_old):
        positions = q[0]
        twists = q[1]

        tangents = positions[1:] - positions[:-1]
        tangents = tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)
        
        ne = tangents.shape[0]
        d1 = jnp.zeros_like(tangents)
        d2 = jnp.zeros_like(tangents)
        
        # Initialize the first frame using cross products.
        m1 = jnp.cross(tangents[0], jnp.array([0.0, 0.0, -1.0]))
        m2 = jnp.cross(tangents[0], m1)
        d1 = d1.at[0].set(m1)
        d2 = d2.at[0].set(m2)
        
        def body(i, val):
            d1, d2 = val
            # Previous frame and tangents for the transport step.
            a = d1[i]
            b = tangents[i]
            c = tangents[i + 1]
            d = parallel_transport(a, b, c)
            # Update the frames at index i+1.
            d1 = d1.at[i + 1].set(d)
            d2 = d2.at[i + 1].set(jnp.cross(c, d))
            return (d1, d2)
        
        d1, d2 = lax.fori_loop(0, ne - 1, body, (d1, d2))
        bishop_frames = jnp.stack([tangents, d1, d2], axis=1)

        t1 = bishop_frames[:-1, 0, :]  # shape (N-1, 3)
        t2 = bishop_frames[1:, 0, :]   # shape (N-1, 3)
        
        cross_vals = jnp.cross(t1, t2)        # shape (N-1, 3)
        dot_vals = jnp.sum(t1 * t2, axis=1)     # shape (N-1,)
        curvature = 2 * cross_vals / (1.0 + dot_vals)[:, None]  # shape (N-1, 3)

        natural_lengths = jnp.sqrt( jnp.sum(tangents**2,axis=1) )

        ref_twists = jnp.zeros( bishop_frames.shape[0] )
        t21 = bishop_frames[:-1, 0, :] # t0
        t22 = bishop_frames[1:, 0, :]  # t1

        u21 = bishop_frames[:-1, 1, :] # u0
        u22 = bishop_frames[1:, 1, :]  # u1

        ut = batch_parallel_transport(u21,t21,t22)
        ut = batch_rotate_vector(ut, t22, ref_twists_old[1:])

        ref_twists = ref_twists_old[1:] + batch_signed_angle(ut,u22,t22)

        # pad zero at the first entry
        ref_twists = jnp.concatenate([jnp.array([0.0]), ref_twists], axis=0)


        ref_twists = ref_twists[:, None]
        m1 =  jnp.cos(ref_twists) * bishop_frames[:,1, :] + jnp.sin(ref_twists) * bishop_frames[:,2, :]
        m2 = -jnp.sin(ref_twists) * bishop_frames[:,1, :] + jnp.cos(ref_twists) * bishop_frames[:,2, :]
        material_frames = jnp.stack((m1,m2),axis=1)

        m1_averaged = (material_frames[:-1, 1] + material_frames[1:, 1])/2
        m2_averaged = (material_frames[:-1, 0] + material_frames[1:, 0])/2

        kappa1 =  jnp.sum(curvature * m1_averaged, axis=1)
        kappa2 = -jnp.sum(curvature * m2_averaged, axis=1) # ok so far

        bending_energy = 10*jnp.sum( (jnp.abs(kappa1-initial_k1)**2 + jnp.abs(kappa2-initial_k2)**2)/natural_lengths[1:] )
        twisting_energy = jnp.sum(((twists[1:] - twists[:-1] - ref_twists[1:])**2)/natural_lengths[1:])
        stretching_energy = 10000*jnp.sum( (natural_lengths-initial_lengths)**2 )
        return stretching_energy + bending_energy + twisting_energy


    
    e = elastic_energy(q0,previous_setup)
    e2 = elastic_energy_all_in_one(q0,setup['ref_twist'])

    

    # benchmark
    start = time.time()
    for i in range(1000):
        e = elastic_energy(q0,previous_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (elastic energy): {(end - start)/1000} seconds")

    start = time.time()
    for i in range(1000):
        e2 = elastic_energy_all_in_one(q0,setup['ref_twist'])
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (elastic energy2): {(end - start)/1000} seconds")
    
    # grad_e = jit(grad(lambda q: elastic_energy(q, previous_setup)))
    grad_e = jit(grad(elastic_energy,argnums=0))
    grad_e2 = jit(grad(elastic_energy_all_in_one,argnums=0))

    f = grad_e(q0,previous_setup)
    # benchmark
    start = time.time()
    for i in range(1000):
        f = grad_e(q0,previous_setup)
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (grad): {(end - start)/1000} seconds")

    start = time.time()
    for i in range(1000):
        f2 = grad_e2(q0,setup['ref_twist'])
    end = time.time()
    print(f"Time taken for 1000 iterations: {end - start} seconds")
    print(f"Time per iteration (grad ): {(end - start)/1000} seconds")
        
    import optax
    num_steps = 1000000
    learning_rate = 1.e-2
    intercept = 1000
    optimizer = optax.adam(learning_rate)
    
    q_init = (flattened_positions,twists)
    opt_state = optimizer.init(q_init)

    qq = []
    qq.append(q_init)
    q = q_init

    @jit
    def update(q, previous_setup, opt_state):
        x = q[0]
        t = q[1]
        dqds = grad_e(q, previous_setup)
        # updates, opt_state = optimizer.update(dqds, opt_state)
        # new_q = optax.apply_updates(q, updates)
        # new_q = q - 1e-5*dqds
        new_q = (x - 1e-5 * dqds[0], t - 1e-5 * dqds[1])

        previous_setup = setup_frame(new_q,previous_setup)

        return new_q, previous_setup, opt_state
    

    @jit
    def project_positions_to_fixed_lengths(positions, initial_lengths):
        # Loop over segments and rescale to target length
        def body_fn(i, pos):
            p0 = pos[i]
            p1 = pos[i+1]
            dir = p1 - p0
            dir = dir / jnp.linalg.norm(dir)
            new_p1 = p0 + initial_lengths[i] * dir
            return pos.at[i+1].set(new_p1)

        # Fix first point, project all others
        positions = lax.fori_loop(0, positions.shape[0] - 1, body_fn, positions)
        return positions
    
    # @jit
    # def project_curve_scan(first_point, first_direction, lengths):
    #     """
    #     Project positions along the fixed segment directions to match reference lengths.
    #     """
    #     def body_fn(carry, length):
    #         prev_pos, direction = carry
    #         new_pos = prev_pos + length * direction
    #         return (new_pos, direction), new_pos

    #     # Initial tangent should be normalized
    #     direction = first_direction / jnp.linalg.norm(first_direction)
    #     _, positions = lax.scan(body_fn, (first_point, direction), lengths)
    #     # Prepend the first point
    #     positions = jnp.vstack([first_point[None, :], positions])
    #     return positions
    
    import pickle
    qq = []
    for step in range(num_steps):

        q, previous_setup, opt_state  = update(q, previous_setup, opt_state)
        

        # q = (project_positions_to_fixed_lengths(q[0], initial_lengths), q[1])        
        # dqds = grad_e(q, previous_setup)
        # new_q = (q[0] - learning_rate * dqds[0], q[1] - learning_rate * dqds[1])
        # previous_setup = setup_frame(new_q,previous_setup)
        # q = new_q

        if step % intercept == 0:
            qq.append(q)
            print(f'Step {step}, Energy: {elastic_energy(q, previous_setup)}')
            # print(f'Energy: {elastic_energy(curr_positions, prev_bishop_frames, previous_twists, kappa1_bar, kappa2_bar, natural_lengths, natural_curvatures)}')
            # print(f'Step {step}, Energy: {elastic_energy(curr_positions, prev_bishop_frames, previous_twists, kappa1_bar, kappa2_bar, natural_lengths, natural_curvatures)}, Grad Norm: {jnp.linalg.norm(grad(elastic_energy)(curr_positions, prev_bishop_frames, previous_twists, kappa1_bar, kappa2_bar, natural_lengths, natural_curvatures))}')
        if step * 10000 == 0:
            with open('qq.pkl', 'wb') as f:
                pickle.dump(qq, f)

    


    initial_positions = q_init[0]
    final_positions = q[0]
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # ax.plot(*initial_positions.T)
    ax.plot(*final_positions.T)
    # plot_with_bishop_frames(final_positions,setup['bishop_frames'],ax=ax)
    plt.show()

    # save q
    with open('qq.pkl', 'wb') as f:
        pickle.dump(qq, f)

def top_level_knotting():
    import jax
    print(jax.devices())

    batch_parallel_transport = jit(vmap(parallel_transport, in_axes=(0, 0, 0)))
    batch_rotate_vector = jit(vmap(rotate_vector, in_axes=(0, 0, 0)))
    batch_signed_angle = jit(vmap(signed_angle, in_axes=(0, 0, 0)))

    # benchmark()
    # benchmark_energy_impl()
    trefoil_knotting()

    # simple control problem: reaching a target point, with low curvature

    # starting from (almost flat curve)

def test_recon_from_kappa():

    import numpy as np
    from matplotlib import pyplot as plt
    
    pos = np.loadtxt('/Users/yeonsu/GitHub/elastic-graph/node_edge_data/helix_positions_data.txt', delimiter=',')

    twists = np.zeros(pos.shape[0]-1) # N - 1
    N = pos.shape[0] # number of nodes

    q0 = (pos,twists)
    initial_setup = {
        "bishop_frames": jnp.zeros((N-1,3)),
        "material_frames": jnp.zeros((N-2,3)),
        "natural_curvatures": jnp.zeros((N-2,3)),
        "natural_lengths": jnp.zeros((N-1,)),
        "kappa1_bar": jnp.zeros((N-2,)),
        "kappa2_bar": jnp.zeros((N-2,)),
        "ref_twist": jnp.zeros((N-1,))
    }

    
    batch_parallel_transport = jit(vmap(parallel_transport, in_axes=(0, 0, 0)))
    batch_rotate_vector = jit(vmap(rotate_vector, in_axes=(0, 0, 0)))
    batch_signed_angle = jit(vmap(signed_angle, in_axes=(0, 0, 0)))

    setup = setup_frame(q0,initial_setup)

    # kappa = setup['natural_curvatures']
    tan = pos[1:] - pos[:-1]
    tan = tan / np.linalg.norm(tan,axis=1, keepdims=True)  # normalize tangents

    num = 2*np.cross(tan[:-1],tan[1:],axis=1)
    den = 1 + np.sum(tan[1:] * tan[:-1], axis=1)

    # dot_vals = jnp.sum(t1 * t2, axis=1)

    
    

    kappa = num/den[:, None]  # curvature vector
    nat_len = get_lengths_for_a_curve(pos)

    kappa = jnp.array(kappa)
    pos = jnp.array(pos)
    nat_len = jnp.array(nat_len)



    pos_rec,_ = reconstruct_curve_from_curvature_and_length(pos[0],pos[1]-pos[0],kappa,nat_len)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(*pos.T)
    ax.plot(*pos_rec.T, '.', color='red', alpha=0.3)
    plt.axis('equal')
    

    plt.show()

    

if __name__ == "__main__":
    
    # testing bending force - edges
    batch_parallel_transport = jit(vmap(parallel_transport, in_axes=(0, 0, 0)))
    batch_rotate_vector = jit(vmap(rotate_vector, in_axes=(0, 0, 0)))
    batch_signed_angle = jit(vmap(signed_angle, in_axes=(0, 0, 0)))
    trefoil_knotting()

