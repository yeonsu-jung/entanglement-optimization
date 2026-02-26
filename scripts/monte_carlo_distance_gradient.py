
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

def fixbound(x):
    return jnp.clip(x, 0.0, 1.0)

def lumelsky_dist_jax_full(point1s, point1e, point2s, point2e):
    """ 
    Calculate the shortest distance between two line segments using JAX. 
    Returns distance, t, u (parameters on line 1 and line 2).
    """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    # We add a small epsilon to denominator to avoid division by zero in parallel case
    # Ideally checking den < eps
    
    t = (S1 * D2 - S2 * R) / (den + 1e-12)
    t = fixbound(t)
    
    u = (t * R - S2) / (D2 + 1e-12)
    uf = fixbound(u)
    
    # If u was clamped, we might need to adjust t
    t_new = fixbound((uf * R + S1) / (D1 + 1e-12))
    t = jnp.where(uf != u, t_new, t)
    u = uf

    # Compute distance vector
    dist_vec = d1 * t - d2 * u - d12
    return jnp.linalg.norm(dist_vec), t, u

def lumelsky_dist_jax(point1s, point1e, point2s, point2e):
    # Wrapper that only returns distance for differentiation
    d, _, _ = lumelsky_dist_jax_full(point1s, point1e, point2s, point2e)
    return d

def sph2cart(theta, phi):
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)

def q_to_endpoints(q, l=1.0):
    # q = (x, y, z, theta, phi)
    # x,y,z is centroid
    centroid = q[:3]
    theta = q[3]
    phi = q[4]
    
    direction = sph2cart(theta, phi)
    
    p_start = centroid - 0.5 * l * direction
    p_end = centroid + 0.5 * l * direction
    return p_start, p_end, direction

def run_experiment(num_trials=100):
    key = jax.random.PRNGKey(42)
    
    success_count = 0
    
    print(f"Running {num_trials} trials...")
    print(f"{'Trial':<6} {'Dist':<8} {'t':<6} {'u':<6} {'Grad.Proj':<10} {'d/dTheta':<10} {'d/dPhi':<10} {'Result':<10}")
    print("-" * 70)

    for i in range(num_trials):
        key, subkey = jax.random.split(key)
        
        # Generate random rod 1
        pos1 = jax.random.uniform(subkey, (3,), minval=-2.0, maxval=2.0)
        theta1 = jax.random.uniform(subkey, (), minval=0.1, maxval=np.pi-0.1)
        phi1 = jax.random.uniform(subkey, (), minval=0, maxval=2*np.pi)
        
        key, subkey = jax.random.split(key)
        # Generate random rod 2
        pos2 = jax.random.uniform(subkey, (3,), minval=-2.0, maxval=2.0)
        theta2 = jax.random.uniform(subkey, (), minval=0.1, maxval=np.pi-0.1)
        phi2 = jax.random.uniform(subkey, (), minval=0, maxval=2*np.pi)
        
        q1 = jnp.concatenate([pos1, jnp.array([theta1, phi1])])
        q2 = jnp.concatenate([pos2, jnp.array([theta2, phi2])])
        
        # Check if closest points are internal
        p1s, p1e, dir1 = q_to_endpoints(q1)
        p2s, p2e, dir2 = q_to_endpoints(q2)
        
        dist, t, u = lumelsky_dist_jax_full(p1s, p1e, p2s, p2e)
        
        # We only care about cases where closest points are strictly internal
        epsilon = 1e-3
        if t > epsilon and t < 1.0 - epsilon and u > epsilon and u < 1.0 - epsilon:
            
            # Compute gradient
            def dist_func(q_var):
                ps, pe, _ = q_to_endpoints(q_var)
                # p2s, p2e fixed from closure
                return lumelsky_dist_jax(ps, pe, p2s, p2e)
            
            grad_q = jax.grad(dist_func)(q1)
            grad_pos = grad_q[:3]
            grad_theta = grad_q[3]
            grad_phi = grad_q[4]
            
            # Project gradient onto rod axis
            proj = jnp.dot(grad_pos, dir1)
            
            is_zero = jnp.abs(proj) < 1e-8
            result_str = "PASS" if is_zero else "FAIL"
            
            print(f"{i:<6} {dist:<8.3f} {t:<6.2f} {u:<6.2f} {proj:<10.2e} {grad_theta:<10.2e} {grad_phi:<10.2e} {result_str}")
            
            if not is_zero:
                 print(f"   -> FAIL DETAILS: Grad_pos: {grad_pos}, Dir: {dir1}")
            
            success_count += 1
            if success_count >= 10:
                break
    
    if success_count == 0:
        print("No internal closest points found in random trials.")

def test_perpendicular_case():
    print("\nRunning Critical Example: Perpendicular Rods (Centroids Closest)")
    print("-" * 70)
    
    # Rod 1: Centroid (0,0,0), along X-axis
    # theta = pi/2, phi = 0
    pos1 = jnp.array([0.0, 0.0, 0.0])
    theta1 = np.pi/2
    phi1 = 0.0
    q1 = jnp.concatenate([pos1, jnp.array([theta1, phi1])])
    
    # Rod 2: Centroid (0,0,2), along Y-axis
    # theta = pi/2, phi = pi/2
    pos2 = jnp.array([0.0, 0.0, 2.0])
    theta2 = np.pi/2
    phi2 = np.pi/2
    q2 = jnp.concatenate([pos2, jnp.array([theta2, phi2])])
    
    # Expected distance: 2.0
    # Expected t, u: 0.5 (middle of rods)
    
    p1s, p1e, dir1 = q_to_endpoints(q1)
    p2s, p2e, dir2 = q_to_endpoints(q2)
    
    dist, t, u = lumelsky_dist_jax_full(p1s, p1e, p2s, p2e)
    
    # Compute gradients
    def dist_func(q_var):
        ps, pe, _ = q_to_endpoints(q_var)
        return lumelsky_dist_jax(ps, pe, p2s, p2e)
        
    grad_q = jax.grad(dist_func)(q1)
    grad_pos = grad_q[:3]
    grad_theta = grad_q[3]
    grad_phi = grad_q[4]
    
    proj = jnp.dot(grad_pos, dir1)
    
    print(f"Dist: {dist:.5f} (Expected: 2.00000)")
    print(f"t:    {t:.3f}   (Expected: 0.500)")
    print(f"u:    {u:.3f}   (Expected: 0.500)")
    print(f"Grad Projection on Axis: {proj:.2e}")
    print(f"d/dTheta: {grad_theta:.2e}")
    print(f"d/dPhi:   {grad_phi:.2e}")
    print("-" * 70)

if __name__ == "__main__":
    test_perpendicular_case()
    # run_experiment(1000)
