# %%
from jax import jit, grad, vmap, tree_map
import jax.numpy as jnp
import numpy as onp
from jax import lax

def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    
    return dist


@jit
def pairwise_distance(q_pair):
    x_i =     q_pair[0]
    y_i =     q_pair[1]
    z_i =     q_pair[2]
    phi_i =   q_pair[3]
    theta_i = q_pair[4]
  
    x_j =     q_pair[5]
    y_j =     q_pair[6]
    z_j =     q_pair[7]
    phi_j =   q_pair[8]
    theta_j = q_pair[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    return dist_lin_seg(p_i, p_ii, p_j, p_jj)

@jit
def pairwise_distance_xyz(q_pair):
    p_i = jnp.array([q_pair[0], q_pair[1], q_pair[2]])
    p_j = jnp.array([q_pair[6], q_pair[7], q_pair[8]])
    p_ii = jnp.array([q_pair[3], q_pair[4], q_pair[5]])
    p_jj = jnp.array([q_pair[9], q_pair[10], q_pair[11]])
    return dist_lin_seg(p_i, p_ii, p_j, p_jj)

@jit
def all_pairwise_distances(q_pairs):
    return vmap(pairwise_distance)(q_pairs)

def create_pairs(m):
    N, M = m.shape
    # Get the upper triangular indices excluding the diagonal
    i, j = jnp.triu_indices(N, k=1)
    # Retrieve rows for each index in the pairs
    m_i = m[i]  # Shape will be (N(N-1)/2, M)
    m_j = m[j]  # Shape will be (N(N-1)/2, M)
    # Concatenate the rows from each pair horizontally
    m_pairs = jnp.concatenate([m_i, m_j], axis=1)  # Resulting shape will be (N(N-1)/2, 2M)
    return m_pairs


def optimize_fire_nonjax_individual(q0,f,df,Nmax,atol=1e-4,dt = 0.002,logoutput=False,callback=None):
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1  # example starting value for alpha
    alpha = alpha0
    Ndelay = 10   # example delay for adjusting dt
    finc = 1.1    # factor to increase dt
    fdec = 0.5    # factor to decrease dt
    fa = 0.99     # factor to adjust alpha
    
    alpha0 = 0.1
    Ndelay = 5
    finc = 1.1
    fdec = 0.5
    fa = 0.99
    Nnegmax = 200000
    
    error = 10*atol 
    
    dtmin = 0.02*dt
    alpha = alpha0
    
    Npos = jnp.zeros(q0.shape[0])

    q = q0.copy()
    V = jnp.zeros(q.shape)
    F = -df(q)
    dt_array = jnp.ones(q.shape)*dt
    dtmax = 10*dt_array

    # disgusting hack to save the q values
    from pathlib import Path
    k = 0
    
    break_or_continue = False
    for i in range(Nmax):

        # P = (F*V).sum() # dissipated power
        P = F*V
        V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
        Npos = jnp.where(P>0,Npos+1,0)
        
        dt_choice = jnp.array([dt_array * finc, dtmax])
        
        dt_array = jnp.where(P > 0, jnp.where(Npos > Ndelay,jnp.min(dt_choice),dt_array),dt_array)
        dt_array = jnp.where(P <= 0, dt * fdec, dt)
        
        alpha = jnp.where(P > 0,jnp.where(Npos > Ndelay,
                                alpha * fa,
                                alpha),alpha)
        
        alpha = jnp.where(P <= 0, alpha0, alpha)
        # P = tree_map(lambda p: (F_dot_P >= 0) * p, P)
        
        # V = jnp.where(P >= 0,V + 0.5*dt*F,V)
        V = V + 0.25*dt*F
        q = q + 0.5*dt*V
        F = -df(q)
        V = V + 0.25*dt*F
        
        V = tree_map(lambda v: (P >= 0) * v, V)
        
        V = V + 0.25*dt*F
        q = q + 0.5*dt*V
        F = -df(q)
        V = V + 0.25*dt*F

        error = jnp.max(jnp.abs(F))
        if onp.mod(i,100) == 0:
            q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
            d = all_pairwise_distances(q_pairs)
            print(f"Iteration: {i:3d}, "
                    f"fval: {f(q):12.7f}, "
                    f"error: {error:12.7f}, "
                    f"min. dist.: {jnp.min(d):12.7f}")

            callback_params = {"numbering": k, "min_distance": jnp.min(d)}
            if callback is not None:
                break_or_continue = callback(q,callback_params)

            if break_or_continue:
                print("Callback requested to break the loop")
                break            

            k += 1
        
        # if jnp.isnan(F).any():
        #     print("NaN detected in variables")
            
        # if error < atol: break

        if logoutput: print(f(q),error)

    # del V, F  
    return q, f(q), Npos, error

def collision_relaxation(q,f_in,params,N_outer,Nmax,atol,dt,atol_min=1,visualize=False,callback=None):    

    num_rods = q.shape[0]//5
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    from potentials import dist_lin_seg_over_ij
    from transforms import q_to_x
    
    for k in range(N_outer):
        col_rad_0 = params["col_rad"]
        params["col_rad"] = params["col_rad"]*(1)
        f = lambda q: f_in(q,params)
        df = jit(grad(jit(f)))
        # q, f_val, num_iterations, error = optimize_fire_nonjax(q, f, df, Nmax, atol, dt, False)
        q, f_val, _, error = optimize_fire_nonjax_individual(q, f, df, Nmax, atol, dt, callback=callback)
        # q, f_val, _, error = optimize_fire_jax_individual(q, f, df, Nmax, atol, dt, callback=callback)
        
        # if (error < atol_min):
        #     print(f"Error is smaller than atol_min: {error}")
        #     break
        
        q_pairs = create_pairs(q.reshape(-1,5))
        distances = all_pairwise_distances(q_pairs)
        # dt = dt/1.1     # TO DO: factor out this numbers
        # if jnp.abs(jnp.min(distances) - col_rad_0) < col_rad_0*1e-6:

        # if k % 100 == 0:
        #     x = q_to_x(q)
        #     r1 = x[:,0:3]
        #     r2 = x[:,3:6]
        #     dist_mat = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
        #     print(f"Min distance between rods: {jnp.min(dist_mat)}")

        if (jnp.min(distances) - col_rad_0*2) > 0:
            print(f"Enough pushoff: {jnp.min(distances)}")           
            break
    
    return q


@jit
def simple_harmonic_line_jump(q,params):
    col_rad = params["col_rad"]
    amp = params["amp"]    
    x_i = q[0]
    y_i = q[1]
    z_i = q[2]
    phi_i = q[3]
    theta_i = q[4]

    x_j = q[5]
    y_j = q[6]
    z_j = q[7]
    phi_j = q[8]
    theta_j = q[9]

    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    dist = dist_lin_seg(p_i, p_ii, p_j, p_jj)
    
    dist_cont = lax.cond(dist < (col_rad*2),
                         lambda _: amp*(dist-col_rad*2)**2,
                         lambda _: 0., # decrease to get more contacts
                         
                         None)
    return dist_cont

@jit
def total_harmonic_line(q,params):
    q = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q)
    
    def body_fun(carry, q_pair):
        # Increment carry by the result of effective_potential applied to q_pair
        return carry + simple_harmonic_line_jump(q_pair,params), None
    # Perform scan; initial carry value is 0
    total, _ = lax.scan(body_fun, 0, q_pairs)
    
    return total


# %%
if __name__ == "__main__":
    
    # pick up random number from spherical coordinates
    from protocols import create_intersecting_rods
    from visualizations import plot_many_rods
    num_rods = 100
    q = create_intersecting_rods(num_rods)

    ax = plot_many_rods(q.reshape(-1,5))
    ax.axis('equal')


    # %%
    # nudging
    q = q.reshape(-1,5)
    centers = q[:,0:3]

    # give a gaussian noise to the centers
    import jax
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key,shape=centers.shape)
    noise = noise / jnp.linalg.norm(noise,axis=1,keepdims=True)
    noise = 1.e-5 * noise

    new_centers = centers + noise
    q = q.at[:,0:3].set(new_centers)

    # %%
    plot_many_rods(q.reshape(-1,5))
    # %%

    f = total_harmonic_line
    params = {}
    params["col_rad"] = 1.e-3
    params["amp"] = 1
    dt = 1.e-2
    N_outer = 1
    Nmax = 1000
    atol = 1e-4
    f_in = lambda q,params: total_harmonic_line(q,params)

    collision_relaxation(q.flatten(),f_in,params,N_outer,Nmax,atol,dt,atol_min=1,visualize=False)
    