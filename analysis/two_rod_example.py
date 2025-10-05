# %%
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, tree_map
from protocols import create_random_rods

@jit
def compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j

    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    tol = 0.

    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))

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
    
    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    
    return dist

@jit
def collision_penalized_entanglement_potential(q):
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
    
    collision_radius = 0.01
    dist_cont = 5.e4*(dist-collision_radius)**2
    # dist_cont = 0.
    
    # eff_pot = compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)
    # eff_pot = compute_linking_number_arai(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1) + dist_cont
    eff_pot = compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, 1)
    return eff_pot

def sph2cart(phi,theta):
    x = jnp.sin(phi)*jnp.cos(theta)
    y = jnp.sin(phi)*jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x,y,z]).transpose()

def cart2sph(x):
    phi = jnp.arccos(x[:,2])
    theta = jnp.arctan2(x[:,1],x[:,0])
    return jnp.array([phi,theta]).transpose()


def q_to_x(q):
    # q = jnp.array(q)
    q = q.reshape((-1,5))
    x = jnp.zeros((q.shape[0],6))
    x = x.at[:,:3].set(q[:,:3])
    x = x.at[:,3:6].set(sph2cart(q[:,3],q[:,4]) + x[:,0:3])
    return x

def x_to_q(x):
    # x = jnp.array(x)
    x = x.reshape((-1,6))
    q = jnp.zeros((x.shape[0],5))
    q = q.at[:,:3].set(x[:,:3])
    q = q.at[:,3:5].set(cart2sph(x[:,3:6] - x[:,:3]))
    return q

def compute_linking_number_jax(p_i1,p_i2,p_j1,p_j2):
    r_i1j1 = p_i1 - p_j1
    r_i1j2 = p_i1 - p_j2
    r_i2j1 = p_i2 - p_j1
    r_i2j2 = p_i2 - p_j2

    tol = 1e-6
    n1 = jnp.cross(r_i1j1, r_i1j2)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_i1j2, r_i2j2)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_i2j2, r_i2j1)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_i2j1, r_i1j1)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    return (-1/4/jnp.pi)*jnp.abs(jnp.arcsin(jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))


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
    num_existing_files = len(list(Path('qs').glob('*.npy')))
    k = 0
    
    for i in range(Nmax):

        # P = (F*V).sum() # dissipated power
        P = F*V
        V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
        Npos = jnp.where(P>=0,Npos+1,0)
        
        dt_choice = jnp.array([dt_array * finc, dtmax])
        
        dt_array = jnp.where(P >= 0, jnp.where(Npos > Ndelay,jnp.min(dt_choice),dt_array),dt_array)
        dt_array = jnp.where(P < 0, dt * fdec, dt)
        
        alpha = jnp.where(P >= 0,jnp.where(Npos > Ndelay,
                                alpha * fa,
                                alpha),alpha)
        
        alpha = jnp.where(P < 0, alpha0, alpha)
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
        if np.mod(i,10) == 0:
            print(f"Iteration: {i}, fval: {f(q):.7f},error: {error:.7f}")
            if callback is not None:
                callback(q)
            np.save(f"qs/q_{k+num_existing_files}.npy",np.array(q))
            k += 1
        
        if jnp.isnan(F).any():
            print("NaN detected in variables")
            
        if error < atol: break

        if logoutput: print(f(q),error)

    # del V, F  
    return q, f(q), Npos, error


# %%
x1 = jnp.array([-1.,0.,0.,1.,0.,0.])
x2 = jnp.array([0.,-1.,0.5,0.,1.,0.5])

q = create_random_rods(2)
x = q_to_x(q)

from visualizations import plot_many_rods
plot_many_rods(q.reshape(-1,5))

# %%
lk = compute_linking_number_jax(x[0,:3],x[0,3:],x[1,:3],x[1,3:])
print(lk)

# %%

