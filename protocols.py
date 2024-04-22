import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from optimization import optimize_fire2,optimize_fire_debug,optimize_fire_nonjax
from potentials import effective_potential,total_effective_potential,total_effective_potential_ref,pairwise_distance,create_pairs,collision_penalized_entanglement_potential,total_harmonline_nonjax,total_gaussian_line
import numpy as onp
from matplotlib import pyplot as plt
import time
from datetime import datetime
from potentials import total_harmonic_line,simple_harmonic_line

import jax
jax.config.update("jax_enable_x64", True)

import potentials as pt
import os
    
from visualizations import plot_many_rods    

def create_random_rods(num_rods):    
    # create jnp random array
    key = random.key(0)
    p1s = random.uniform(key, (num_rods,3))
    key = random.key(1)
    phi1 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(2)
    theta1 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)
    q0 = jnp.concatenate([p1s, phi1, theta1], axis=1)    
    q0 = q0.flatten()
    q0 = jnp.array(q0,dtype=jnp.float64)
    return q0
    
def create_entangled_rods(num_rods,f,Nmax=1e4,atol=1e-4,dt=1e-3,logoutput=False,visualize=False):
    # f = total_effective_potential # bad name...    
    q0 = create_random_rods(num_rods)    
    df = grad(f)
    
    df0 = df(q0)
    print(f"Initial error: {jnp.max(jnp.abs(df0))}")
    atol = atol*jnp.max(jnp.abs(df0))
        
    q = q0
    for k in range(1):
        q, f_val, num_iterations, error = optimize_fire_nonjax(q, f, df, Nmax,atol, dt, logoutput)
        atol = atol/2
        # print(f"iteration: {k}")
        # print(f"f_val: {f_val:.2f}")
        # print(f"num_iterations: {num_iterations}")
        # print(f"error: {error}")
    
     # print(f"q: {q_onp:.2f}")
    fval0 = f(q0)
    print(f"f_val, initial: {fval0:.2f}")
    print(f"f_val: {f_val:.2f}")
    print(f"error: {error}") # which is maximum of gradient vector
    print(f"num_iterations: {num_iterations}")
    
    return q

def minimize_rod_energy(q0,f,Nmax=1e4,atol=1e-4,dt=1e-3,logoutput=False,visualize=False):
    # f = total_effective_potential # bad name...    
    df = grad(f)    
    df0 = df(q0)
    
    print(f"Initial error: {jnp.max(jnp.abs(df0))}")
    atol = atol*jnp.max(jnp.abs(df0))
        
    q = q0
    for k in range(1):
        q, f_val, num_iterations, error = optimize_fire2(q, f, df, Nmax,atol, dt, logoutput)
        atol = atol/10
        # print(f"iteration: {k}")
        # print(f"f_val: {f_val:.2f}")
        # print(f"num_iterations: {num_iterations}")
        # print(f"error: {error}")
    
     # print(f"q: {q_onp:.2f}")
    fval0 = f(q0)
    print(f"f_val, initial: {fval0:.2f}")
    print(f"f_val: {f_val:.2f}")
    print(f"error: {error}") # which is maximum of gradient vector
    print(f"num_iterations: {num_iterations}")
    
    return q

def maximally_entangled_rods(num_rods):    
    q0 = create_random_rods(num_rods)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_many_rods(jnp.reshape(q0,(-1,5)))
    
    # savefig    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    plt.savefig(f"/Users/yeonsu/Figures/initial_N{num_rods}_{dt_string}.png")
    
    start_time = time()
    q = create_entangled_rods(num_rods,total_effective_potential,Nmax=1e6,atol=1e-1,dt=1e-5,logoutput=False,visualize=False)    
    end_time = time()
    print(f"Elapsed time: {end_time-start_time}")
    
    q_onp = onp.array(q)
    onp.savetxt(f"/Users/yeonsu/Data/entangled_rods_N{num_rods}_{dt_string}.txt",q_onp)
    

def collision_relaxation(q_in,f_in,params,N_outer,Nmax,atol,dt,atol_min=1,visualize=False):    
    
    num_rods = q_in.shape[0]//5            
    q_pairs = create_pairs(jnp.reshape(q_in,(-1,5)))
    print(q_pairs.shape)
        
    q = q_in    
    for k in range(N_outer):
        params["amp"] = params["amp"]*1.3
        f = lambda q: f_in(q,params)    
        df = grad(f)
        df0 = jnp.max(jnp.abs(df(q_in)))
        print(f"Initial error: {df0}")
        
        q, f_val, num_iterations, error = optimize_fire_nonjax(q, f, df, Nmax, atol, dt, False)
        
        if (error < atol_min):
            break
        
        print(f"Outer iteration: {k}")
        print(f"f_val: {f_val:.2f}")
        print(f"num_iterations: {num_iterations}")
        print(f"error: {error}")
        print(f"dt: {dt}")
        atol = atol/1.3 # TO DO: factor out this numbers
        dt = dt/1.3     # TO DO: factor out this numbers
    
    fval0 = f(q_in)
    print(f"f_val, initial: {fval0:.2f}")
    print(f"f_val: {f_val:.2f}")
    print(f"error: {error}") # which is maximum of gradient vector
    print(f"num_iterations: {num_iterations}")
    
    return q

def sph2cart(theta,phi):
    x = jnp.sin(phi)*jnp.cos(theta)
    y = jnp.sin(phi)*jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x,y,z]).transpose()

def explode_rods(q):
    q_in_matrix = jnp.reshape(q,(-1,5))        
    expansion_factor = 1e4
    start_points = q_in_matrix[:,0:3]
    orientations = sph2cart(q_in_matrix[:,4],q_in_matrix[:,3])
    last_points = q_in_matrix[:,0:3] + orientations
    
    center = (start_points + last_points)/2
    
    foot_parameter = jnp.sum((center - start_points)*orientations,axis=1)    
    
    # (300,) times (300,3) how to implement this vectorized form?
    # tmp = jnp.sum((center - start_points)*orientations,axis=1) and orientation    
    normal_directions = -((center-start_points) - foot_parameter[:,jnp.newaxis]*orientations)
    
    start_points = start_points + normal_directions*expansion_factor
    q_exploded = jnp.concatenate([start_points,q_in_matrix[:,3:]],axis=1)
    q_exploded = q_exploded.flatten()
    
    return q_exploded

def explode_in_point_sense(q):
    # which is not we wanted.
    q_in_matrix = jnp.reshape(q,(-1,5))        
    center = jnp.mean(q_in_matrix[:,:3],axis=0)    
    expansion_factor = 10    
    start_points = q_in_matrix[:,0:3]
    orientations = sph2cart(q_in_matrix[:,4],q_in_matrix[:,3])
    last_points = q_in_matrix[:,0:3] + orientations
    centroids = (start_points + last_points)/2        
    centroids_exapnded = (centroids - center)*expansion_factor    
    start_points_exapnded = centroids_exapnded - orientations/2 + center
    q_exploded = jnp.concatenate([start_points_exapnded,q_in_matrix[:,3:]],axis=1)
    return q_exploded
    
def entangle_and_relax(num_rods,params):
    
    # entangle
    q_ent = create_entangled_rods(num_rods,total_effective_potential,Nmax=1000,atol=1.e-1,dt=1.e-3,logoutput=False,visualize=False)    
    
    # relax
    q_ent = jnp.array(q_ent,dtype=jnp.float64)
    q_rel = collision_relaxation(q_ent,total_harmonic_line,
                                 params,
                                 N_outer=5,
                                 Nmax=500,atol=1e-3,dt=1.e-3,atol_min=1e-5,
                                 visualize=False)
    
    return q_rel, q_ent
    
def relax_collision(q,params,N_outer,Nmax):
    # params = {"col_rad": 0.01, "amp": 0.1, "sigma": 0.025}
    f = lambda q: total_harmonic_line(q,params)
    df = grad(f)    
    df0 = jnp.max(jnp.abs(df(q)))
    print(f"Initial error: {df0}")
    
    # q_rel = collision_relaxation(q_ent,total_harmonic_line,
    #                              params,Nmax=1000,atol=1e-3,dt=1.e-3,atol_min=1e-5,
    #                              visualize=False)
    
    q = collision_relaxation(q,total_harmonic_line,
                                 params,
                                 N_outer=N_outer,
                                 Nmax=Nmax,atol=1e-3,dt=1.e-3,atol_min=1e-5,
                                 visualize=False)
    
    return q
    
def inspect_packing(q):    
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()   
    
    return 1

def load_data(pth):
    q = onp.loadtxt(pth)
    q = jnp.array(q,dtype=jnp.float64)
    return q

def example_Apr22():
    num_rods = 100
    
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"    
    filename = f"{cache_folder}/ent_rel_packing_N{num_rods}.txt"
    filename0 = f"{cache_folder}/ent_packing_N{num_rods}.txt"
    
    if not os.path.exists(filename): 
        params = {"col_rad": 0.005, "amp": 0.1, "sigma": 0.025} # relaxation parameters
        q,q0 = entangle_and_relax(num_rods, params)
        onp.savetxt(filename,onp.array(q))
        onp.savetxt(filename0,onp.array(q0))
    else:
        q = onp.loadtxt(filename)
        q = jnp.array(q,dtype=jnp.float64)
    
    params = {"col_rad": 0.1, "amp": 1, "sigma": 0.025}
    N_outer = 5
    Nmax = 1000
    
    # N300: Nmax = 200
    q = relax_collision(q,params,N_outer,Nmax)

    # visualization
    num_rods = q.shape[0]//5
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    from visualizations import set_3d_plot    
    set_3d_plot()
    plot_params = {"alpha": 1., "color": "blue", "linewidth": 2}
    plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)
    plot_params = {"alpha": 0.5, "linewidth": 1}
    plot_many_rods(jnp.reshape(q0,(-1,5)),plot_params)
    plt.savefig(f"/Users/yeonsu/Figures/protocolApr22_N{num_rods}_{dt_string}.png",dpi=300)
    
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/yeonsu/Figures/histogram_N{num_rods}_{dt_string}.png",dpi=300)
    
if __name__ == "__main__":
    example_Apr22()
    
    # filename = "/Users/yeonsu/Data/cache/ent_rel_packing_N300.txt"    
    # q = load_data(filename)
    # print(q.shape)
    
    # inspect_packing(q)