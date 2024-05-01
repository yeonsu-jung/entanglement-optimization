import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from optimization import optimize_fire2,optimize_fire_debug,optimize_fire_nonjax
from potentials import effective_potential,total_effective_potential,total_effective_potential_ref,pairwise_distance,create_pairs,collision_penalized_entanglement_potential,total_harmonline_nonjax,total_gaussian_line, total_harmonic_line_with_gravity_floor, total_harmonic_line_with_hook
import numpy as onp
from matplotlib import pyplot as plt
import time
from datetime import datetime
from potentials import total_harmonic_line,simple_harmonic_line

from visualizations import set_3d_plot, plot_many_rods


from transforms import q_to_x
from utils import parse_id_string

import sys

import jax
jax.config.update("jax_enable_x64", True)

import potentials as pt

import glob, os, shutil    

def create_random_rods(num_rods):
    # create jnp random array
    key = random.key(0)
    p1s = random.uniform(key, (num_rods,3), minval=-0.5, maxval=0.5)
    key = random.key(1)
    phi1 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(2)
    theta1 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)
    q0 = jnp.concatenate([p1s, phi1, theta1], axis=1)
    
    x0 = q_to_x(q0)
    center = jnp.mean(x0[:,:3],axis=0)
    # q0[:,:3] = q0[:,:3] - center    
    q0 = q0.at[:,:3].set(q0[:,:3] - center)
    
    q0 = q0.flatten()
    q0 = jnp.array(q0,dtype=jnp.float64)
    return q0

# @jit
def create_nonintersecting_random_rods(num_rods,rod_diameter,max_attempts):
    
    q = jnp.zeros((num_rods,5))
    
    for i in range(num_rods):
        created = False
        attempts = 0
        while not created and attempts < max_attempts:
            # Generate random center and radius
            # key = random.key(0
            # Generate random center and solid angle
            # key = random.PRNGKey(0)
            
            # random key
            key = random.PRNGKey(0)
            x = random.uniform(key, (), minval=-10, maxval=10)
            y = random.uniform(key, (), minval=-10, maxval=10)
            z = random.uniform(key, (), minval=-10, maxval=10)
            phi = random.uniform(key, (), minval=0, maxval=jnp.pi)
            theta = random.uniform(key, (), minval=0, maxval=2*jnp.pi)
            
            intersect = False
            
            # Check for intersection with existing circles
            for j in range(i):
                x2, y2, z2, phi2,theta2 = q[j]
                q_pair = jnp.array([q[i],q[j]])
                
                distance = 1;
                
                if distance < rod_diameter:
                    intersect = True
                    break
            
            if not intersect:
                print(jnp.array([x, y, z, phi, theta]))
                q = q.at[i].set(jnp.array([x, y, z, phi, theta]))
                created = True
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all circles without intersection")
            return q[:i]  # Return only the circles that were placed successfully
            
    return q
    
    # circles = np.zeros((n, 4))
    
    
    
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

def sph2cart(phi,theta):
    # u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    # u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])
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
    
    q = collision_relaxation(q,total_harmonic_line,
                                 params,
                                 N_outer=N_outer,
                                 Nmax=Nmax,atol=1e-3,dt=1.e-3,atol_min=1e-5,
                                 visualize=False)
    
    return q

def relax_collision_with_hook(q,params,N_outer,Nmax):
    # params = {"col_rad": 0.01, "amp": 0.1, "sigma": 0.025}
    f = lambda q: total_harmonic_line_with_hook(q,params)
    df = grad(f)    
    df0 = jnp.max(jnp.abs(df(q)))
    print(f"Initial error: {df0}")
    
    # hook structure, keep y zero, keep theta zero
    half_side = 0.05
    h1 = jnp.array([-half_side,0,+half_side,0,0])
    
    q = collision_relaxation(q,total_harmonic_line_with_hook,
                                 params,
                                 N_outer=N_outer,
                                 Nmax=Nmax,atol=1e-3,dt=1.e-3,atol_min=1e-5,
                                 visualize=False)
    
    return q

def relax_collision_with_gravity(q,params,N_outer,Nmax):
    # params = {"col_rad": 0.01, "amp": 0.1, "sigma": 0.025}
    f = lambda q: total_harmonic_line(q,params)
    df = grad(f)    
    df0 = jnp.max(jnp.abs(df(q)))
    print(f"Initial error: {df0}")
    
    
    # f = lambda q: total_harmonic_line_with_gravity_floor(q,params)
    # q_rel = collision_relaxation(q_ent,total_harmonic_line,
    #                              params,Nmax=1000,atol=1e-3,dt=1.e-3,atol_min=1e-5,
    #                              visualize=False)
    
    q = collision_relaxation(q,total_harmonic_line_with_gravity_floor,
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

def example_Apr22(num_rods,AR,dt_string,folder_name):
    
    # just for debugging phase
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    filename = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"
    filename0 = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"
    
    if not os.path.exists(filename): 
        params = {"col_rad": 0.005, "amp": 0.1, "sigma": 0.025} # relaxation parameters
        q,q0 = entangle_and_relax(num_rods, params)
        onp.savetxt(filename,onp.array(q))
        onp.savetxt(filename0,onp.array(q0))
    else:
        q = onp.loadtxt(filename)
        q0 = onp.loadtxt(filename0)
        q = jnp.array(q,dtype=jnp.float64)
        
    # just for debugging phase
    
    col_rad = 1./AR/2.
    params = {"col_rad": col_rad, "amp": 0.1, "sigma": 0.025}
    N_outer = 5
    Nmax = 1000    
    
    # N300: Nmax = 200
    q = relax_collision(q,params,N_outer,Nmax)
    
    onp.savetxt(f"{folder_name}/EntRelPacking_N{num_rods}_AR{AR}.txt",onp.array(q))

    # visualization
    num_rods = q.shape[0]//5
    
    from visualizations import set_3d_plot    
    set_3d_plot()
    plot_params = {"alpha": 1., "linewidth": 1}
    plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)
    plot_params = {"alpha": 0.5, "linewidth": 1}
    plot_many_rods(jnp.reshape(q0,(-1,5)),plot_params)
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}.png",dpi=300)
    
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_histogram_N{num_rods}.png",dpi=300)
    
def example_Apr25(num_rods,dt_string,folder_name):
    
    # just for debugging phase
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    filename = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"
    filename0 = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"
    
    # entangle
    q = create_entangled_rods(num_rods,total_effective_potential,Nmax=1000,atol=1.e-1,dt=1.e-3,logoutput=False,visualize=False)
    onp.savetxt(f"{folder_name}/EntangledPacking_N{num_rods}.txt",onp.array(q))

    # visualization
    num_rods = q.shape[0]//5
    
    from visualizations import set_3d_plot    
    set_3d_plot()
    plot_params = {"alpha": 1., "linewidth": 1}
    plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)
    plot_params = {"alpha": 0.5, "linewidth": 1}
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}.png",dpi=300)
    
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_histogram_N{num_rods}.png",dpi=300)
    
def example_Apr25_relaxation(num_rods,AR,dt_string,folder_name):
    
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    filename = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"
    filename0 = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"
    
    if not os.path.exists(filename):
        params = {"col_rad": 0.005, "amp": 0.1, "sigma": 0.025} # relaxation parameters
        q,q0 = entangle_and_relax(num_rods, params)
        onp.savetxt(filename,onp.array(q))
        onp.savetxt(filename0,onp.array(q0))
    else:
        q = onp.loadtxt(filename)
        q0 = onp.loadtxt(filename0)
        q = jnp.array(q,dtype=jnp.float64)
        
    # just for debugging phase
    
    col_rad = 1./AR/2.
    params = {"col_rad": col_rad, "amp": 0.1, "sigma": 0.025}
    N_outer = 50
    Nmax = 1000
    
    # N300: Nmax = 200
    q = relax_collision(q,params,N_outer,Nmax)

    onp.savetxt(f"{folder_name}/EntRelPacking_N{num_rods}_AR{AR}.txt",onp.array(q))

    # visualization
    num_rods = q.shape[0]//5
    
    from visualizations import set_3d_plot    
    set_3d_plot()
    plot_params = {"alpha": 1., "linewidth": 1}
    plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)
    plot_params = {"alpha": 0.5, "linewidth": 1}
    plot_many_rods(jnp.reshape(q0,(-1,5)),plot_params)
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}.png",dpi=300)
    
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_histogram_N{num_rods}.png",dpi=300)
    
    d = pt.all_pairwise_distances(q_pairs)
    rod_radius = 1/AR/2
    print(f"Minimum distance: {jnp.min(d)}")
    print(f"Distance median: {jnp.median(d)}")
    print(f"Number of rod pairs in contact: {jnp.count_nonzero(d<2*rod_radius)}")
    print(f"Total number of rod pairs: {q_pairs.shape[0]}")
    # filename = f"{cache_folder}/EntRelPacking_N{num_rods}.txt"

    # caching again
    onp.savetxt(f"{cache_folder}/EntRelPacking_N{num_rods}.txt",onp.array(q))
    
    return 1

def example_Apr30_relaxation(num_rods,AR,dt_string,N_outer,Nmax):
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    filename = f"{cache_folder}/EntangledPacking_N{num_rods}_AR{AR}.txt"
    
    if not os.path.exists(filename):
        # q0 = create_random_rods(num_rods)                     
        q0 = create_entangled_rods(num_rods,total_effective_potential,Nmax=1000,atol=1e-4,dt=1e-3,logoutput=False,visualize=False)
        
        fig,ax = set_3d_plot()
        plot_params = {"alpha": 1., "linewidth": 1}
        plot_many_rods(jnp.reshape(q0,(-1,5)),plot_params)
        ax.set_xlim([-1,2])
        ax.set_ylim([-1,2])
        ax.set_zlim([-1,2])
        plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}_AR{AR}.png",dpi=300)
        
        onp.savetxt(filename,onp.array(q0))
    else:        
        q0 = onp.loadtxt(filename)
    
    filename = f"{cache_folder}/{dt_string}/EntangledAndRelaxedPacking_N{num_rods}_AR{AR}.txt"
    
    col_rad = 1./AR/2.
    params = {"col_rad": col_rad, "amp": 10., "sigma": 0.025}    
    q = relax_collision_with_hook(q0,params,N_outer,Nmax)
        
    num_rods = q.shape[0]//5
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))            
    d = pt.all_pairwise_distances(q_pairs)
    # print bunch of messages
    print(f"Minimum distance: {jnp.min(d)}")
    print(f"Distance median: {jnp.median(d)}")
    print(f"Distance near contact: {jnp.median(d[d < 2*col_rad*(1+1.e-6)])}")
    print(f"rod radius: {col_rad}")
    print(f"Number of rod pairs in contact: {jnp.count_nonzero(d<2*col_rad)}")
    print(f"Total number of rod pairs: {q_pairs.shape[0]}")
        
    onp.savetxt(filename,onp.array(q))
    
    visualize = 1
    if visualize:
        fig,ax = set_3d_plot()
        plot_params = {"alpha": 1., "linewidth": 1}
        plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)
        # plot_params = {"color":"k", "alpha": 0.5, "linewidth": 0.5}
        # plot_many_rods(jnp.reshape(q0,(-1,5)),plot_params)
        # ax.set_xlim([-1,2])
        # ax.set_ylim([-1,2])
        # ax.set_zlim([-1,2])
        # viewing angle
        ax.view_init(elev=0, azim=90)
        
        half_side = 0.05
        h1 = jnp.array([-half_side,0,half_side,-half_side,0,-half_side])
        h2 = jnp.array([-half_side,0,-half_side,half_side,0,-half_side])
        h3 = jnp.array([half_side,0,-half_side,half_side,0,half_side])
        h4 = jnp.array([half_side,0,half_side,-half_side,0,half_side])
        h5 = jnp.array([0,0,half_side,0,0,10*half_side])
        
        # from visualizations import set_3d_plot, plot_edges
        # set_3d_plot()
        plot_edges(jnp.array([h1,h2,h3,h4,h5]),ax=ax)
        plt.axis('equal')
    
        plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}_AR{AR}.png",dpi=300)
        
        fig = plt.figure()
        plt.hist(d[d<2*col_rad*2],bins=100)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_histogram_N{num_rods}.png",dpi=300)
    
    return 1
    
def archiving():
    # dt string in YYYY-MM-DD_HH-MM-SS
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"/Users/yeonsu/Data/cache/{dt_string}"
    # hash = hashlib.md5(dt_string.encode()).hexdigest()
    
    os.makedirs(folder_name, exist_ok=False)
    
    source_dir = './'    
    # copy every py file to the folder
    files = glob.iglob(os.path.join(source_dir, "*.py"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, folder_name)
    
    return dt_string, folder_name
       
def load_data_from_cache(dt_string):
    cache_dir = f'/Users/yeonsu/Data/cache/{dt_string}'
    
    # assert(len(glob.glob(f'{cache_dir}/*.txt')) == 1, "There should be only one txt file in the folder")    
    # # find .txt file in the folder, including N and number in the filename
    # filename = glob.glob(f'{cache_dir}/*.txt')[0]
    filename = [f for f in glob.glob(f'{cache_dir}/*.txt') if 'AR' in f][0]
    print(filename)
    # check if the filename contains N and AR
    if 'N' in filename:
        splitted = parse_id_string(filename)
        for s in splitted:
            if 'N' in s:
                num_rods = int(float(s[1:]))
            if 'AR' in s:
                AR = int(float(s[2:]))
        
        print(f"num_rods: {num_rods}")
        print(f"AR: {AR}")
        
    q = load_data(filename)
    return q, num_rods, AR
    
def inspect_run(dt_string):
    cache_dir = f'/Users/yeonsu/Data/cache/{dt_string}'
    
    # assert(len(glob.glob(f'{cache_dir}/*.txt')) == 1, "There should be only one txt file in the folder")    
    # # find .txt file in the folder, including N and number in the filename
    filename = glob.glob(f'{cache_dir}/*.txt')[0]    
    print(filename)
    # check if the filename contains N and AR
    if 'N' in filename:
        splitted = parse_id_string(filename)
        for s in splitted:
            if 'N' in s:
                num_rods = int(float(s[1:]))
            if 'AR' in s:
                AR = int(float(s[2:]))
        
        print(f"num_rods: {num_rods}")
        print(f"AR: {AR}")
        
    q = load_data(filename)
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    params = {"col_rad": 1./AR/2., "amp": 1, "sigma": 0.025}
    f = lambda q: total_harmonic_line(q,params)
    f0 = f(q)
    df0 = grad(f)(q)
    
    print(f"Initial energy: {f0}")
    print(f"Initial gradient: {jnp.max(jnp.abs(df0))}")
    
    # fig = plt.figure()
    # plt.hist(d,bins=100)
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.show()
    
    from analysis import find_contacts
    rod_radius = 1/AR/2
    contacts, neighbors, contact_degrees = find_contacts(q,rod_radius)
    
    from visualizations import plot_contacts, set_3d_plot
    fig,ax = set_3d_plot()
    plot_contacts(q,0,neighbors[0])
    plt.show()
    
def inspect_packing_from_cache(dt_string):
    # num_rods = q.shape[0]//5    
    data,num_rods,AR = load_data_from_cache(dt_string)
    
    from visualizations import set_3d_plot
    set_3d_plot()
    plot_many_rods(data.reshape(-1,5))
    plt.show()
    
    data = data.reshape((-1,5))
    new_data = onp.zeros((data.shape[0],6))
    new_data[:,:3] = data[:,:3]
    
    N = data.shape[0]
    for i in range(N):
        new_data[i,3:6] = data[i,:3] + sph2cart(data[i,3],data[i,4])
        
    # export path
    export_dir = f'/Users/yeonsu/Data/export/'
    # os.makedirs(export_dir,exist_ok=False)
    plot_edges(new_data)
    
    scale_factor = 100    
    length = 1*scale_factor
    center = onp.concatenate([onp.mean(new_data[:,:3],axis=0),onp.mean(new_data[:,:3],axis=0)])
    new_data = (new_data-center)*scale_factor
    plot_edges(new_data)
    
    newfile = f'{export_dir}/{dt_string}_edges_N{num_rods}_AR{AR}_length{length}.txt'
    onp.savetxt(newfile, new_data)
    
    # from visualizations import set_3d_plot    
    # set_3d_plot()
    # plot_params = {"alpha": 1., "linewidth": 1}
    # plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)
    # plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}.png",dpi=300)
    
    # q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    # d = pt.all_pairwise_distances(q_pairs)
    
    # fig = plt.figure()
    # plt.hist(d,bins=100)
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_histogram_N{num_rods}.png",dpi=300)
    
    # d = pt.all_pairwise_distances(q_pairs)
    # rod_radius = 1/AR/2
    # print(f"Minimum distance: {jnp.min(d)}")
    # print(f"Distance median: {jnp.median(d)}")
    # print(f"Number of rod pairs in contact: {jnp.count_nonzero(d<2*rod_radius)}")
    # print(f"Total number of rod pairs: {q_pairs.shape[0]}")
    
    
    
# TO DO: move this to visualizations module
def plot_edges(edges,ax=None):
    # edges are Nx6 matrix. first 3 columns are start points, last 3 columns are end points
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    for i in range(edges.shape[0]):
        ax.plot([edges[i,0],edges[i,3]],[edges[i,1],edges[i,4]],[edges[i,2],edges[i,5]],'b')    
    
def example1():
    dt_string = '20240422-161737'
    data,num_rods,AR = load_data_from_cache(dt_string)
    
    from visualizations import set_3d_plot
    set_3d_plot()
    plot_many_rods(data.reshape(-1,5))
    plt.show()
    
    data = data.reshape((-1,5))
    new_data = onp.zeros((data.shape[0],6))
    new_data[:,:3] = data[:,:3]
    
    N = data.shape[0]
    for i in range(N):
        new_data[i,3:6] = data[i,:3] + sph2cart(data[i,3],data[i,4])
        
    # export path
    export_dir = f'/Users/yeonsu/Data/export/'
    # os.makedirs(export_dir,exist_ok=False)
    plot_edges(new_data)
    
    scale_factor = 100    
    length = 1*scale_factor
    center = onp.concatenate([onp.mean(new_data[:,:3],axis=0),onp.mean(new_data[:,:3],axis=0)])
    new_data = (new_data-center)*scale_factor
    plot_edges(new_data)
    
    newfile = f'{export_dir}/{dt_string}_edges_N{num_rods}_AR{AR}_length{length}.txt'
    onp.savetxt(newfile, new_data)
    
def protocol_for_N100_AR200():
    num_rods = 100
    AR = 200
    
    N_outer = 10
    Nmax = 500
    
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    
    dt_string, folder_name = archiving()
    num_rods = 100
    AR = 200
    
    dt_string = '20240430-195832'    
    for textfiles in glob.glob(f'{cache_folder}/{dt_string}/*.txt'):
        splitted = textfiles.split('/')[-1].split('.')[0].split('_')
        num_rods = int([x for x in splitted if 'N' in x][0].split('N')[1])
        AR = float([x for x in splitted if 'AR' in x][0].split('AR')[1])
    
    print(f"num_rods: {num_rods}")
    print(f"AR: {AR}")
    
    example_Apr30_relaxation(num_rods,AR,dt_string,N_outer,Nmax)
    
    export_scaled_packing(cache_folder,dt_string)
    
def protocol_for_N100_AR25(num_rods,AR,dt_string,N_outer,Nmax):
    
    
    example_Apr30_relaxation(num_rods,AR,dt_string,N_outer,Nmax)
    
    export_scaled_packing(cache_folder,dt_string)
    
def export_scaled_packing(cache_folder,dt_string):
    
    for textfiles in glob.glob(f'{cache_folder}/{dt_string}/*.txt'):
        splitted = textfiles.split('/')[-1].split('.')[0].split('_')
        num_rods = int([x for x in splitted if 'N' in x][0].split('N')[1])
        AR = float([x for x in splitted if 'AR' in x][0].split('AR')[1])    
    
    filename = glob.glob(f'{cache_folder}/{dt_string}/*.txt')[0]
    print(filename)
    
    q = onp.loadtxt(filename)
    x = q_to_x(q)
    x = x - onp.mean(x,axis=0)
    x = 100*x
    onp.savetxt(f'/Users/yeonsu/Data/export/EntangledRelaxedPackingXYZ_N{num_rods}_AR{AR}.txt',x)
    
if __name__ == "__main__":
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    
    num_rods = 300
    AR = 200
    N_outer = 10
    Nmax = 500
    
    if sys.argv[1] == 'new':
        dt_string, folder_name = archiving()
    else:
        dt_string = sys.argv[1]
        
        for textfiles in glob.glob(f'{cache_folder}/{dt_string}/*.txt'):
            splitted = textfiles.split('/')[-1].split('.')[0].split('_')
            num_rods = int([x for x in splitted if 'N' in x][0].split('N')[1])
            AR = float([x for x in splitted if 'AR' in x][0].split('AR')[1])    
        print(f"num_rods: {num_rods}")
        print(f"AR: {AR}")
    
    protocol_for_N100_AR25(num_rods,AR,dt_string,N_outer,Nmax)