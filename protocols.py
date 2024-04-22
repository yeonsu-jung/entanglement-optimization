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
    
    # params = {"col_rad": 0.05,
    #           "amp": 0.01}
    
    # pass this params to the function and take grad
    
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
        # q, f_val, num_iterations, error = optimize_fire2(q, f, df, Nmax, atol, dt, False)
        # atol = atol/10
        print(f"Outer iteration: {k}")
        print(f"f_val: {f_val:.2f}")
        print(f"num_iterations: {num_iterations}")
        print(f"error: {error}")
        print(f"dt: {dt}")
        atol = atol/1.3
        dt = dt/1.3
    
     # print(f"q: {q_onp:.2f}")
    fval0 = f(q_in)
    print(f"f_val, initial: {fval0:.2f}")
    print(f"f_val: {f_val:.2f}")
    print(f"error: {error}") # which is maximum of gradient vector
    print(f"num_iterations: {num_iterations}")
    
    # save data
    q_onp = onp.array(q)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    onp.savetxt(f"/Users/yeonsu/Data/entangled_rods_N30_{dt_string}.txt",q_onp)
    
    # from potentials import all_pairwise_distances
    # q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    # print(all_pairwise_distances(q_pairs))
    num_rods = q_onp.shape[0]//5
    if visualize:
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plot_many_rods(jnp.reshape(q,(-1,5)))
        plt.savefig(f"/Users/yeonsu/Figures/entanglement_and_energy_optimized_N{num_rods}_{dt_string}.png")

    return q

def collision_relaxation_fast(q_in,f_in,params,Nmax,atol,dt,atol_min=1,visualize=False):    
    
    num_rods = q_in.shape[0]//5    
        
    q_pairs = create_pairs(jnp.reshape(q_in,(-1,5)))
    print(q_pairs.shape)
    
    # params = {"col_rad": 0.05,
    #           "amp": 0.01}
    
    # pass this params to the function and take grad
    f = lambda q: f_in(q,params)
    
    df = grad(f)
    df0 = jnp.max(jnp.abs(df(q_in)))
    print(f"Initial error: {df0}")
    
    q = q_in    
    for k in range(5):
        q, f_val, num_iterations, error = optimize_fire_nonjax(q, f, df, Nmax, atol, dt, False)
        atol = atol/2
        dt = dt/2
        if (error < atol_min):
            break
        # q, f_val, num_iterations, error = optimize_fire2(q, f, df, Nmax, atol, dt, False)
        # atol = atol/10
        print(f"Outer iteration: {k}")
        # print(f"f_val: {f_val:.2f}")
        # print(f"num_iterations: {num_iterations}")
        # print(f"error: {error}")
    
     # print(f"q: {q_onp:.2f}")
    fval0 = f(q_in)
    print(f"f_val, initial: {fval0:.2f}")
    print(f"f_val: {f_val:.2f}")
    print(f"error: {error}") # which is maximum of gradient vector
    print(f"num_iterations: {num_iterations}")
    
    # save data
    q_onp = onp.array(q)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    onp.savetxt(f"/Users/yeonsu/Data/entangled_rods_N30_{dt_string}.txt",q_onp)
    
    # from potentials import all_pairwise_distances
    # q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    # print(all_pairwise_distances(q_pairs))
    num_rods = q_onp.shape[0]//5
    if visualize:
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plot_many_rods(jnp.reshape(q,(-1,5)))
        plt.savefig(f"/Users/yeonsu/Figures/entanglement_and_energy_optimized_N{num_rods}_{dt_string}.png")

    return q

def debug1():
    pth = '/Users/yeonsu/Data/entangled_rods_N100_21-04-2024_12-23-24.txt'
    q_ent = onp.loadtxt(pth)
    num_rods = q_ent.shape[0]//5

    params = {"col_rad": 0.5, "amp": 0.01}
    q_rel = collision_relaxation(q_ent,params,1000,visualize=False)
       

    # for debugging
    # there were some numerical issues regarding grad
    # q_pairs = create_pairs(jnp.reshape(q,(-1,5)))    
    # for i in range(q_pairs.shape[0]):
    #     print(pt.compute_linking_number(*q_pairs[i,:],1))
        
    # for i in range(q_pairs.shape[0]):
    #     d = pairwise_distance(q_pairs[i,:])
    #     print(d)
    
    # df = grad(total_effective_potential)
    # df(q)
    # print(jnp.max(jnp.abs(df(q))))
    
    # simple_harmonic_line(q_pairs[0,:])
    # df = grad(simple_harmonic_line)
    
    # for plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plot_many_rods(jnp.reshape(q,(-1,5)))
    # plt.savefig(f"/Users/yeonsu/Figures/entanglement_optimized_N{num_rods}_{dt_string}.png")
    
    # start_time = time()
    # q = minimize_rod_energy(q,total_harmonic_line,Nmax=1e3,atol=1e-2,dt=1e-10,logoutput=False,visualize=False)
    
    # end_time = time()
    # print(f"Elapsed time: {end_time-start_time}")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plot_many_rods(jnp.reshape(q,(-1,5)))
    # plt.savefig(f"/Users/yeonsu/Figures/entanglement_and_energy_optimized_N{num_rods}_{dt_string}.png")

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
    
    
def debug2():
    # num_rods = 150
    # q_ent = create_entangled_rods(num_rods,total_effective_potential,Nmax=200,atol=1e-3,dt=1.e-3,logoutput=False,visualize=False)
    
    pth = '/Users/yeonsu/Data/entangled_rods_N300_21-04-2024_13-06-17.txt'
    # pth = '/Users/yeonsu/Data/entangled_rods_N150_21-04-2024_12-47-17.txt'
    # pth = '/Users/yeonsu/Data/entangled_rods_N5_21-04-2024_12-49-28.txt'
    q_ent = onp.loadtxt(pth)
    
    q_ent = jnp.array(q_ent,dtype=onp.float64)        
    print(q_ent.dtype)
    
    # q_exploded = explode_rods(q_ent)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plot_many_rods(jnp.reshape(q_exploded,(-1,5)))
    
    # q_ent = create_random_rods(300)    
    # q_pairs = create_pairs(jnp.reshape(q_ent,(-1,5)))
    # from potentials import all_pairwise_distances
    # d = all_pairwise_distances(q_pairs)
    
    params = {"col_rad": 0.05, "amp": 0.01, "K": 10000}
    f = lambda q: total_gaussian_line(q,params)
    
    print(f"Initial fval: {f(q_ent)}")
    
    df = grad(f)
    print(df(q_ent))
    df0 = jnp.max(jnp.abs(df(q_ent)))    
    print(f"Initial error: {df0}")    
    # print(jnp.max(d))
    # print(jnp.min(d))
    # print(jnp.mean(d))
    
    params = {"col_rad": 0.05, "amp": 0.01}
    q_rel = collision_relaxation(q_ent,f,params,Nmax=1000,atol=1e-3,dt=1.e-2,visualize=False)
    
    
    
    print(q_rel.shape)
    
def entangle_and_relax():
    
    num_rods = 300
    # num_rods = 30; atol = 1e-1; dt = 1, Nmax = 200
    # q_ent = create_entangled_rods(num_rods,total_effective_potential,Nmax=1000,atol=1.e-1,dt=1.e-3,logoutput=False,visualize=False)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plot_many_rods(jnp.reshape(q_ent,(-1,5)))
        
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    # onp.savetxt(f"/Users/yeonsu/Data/entangled_rods_N{num_rods}_{dt_string}.txt",onp.array(q_ent))
    # plt.savefig(f"/Users/yeonsu/Figures/entangled_N{num_rods}_{dt_string}.png",dpi=300)
    
    q_ent = jnp.array(onp.loadtxt('/Users/yeonsu/Data/entangled_rods_N300_21-04-2024_15-32-32.txt'))
    
    params = {"col_rad": 0.005, "amp": 0.1, "sigma": 0.025}
    q_ent = jnp.array(q_ent,dtype=jnp.float64)
    q_rel = collision_relaxation(q_ent,total_harmonic_line,
                                 params,Nmax=1000,atol=1e-3,dt=1.e-3,atol_min=1e-5,
                                 visualize=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_many_rods(jnp.reshape(q_rel,(-1,5)))
    onp.savetxt(f"/Users/yeonsu/Data/entangled_rods_N{num_rods}_relaxed_{dt_string}.txt",onp.array(q_rel))
    plt.savefig(f"/Users/yeonsu/Figures/entanglement_and_energy_optimized_N{num_rods}_{dt_string}.png",dpi=300)
    
    q_pairs = create_pairs(jnp.reshape(q_rel,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()
    
def second_relax(q,params,N_outer,Nmax):
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
    
    # params = {"col_rad": 0.05, "amp": 0.1, "sigma": 0.025}
    # f = lambda q: total_harmonic_line(q,params)
    # df = grad(f)    
    # df0 = jnp.max(jnp.abs(df(q)))
    # print(f"Initial error: {df0}")
    # q = collision_relaxation(q,total_harmonic_line,
    #                              params,
    #                              N_outer=10,
    #                              Nmax=500,atol=1e-3,dt=1.e-3,atol_min=1e-5,
    #                              visualize=False)
    
    return q
    
def inspect_packing(q):
    # pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_21-04-2024_15-16-44.txt'
    # q = onp.loadtxt(pth)
    # q = jnp.array(q,dtype=jnp.float64)
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()
    
    
    return 1

if __name__ == "__main__":
    # entangle_and_relax()    
    # pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_21-04-2024_15-35-59.txt'
    pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_21-04-2024_23-38-05.txt'
    # pth = '/Users/yeonsu/Data/entangled_rods_N100_21-04-2024_12-49-07.txt'
    q = onp.loadtxt(pth)
    q = jnp.array(q,dtype=jnp.float64)
    
    params = {"col_rad": 0.1, "amp": 1, "sigma": 0.025}
    N_outer = 5
    Nmax = 1000
    # N300: Nmax = 200
    q = second_relax(q,params,N_outer,Nmax)

    num_rods = q.shape[0]//5
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_many_rods(jnp.reshape(q,(-1,5)))
    onp.savetxt(f"/Users/yeonsu/Data/entangled_rods_N{num_rods}_relaxed_{dt_string}.txt",onp.array(q))
    plt.savefig(f"/Users/yeonsu/Figures/entanglement_and_energy_optimized_N{num_rods}_{dt_string}.png",dpi=300)
    
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
    d = pt.all_pairwise_distances(q_pairs)
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/yeonsu/Figures/histogram_N{num_rods}_{dt_string}.png",dpi=300)
    
    
    # inspect_packing(q)