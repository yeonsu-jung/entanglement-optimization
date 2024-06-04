import jax.numpy as jnp
from jax import grad,random
from optimization import optimize_fire2, optimize_fire_nonjax, optimize_fire_nonjax_individual

from potentials import total_effective_potential,create_pairs,total_harmonic_line_with_gravity_floor, total_harmonic_line_with_hook,all_distances_between_curves2,all_pairwise_distances_xyz,total_harmonic_line,all_pairwise_distances
import numpy as onp
from matplotlib import pyplot as plt

import time
from datetime import datetime
from visualizations import set_3d_plot, plot_many_rods, plot_edges
from transforms import q_to_x, x_to_rpairs, x_to_epairs
from utils import parse_id_string, archiving

import sys
import jax
jax.config.update("jax_enable_x64", True)

import potentials as pt
import glob, os, shutil

import numba
from pathlib import Path
import os

@numba.jit(nopython=True)
def fixbound_nonjax(num):
    """ Ensure the number is within the bounds [0, 1]. """
    if num < 0:
        return 0
    elif num > 1:
        return 1
    return num

@numba.jit(nopython=True)
def dist_lin_seg_nonjax(point1s, point1e, point2s, point2e):    
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = onp.dot(d1, d1)
    D2 = onp.dot(d2, d2)
    S1 = onp.dot(d1, d12)
    S2 = onp.dot(d2, d12)
    R = onp.dot(d1, d2)

    den = D1 * D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:  # line1 is a segment and line2 is a point
            u = 0
            t = fixbound_nonjax(S1 / D1)
        elif D2 != 0:  # line2 is a segment and line1 is a point
            t = 0
            u = fixbound_nonjax(-S2 / D2)
        else:  # both segments are points
            t = u = 0
    elif den == 0:  # lines are parallel
        t = 0
        u = -S2 / D2
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf
    else:  # general case
        t = fixbound_nonjax((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf

    # Compute distance
    dist = onp.linalg.norm(d1 * t - d2 * u - d12)
    # vec = , (point1s + d1 * t, point2s + d2 * u)
    return dist

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


@numba.jit(nopython=True)
def create_nonintersecting_random_rods(num_rods,rod_diameter,max_attempts=10000):
    print('create_nonintersecting_random_rods')    
    
    q = onp.zeros((num_rods, 5), dtype=onp.float64)
    
    for i in range(num_rods):
        created = False
        attempts = 0
        
        while not created and attempts < max_attempts:
            x = onp.random.uniform(-1, 1)
            y = onp.random.uniform(-1, 1)
            z = onp.random.uniform(-1, 1)
            phi = onp.random.uniform(0, onp.pi)
            theta = onp.random.uniform(0, 2 * onp.pi)
            
            intersect = False
            p_i = onp.array([x, y, z])
            p_ii = p_i + 1 * onp.array([onp.sin(phi) * onp.cos(theta), onp.sin(phi) * onp.sin(theta), onp.cos(phi)])
            
            for j in range(i):
                x2, y2, z2, phi2, theta2 = q[j]
                p_j = onp.array([x2, y2, z2])
                p_jj = p_j + 1 * onp.array([onp.sin(phi2) * onp.cos(theta2), onp.sin(phi2) * onp.sin(theta2), onp.cos(phi2)])
                
                distance = dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)
                if distance < rod_diameter:
                    intersect = True
                    break
            
            if not intersect:
                q[i] = onp.array([x, y, z, phi, theta])
                created = True
                
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all rods without intersection")
            return q[:i]  # Return only the rods that were placed successfully
        
        if i % 100 == 0:
            print(f"Rod {i} placed successfully")
    
    return q

@numba.jit(nopython=True)
def create_nonintersecting_random_rods_contained(num_rods,rod_diameter,container_size,max_attempts=10000):
    print('create_nonintersecting_random_rods in a container')        
    q = onp.zeros((num_rods, 5), dtype=onp.float64)

    for i in range(num_rods):
        created = False
        attempts = 0
        
        while not created and attempts < max_attempts:
            x = onp.random.uniform(-1,1)
            y = onp.random.uniform(-1,1)
            z = onp.random.uniform(-1,1)
            phi = onp.random.uniform(0, onp.pi)
            theta = onp.random.uniform(0, 2 * onp.pi)
            
            intersect = False
            p_i = onp.array([x, y, z])
            p_ii = p_i + 1 * onp.array([onp.sin(phi) * onp.cos(theta), onp.sin(phi) * onp.sin(theta), onp.cos(phi)])
            
            if (i == 0) & (onp.linalg.norm(p_i) > container_size or onp.linalg.norm(p_ii) > container_size):
                intersect = True
            
            for j in range(i):
                x2, y2, z2, phi2, theta2 = q[j]
                p_j = onp.array([x2, y2, z2])
                p_jj = p_j + 1 * onp.array([onp.sin(phi2) * onp.cos(theta2), onp.sin(phi2) * onp.sin(theta2), onp.cos(phi2)])
                
                distance = dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)
                if distance < rod_diameter:
                    intersect = True
                    break
                
                # print([onp.linalg.norm(p_i), onp.linalg.norm(p_ii), onp.linalg.norm(p_j), onp.linalg.norm(p_jj)])
                if (onp.linalg.norm(p_i) > container_size or onp.linalg.norm(p_ii) > container_size or onp.linalg.norm(p_j) > container_size or onp.linalg.norm(p_jj) > container_size):                    
                    intersect = True
                    break
            
            if not intersect:
                q[i] = onp.array([x, y, z, phi, theta])
                created = True
                
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all rods without intersection")
            return q[:i]  # Return only the rods that were placed successfully
        
        if i % 100 == 0:
            print(f"Rod {i} placed successfully")

    return q

@numba.jit(nopython=True)
def create_nonintersecting_random_rods_contained_in_box(num_rods, rod_diameter, container_size, max_attempts=10000):
    q = onp.zeros((num_rods, 5), dtype=onp.float64)

    for i in range(num_rods):
        created = False
        attempts = 0
        
        while not created and attempts < max_attempts:
            x = onp.random.uniform(-container_size/2, container_size/2)
            y = onp.random.uniform(-container_size/2, container_size/2)
            z = onp.random.uniform(-container_size/2, container_size/2)
            phi = onp.random.uniform(0, onp.pi)
            theta = onp.random.uniform(0, 2 * onp.pi)
            
            intersect = False
            p_i = onp.array([x, y, z])
            p_ii = p_i + onp.array([onp.sin(phi) * onp.cos(theta), onp.sin(phi) * onp.sin(theta), onp.cos(phi)])
            
            # Check if the rod's endpoints are within the box boundaries
            if (onp.any(p_i < -container_size/2) or onp.any(p_i > container_size/2) or
                onp.any(p_ii < -container_size/2) or onp.any(p_ii > container_size/2)):
                intersect = True
            
            for j in range(i):
                x2, y2, z2, phi2, theta2 = q[j]
                p_j = onp.array([x2, y2, z2])
                p_jj = p_j + onp.array([onp.sin(phi2) * onp.cos(theta2), onp.sin(phi2) * onp.sin(theta2), onp.cos(phi2)])
                
                distance = dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)
                if distance < rod_diameter:
                    intersect = True
                    break
                
                # Check if the rod's endpoints are within the box boundaries
                if (onp.any(p_j < -container_size/2) or onp.any(p_j > container_size/2) or
                    onp.any(p_jj < -container_size/2) or onp.any(p_jj > container_size/2)):
                    intersect = True
                    break
            
            if not intersect:
                q[i] = onp.array([x, y, z, phi, theta])
                created = True
                
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all rods without intersection")
            return q[:i]  # Return only the rods that were placed successfully
        
        if i % 100 == 0:
            print(f"Rod {i} placed successfully")

    return q

@numba.jit(nopython=True)
def create_nonintersecting_random_rods_contained_in_noncube(num_rods, rod_diameter, container_size, max_attempts=1000000):
    assert(len(container_size) == 3)
    q = onp.zeros((num_rods, 5), dtype=onp.float64)

    for i in range(num_rods):
        created = False
        attempts = 0
        
        while not created and attempts < max_attempts:
            x = onp.random.uniform(-container_size[0]/2, container_size[0]/2)
            y = onp.random.uniform(-container_size[1]/2, container_size[1]/2)
            z = onp.random.uniform(-container_size[2]/2, container_size[2]/2)
            
            tmp = onp.random.uniform(-1,1)
            phi = onp.arccos(tmp)
            theta = onp.random.uniform(0, 2 * onp.pi)
            
            intersect = False
            p_i = onp.array([x, y, z])
            p_ii = p_i + onp.array([onp.sin(phi) * onp.cos(theta), onp.sin(phi) * onp.sin(theta), onp.cos(phi)])
            
            # Check if the rod's endpoints are within the box boundaries
            if (onp.any(p_i < -container_size/2) or onp.any(p_i > container_size/2) or
                onp.any(p_ii < -container_size/2) or onp.any(p_ii > container_size/2)):
                intersect = True
            
            for j in range(i):
                x2, y2, z2, phi2, theta2 = q[j]
                p_j = onp.array([x2, y2, z2])
                p_jj = p_j + onp.array([onp.sin(phi2) * onp.cos(theta2), onp.sin(phi2) * onp.sin(theta2), onp.cos(phi2)])
                
                distance = dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)
                if distance < rod_diameter:
                    intersect = True
                    break
                
                # Check if the rod's endpoints are within the box boundaries
                if (onp.any(p_j < -container_size/2) or onp.any(p_j > container_size/2) or
                    onp.any(p_jj < -container_size/2) or onp.any(p_jj > container_size/2)):
                    intersect = True
                    break
            
            if not intersect:
                q[i] = onp.array([x, y, z, phi, theta])
                created = True
                
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all rods without intersection")
            return q[:i]  # Return only the rods that were placed successfully
        
        if i % 100 == 0:
            print(f"Rod {i} placed successfully")

    return q

@numba.jit(nopython=True)
def create_nonintersecting_random_rods_com_contained_sphere(num_rods,rod_diameter,container_size,max_attempts=10000):
    print('create_nonintersecting_random_rods whose centroids in a container')        
    q = onp.zeros((num_rods, 5), dtype=onp.float64)

    for i in range(num_rods):
        created = False
        attempts = 0
        
        while not created and attempts < max_attempts:
            x = onp.random.uniform(-1,1)
            y = onp.random.uniform(-1,1)
            z = onp.random.uniform(-1,1)
            phi = onp.random.uniform(0, onp.pi)
            theta = onp.random.uniform(0, 2 * onp.pi)
            
            intersect = False
            p_i = onp.array([x, y, z])
            p_ii = p_i + 1 * onp.array([onp.sin(phi) * onp.cos(theta), onp.sin(phi) * onp.sin(theta), onp.cos(phi)])
            com_i = (p_i + p_ii)/2
            
            if (i == 0) & (onp.linalg.norm(com_i) > container_size):
                intersect = True
            
            for j in range(i):
                x2, y2, z2, phi2, theta2 = q[j]
                p_j = onp.array([x2, y2, z2])
                p_jj = p_j + 1 * onp.array([onp.sin(phi2) * onp.cos(theta2), onp.sin(phi2) * onp.sin(theta2), onp.cos(phi2)])
                com_j = (p_j + p_jj)/2
                
                distance = dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)
                if distance < rod_diameter:
                    intersect = True
                    break
                
                # print([onp.linalg.norm(p_i), onp.linalg.norm(p_ii), onp.linalg.norm(p_j), onp.linalg.norm(p_jj)])
                if (onp.linalg.norm(com_i) > container_size or onp.linalg.norm(com_j) > container_size):
                    intersect = True
                    break
            
            if not intersect:
                q[i] = onp.array([x, y, z, phi, theta])
                created = True
                
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all rods without intersection")
            return q[:i]  # Return only the rods that were placed successfully
        
        if i % 100 == 0:
            print(f"Rod {i} placed successfully")

    return q

def create_entangled_rods(num_rods,f,Nmax=1e4,atol=1e-4,dt=1e-3):
    # f = total_effective_potential # bad name...    
    q0 = create_random_rods(num_rods)
    df = grad(f)
    
    df0 = df(q0)
    print(f"Initial error: {jnp.max(jnp.abs(df0))}")
    atol = atol*jnp.max(jnp.abs(df0))
        
    q = q0
    for k in range(1):
        q, f_val, num_iterations, error = optimize_fire_nonjax_individual(q, f, df, Nmax,atol, dt)
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
        
        x = q_to_x(q)
        
        with open(f"/Users/yeonsu/Data/relaxation_process.txt", "ab") as file:
            # write a line
            for i in range(q.shape[0]):
                file.write(f"{q[i]:.4f}".encode())
                file.write(b" ")
            file.write(b"\n")
        
        pairs = create_pairs(x)
        d = all_pairwise_distances_xyz(pairs)
        col_rad = 1./AR/2.*scale_factor
        
        packing_id = f'Entrel-N{num_rods}-AR{AR}-Scale{scale_factor}'
        print_distance_info(d,col_rad,packing_id,export_folder)
        
        if ( jnp.abs(jnp.min(d) - 2*col_rad)/2*col_rad < 1e-2):
            break
        
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
                                 Nmax=Nmax,atol=1e-5,dt=1.e-3,atol_min=1e-9,
                                 visualize=False)
    
    return q

def relax_collision_with_hook(q,params,N_outer,Nmax):
    # params = {"col_rad": 0.01, "amp": 0.1, "sigma": 0.025}
    f = lambda q: total_harmonic_line_with_hook(q,params)
    df = grad(f)    
    df0 = jnp.max(jnp.abs(df(q)))
    print(f"Initial error: {df0}")
    
    # hook structure, keep y zero, keep theta zero
    half_side = 0.25
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
    # q = relax_collision_with_hook(q0,params,N_outer,Nmax)
    q = relax_collision(q0,params,N_outer,Nmax)
        
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
        
        half_side = 0.25
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

def example_May1_relaxation(num_rods,AR,dt_string,N_outer,Nmax):
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
    q = relax_collision(q0,params,N_outer,Nmax)
    # q = relax_collision_with_hook(q0,params,N_outer,Nmax)
        
    num_rods = q.shape[0]//5
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))            
    d = pt.all_pairwise_distances(q_pairs)    
    
    print_distance_info(d,col_rad,dt_string,)
    onp.savetxt(filename,onp.array(q,dtype=onp.float64))
    
    visualize = 1
    if visualize:
        fig,ax = set_3d_plot()
        plot_params = {"alpha": 1., "linewidth": 1}
        plot_many_rods(jnp.reshape(q,(-1,5)),plot_params)        
        ax.view_init(elev=0, azim=90)
        
        half_side = 0.25
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
def plot_edges(edges,ax=None,plot_params={}):
    # edges are Nx6 matrix. first 3 columns are start points, last 3 columns are end points
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    for i in range(edges.shape[0]):
        ax.plot([edges[i,0],edges[i,3]],[edges[i,1],edges[i,4]],[edges[i,2],edges[i,5]],**plot_params)
    
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
    
    N_outer = 2
    Nmax = 500
    
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    
    dt_string, folder_name = archiving()
    num_rods = 300
    AR = 200
    
    dt_string = '20240430-195832'    
    for textfiles in glob.glob(f'{cache_folder}/{dt_string}/*.txt'):
        splitted = textfiles.split('/')[-1].split('.')[0].split('_')
        num_rods = int([x for x in splitted if 'N' in x][0].split('N')[1])
        AR = float([x for x in splitted if 'AR' in x][0].split('AR')[1])
    
    print(f"num_rods: {num_rods}")
    print(f"AR: {AR}")
    
    # example_Apr30_relaxation(num_rods,AR,dt_string,N_outer,Nmax)
    example_May1_relaxation(num_rods,AR,dt_string,N_outer,Nmax)
    
    export_scaled_packing(cache_folder,dt_string)
    
def print_distance_info(d,col_rad,packing_id,export_folder):
    
    print(f"rod radius: {col_rad}")
    print(f"rod diameter: {2*col_rad}")
    print(f"Minimum distance: {jnp.min(d)}")
    print(f"Distance median: {jnp.median(d)}")    
    
    # log in a file
    if not os.path.exists(f'{export_folder}/distance_info'):
        os.makedirs(f'{export_folder}/distance_info')
    with open(f'{export_folder}/distance_info/{packing_id}_distance_info.txt','a') as f:
        f.write(f"rod radius: {col_rad}\n")
        f.write(f"rod diameter: {2*col_rad}\n")
        f.write(f"Minimum distance: {jnp.min(d)}\n")
        f.write(f"Distance median: {jnp.median(d)}\n")        
    

def protocol_for_N100_AR25(num_rods,AR,dt_string,N_outer,Nmax):
    # example_Apr30_relaxation(num_rods,AR,dt_string,N_outer,Nmax)
    example_May1_relaxation(num_rods,AR,dt_string,N_outer,Nmax)
    export_scaled_packing(cache_folder,dt_string)
    
def export_scaled_packing(cache_folder,dt_string,scale_factor):
    
    for textfiles in glob.glob(f'{cache_folder}/{dt_string}/*.txt'):
        splitted = textfiles.split('/')[-1].split('.')[0].split('_')
        num_rods = int([x for x in splitted if 'N' in x][0].split('N')[1])
        AR = float([x for x in splitted if 'AR' in x][0].split('AR')[1])    
    
    filename = glob.glob(f'{cache_folder}/{dt_string}/*.txt')[0]
    print(filename)
    
    q = onp.loadtxt(filename)
    x = q_to_x(q)
    center = jnp.mean((x[:,:3] + x[:,3:])/2,axis=0)
    x = x - jnp.array([*center,*center])
    
    # print_distance_info(d,col_rad)
    
    onp.savetxt(f'/Users/yeonsu/Data/export/EntangledRelaxedPackingXYZ_N{num_rods}_AR{AR}.txt',x)    
        
    center = jnp.mean((x[:,:3] + x[:,3:])/2,axis=0)
    x = x - jnp.array([*center,*center])
    x = scale_factor*x
    
    r = x_to_rpairs(x.flatten(),num_rods)
    i, j = jnp.triu_indices(num_rods, k=1)
    r_i = r[i]  # Shape will be (N(N-1)/2, M)
    r_j = r[j]  # Shape will be (N(N-1)/2, M)
    
    curve_pairs = jnp.concatenate([r_i,r_j],axis=1)
    half_size = curve_pairs.shape[1]//2
    pairs1 = curve_pairs[:,:half_size]
    pairs2 = curve_pairs[:,half_size:]
    
    d = all_distances_between_curves2(pairs1,pairs2)
    print(jnp.min(d))
    
    
    
    e = x_to_epairs(nodes_mat,num_rods)
    r = x_to_rpairs(nodes_mat,num_rods)
    
    i, j = jnp.triu_indices(num_rods, k=1)
    r_i = r[i]  # Shape will be (N(N-1)/2, M)
    r_j = r[j]  # Shape will be (N(N-1)/2, M)
    
    curve_pairs = jnp.concatenate([r_i,r_j],axis=1)
    half_size = curve_pairs.shape[1]//2
    pairs1 = curve_pairs[:,:half_size]
    pairs2 = curve_pairs[:,half_size:]
    
    d = all_distances_between_curves2(pairs1,pairs2)
    print(jnp.min(d))
    
    d = all_pairwise_distances_xyz(pairs)
    col_rad = 1./AR/2.*100
    print_distance_info(d,col_rad)
    
    fig,ax = set_3d_plot()
    plot_edges(x,ax=ax)
    
    
    half_side = 25
    h1 = jnp.array([-half_side,0,half_side,-half_side,0,-half_side])
    h2 = jnp.array([-half_side,0,-half_side,half_side,0,-half_side])
    h3 = jnp.array([half_side,0,-half_side,half_side,0,half_side])
    h4 = jnp.array([half_side,0,half_side,-half_side,0,half_side])
    h5 = jnp.array([0,0,half_side,0,0,10*half_side])
    
    # from visualizations import set_3d_plot, plot_edges
    # set_3d_plot()
    plot_edges(jnp.array([h1,h2,h3,h4,h5]),ax=ax,params={"color":"k", "alpha": 0.5, "linewidth": 2.5})
    plt.axis('equal')
    
    
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    
    plot_edges(x)
    plt.show()
    
    onp.savetxt(f'/Users/yeonsu/Data/export/EntangledRelaxedPackingXYZ_N{num_rods}_AR{AR}.txt',x)
    
def inspect_():    
    # pth = '/Users/yeonsu/Data/export/EntangledRelaxedPackingXYZ_N100_AR200.0.txt'
    pth = '/Users/yeonsu/Documents/GitHub/dismech-rods-main/data/PHYSICAL_EntangledRelaxedPackingXYZ_N100_AR200.0.txt'
    data = onp.loadtxt(pth, skiprows=0)
    data = jnp.array(data, dtype=jnp.float64)

    # from visualizations import plot_edges
    # plot_edges(data)
    
    pairs = create_pairs(data)
    print(pairs[:,:6].shape)

    d = all_pairwise_distances_xyz(pairs)
    print(d.shape)
    
    col_rad = 1./200/2.*100
    print(f"Minimum distance: {jnp.min(d)}")
    print(f"Distance median: {jnp.median(d)}")
    print(f"Distance near contact: {jnp.median(d[d < 2*col_rad*(1+1.e-6)])}")
    print(f"rod radius: {col_rad}")
    print(f"Number of rod pairs in contact: {jnp.count_nonzero(d<2*col_rad)}")
    print(f"Total number of rod pairs: {pairs.shape[0]}")
    
def check_sanity_q_to_x(q):
    num_rods = q.shape[0]//5
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))            
    d = pt.all_pairwise_distances(q_pairs)
    # print bunch of messages
    
    print_distance_info(d,col_rad)
        
    x = q_to_x(q)
    pairs = create_pairs(x)
    d = all_pairwise_distances_xyz(pairs)
    col_rad = 1./AR/2
    print_distance_info(d,col_rad)
    
    # plot_params = {"color":"k", "alpha": 0.5, "linewidth": 0.5}
    # plot_edges(x,ax=ax,plot_params=plot_params)
    
    return 0

def create_entrel_packing_with_hook(num_rods,AR,dt_string,N_outer,Nmax,scale_factor):
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    filename = f"{cache_folder}/EntangledPackingHook_N{num_rods}_AR{AR}.txt"
    
    if not os.path.exists(filename):        
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
    
    filename = f"{cache_folder}/{dt_string}/EntangledAndRelaxedPacking-N{num_rods}-AR{AR}.txt"
    col_rad = 1./AR/2.
    params = {"col_rad": col_rad, "amp": 10., "sigma": 0.025}
        
    q = relax_collision_with_hook(q0,params,N_outer,Nmax)        
    x = q_to_x(q)
    
    center = jnp.mean((x[:,:3] + x[:,3:])/2,axis=0)
    x = x - jnp.array([*center,*center])    
    x = scale_factor*x    
    packing_id = f'EntangledRelaxedPackingHook-N{num_rods}-AR{AR}-Scale{scale_factor}'
    onp.savetxt(f'/Users/yeonsu/Data/export/{packing_id}.txt',x)
    
    pairs = create_pairs(x)
    d = all_pairwise_distances_xyz(pairs)
    col_rad = 1./AR/2.*scale_factor
    print_distance_info(d,col_rad,packing_id)
    
    from visualizations import plot_edges    
    
    fig,ax = set_3d_plot()
    plot_edges(x,ax=ax)    
    half_side = 50
    h1 = jnp.array([-half_side,0,half_side,-half_side,0,-half_side])
    h2 = jnp.array([-half_side,0,-half_side,half_side,0,-half_side])
    h3 = jnp.array([half_side,0,-half_side,half_side,0,half_side])
    h4 = jnp.array([half_side,0,half_side,-half_side,0,half_side])
    h5 = jnp.array([0,0,half_side,0,0,10*half_side])    
    
    plot_edges(jnp.array([h1,h2,h3,h4,h5]),ax=ax,params={"color":"k", "alpha": 0.5, "linewidth": 2.5})
    plt.axis('equal')
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_N{num_rods}_AR{AR}.png",dpi=300)
        
    fig = plt.figure()
    plt.hist(d,bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_histogram_N{num_rods}.png",dpi=300)
    return 0


def create_entrel_packing(num_rods,AR,dt_string,N_outer,Nmax,scale_factor,q0=None):
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    filename = f"{cache_folder}/EntangledPacking_N{num_rods}_AR{AR}.txt"
    
    if q0 is None:
        if not os.path.exists(filename):
            q0 = create_entangled_rods(num_rods,total_effective_potential,Nmax=1000,atol=1e-8,dt=1e-3)
            fig,ax = set_3d_plot()
            plot_params = {"alpha": 1., "linewidth": 1}
            plot_many_rods(jnp.reshape(q0,(-1,5)),plot_params)
            ax.set_xlim([-1,2])
            ax.set_ylim([-1,2])
            ax.set_zlim([-1,2])
            plt.savefig(f"/Users/yeonsu/Figures/{dt_string}_CachedEntPack_N{num_rods}_AR{AR}.png",dpi=150)
            onp.savetxt(filename,onp.array(q0))
        else:        
            q0 = onp.loadtxt(filename)
    
    
    col_rad = 1./AR/2.
    params = {"col_rad": col_rad, "amp": 10., "sigma": 0.025} # 10, 0.025 for initial batch
        
    q = relax_collision(q0,params,N_outer,Nmax)
    x = q_to_x(q)
    
    center = jnp.mean((x[:,:3] + x[:,3:])/2,axis=0)
    x = x - jnp.array([*center,*center])    
    x = scale_factor*x
    return x
    

def create_packing():
    
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    # dt_string = '20240502-104016'
    # export_scaled_packing(cache_folder,dt_string)
    
    num_rods = 100
    AR = 100
    N_outer = 20
    Nmax = 500
    scale_factor = 100
    
    if sys.argv[1] == 'new':
        dt_string, folder_name = archiving()
        create_entrel_packing_with_hook(num_rods,AR,dt_string,N_outer,Nmax,scale_factor)
    else:
        dt_string = sys.argv[1]
        
        for textfiles in glob.glob(f'{cache_folder}/{dt_string}/*.txt'):
            splitted = textfiles.split('/')[-1].split('.')[0].split('_')
            num_rods = int([x for x in splitted if 'N' in x][0].split('N')[1])
            AR = float([x for x in splitted if 'AR' in x][0].split('AR')[1])    
        print(f"num_rods: {num_rods}")
        print(f"AR: {AR}")
        
        export_scaled_packing(cache_folder,dt_string,1)
        
def create_multiple_packings():
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
        
    # num_rods = 100
    # AR = 100
    
    N_outer = 20
    Nmax = 500
    scale_factor = 100
        
    for num_rods in [100,200,300]:
        for AR in [20,50,100,200,500,1000]:
            dt_string, folder_name = archiving()
            create_entrel_packing_with_hook(num_rods,AR,dt_string,N_outer,Nmax,scale_factor)
   
def two_rings():
    # a,b,c = two_rings()    
    # filename = f"two_rings.txt"
    # filepath = f"{export_folder}/{filename}"
    # data_out = onp.array([a.flatten(),b.flatten(),c.flatten()])
    # print(data_out)
    # onp.savetxt(filepath,data_out)
    
    s = onp.linspace(0, 2*onp.pi,10)
    a = onp.array([onp.cos(s), onp.sin(s), onp.zeros_like(s)]).T
    b = onp.array([onp.cos(s)+3./2., onp.zeros_like(s), onp.sin(s)]).T
    c = onp.array([onp.cos(s)-3./2., onp.zeros_like(s), onp.sin(s)]).T        
    
    fig,ax = set_3d_plot()
    plt.plot(a[:,0],a[:,1],a[:,2])
    plt.plot(b[:,0],b[:,1],b[:,2])
    plt.plot(c[:,0],c[:,1],c[:,2])
    
    return a,b,c

def generate_batch():
    packing_batch_id = sys.argv[1]
    print(f"Creating a set of packings for batch: {packing_batch_id}")
    
    data_folder = '/Users/yeonsu/Data/'
    cache_folder = f"{data_folder}/cache"
    export_folder = f"{data_folder}/export/{packing_batch_id}"
        
    N_outer = 20
    Nmax = 500
    scale_factor = 1
        
    # num_rods = 100
    # AR = 20
    
    for num_rods in [100,200,300]:
        for AR in [20,50,100,200,500,1000]:
            dt_string, folder_name = archiving()
            create_entrel_packing(num_rods,AR,dt_string,N_outer,Nmax,scale_factor)
            
def test_create_random_rods():
    # pth = '/Users/yeonsu/Data/cache/EntangledPackingHook_N300_AR200.txt'    
    # from protocols import create_random_rods
    
    # num_rods = 300    
    # scale_factor = 1
    
    # q0 = create_random_rods(num_rods)    
    # plot_many_rods(q0.reshape(-1,5))
    # plt.savefig(f"/Users/yeonsu/Figures/RandomPacking-N{num_rods}-Scale{scale_factor}.png")
    
    # x = q_to_x(q0)    
    # center = jnp.mean((x[:,:3] + x[:,3:])/2,axis=0)
    # x = x - jnp.array([*center,*center])    
    # x = scale_factor*x
    # packing_id = f'RandomPacking-N{num_rods}-Scale{scale_factor}'
    # np.savetxt(f'/Users/yeonsu/Data/export/{packing_id}.txt',x)
    return 0

def random_nonintersection_contained_protocol():
    num_rods = 300
    AR = 200
    scale_factor = 1
    rod_diameter = scale_factor/AR
    max_attempts = 10000000
    print(f"Rod diameter: {rod_diameter}")
    
    container_size = 0.7
    packing_id = f'RandomNonintersectingPackingContained-N{num_rods}-AR{AR}-Scale{scale_factor}-Container{container_size}'
    # length is fixed 1
    
    q = create_nonintersecting_random_rods_contained(num_rods,rod_diameter,container_size,max_attempts=max_attempts)
    plot_many_rods(q.reshape(-1,5))
    plt.savefig(f"/Users/yeonsu/Figures/{packing_id}.png",dpi=300)
    
    pairs = create_pairs(q.reshape(-1,5))
    d = all_pairwise_distances(pairs)
    print(onp.min(d))
    
    x = q_to_x(q)
    center = jnp.mean((x[:,:3] + x[:,3:])/2,axis=0)
    x = x - jnp.array([*center,*center])    
    x = scale_factor*x
    
    onp.savetxt(f'/Users/yeonsu/Data/export/{packing_id}.txt',x)
    
def create_large_entangled_packing(num_rods):
    # assumming num_rods > 500
    key = random.key(0)
    p1s = random.uniform(key, (num_rods,3), minval=-0.005, maxval=0.005)
    
    key = random.key(1)
    phi1 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(2)
    theta1 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)
    
    x1 = -0.5*jnp.cos(theta1)*jnp.sin(phi1)
    y1 = -0.5*jnp.sin(theta1)*jnp.sin(phi1)
    z1 = -0.5*jnp.cos(phi1)
    
    x2 = 0.5*jnp.cos(theta1)*jnp.sin(phi1)
    y2 = 0.5*jnp.sin(theta1)*jnp.sin(phi1)
    z2 = 0.5*jnp.cos(phi1)
    
    x0 = jnp.concatenate([x1,y1,z1,x2,y2,z2],axis=1)
    x0 = x0 + jnp.concatenate([p1s,p1s],axis=1)
    # plot_edges(x0)
    
    def x_to_q(x):
        orientation = x[:,3:]-x[:,:3]
        
        phi = jnp.arccos(orientation[:,2])
        theta = jnp.arctan2(orientation[:,1],orientation[:,0])
        q = jnp.concatenate([x[:,:3],phi[:,None],theta[:,None]],axis=1)
        return q.flatten()
    
    q0 = x_to_q(x0)
    plot_many_rods(q0.reshape(-1,5))
    print()
    
    return q0.flatten()
    
class logger():
    def __init__(self,log_file):
        self.log_file = log_file
        with open(self.log_file,'w') as f:
            f.write('Logging starts\n')
        
    def log(self,message):
        with open(self.log_file,'a') as f:
            f.write(message)
            f.write('\n')
    
    
def batch_process():
    packing_batch_id = sys.argv[1]
    assert(packing_batch_id is not None)
    print(f"Creating a set of packings for batch: {packing_batch_id}")
    
    data_folder = '/Users/yeonsu/Data/'
    
    
    
    
    cache_folder = f"{data_folder}/cache"
    export_folder = f"{data_folder}/export/{packing_batch_id}"
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
        
    # Initial batch: 20, 500, 1
    N_outer = 10
    Nmax = 500
    scale_factor = 1
        
    # num_rods = 100
    # AR = 20
    
    # initial batch for Jesse and Nacho
    # for num_rods in [100,200,300]:
    #     for AR in [20,50,100,200,500,1000]:
    
    if not os.path.exists(f'{export_folder}/figures'):
        os.makedirs(f'{export_folder}/figures')
    
    dt_string, _ = archiving(export_folder)
    
    # N > 300 needs a bit different parameters
    #
    from protocols import create_large_entangled_packing
    for num_rods in [500]:
        q0 = create_large_entangled_packing(num_rods)
        for AR in [200,300]:
            x = create_entrel_packing(num_rods,AR,dt_string,N_outer,Nmax,scale_factor,q0=q0)
            
            packing_id = f'Entrel-N{num_rods}-AR{AR}-Scale{scale_factor}'
            onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
            
            pairs = create_pairs(x)
            d = all_pairwise_distances_xyz(pairs)
            col_rad = 1./AR/2.*scale_factor
            print_distance_info(d,col_rad,packing_id,export_folder)
            
            fig,ax = set_3d_plot()
            plot_edges(x,ax=ax)
            plt.axis('equal')
            plt.savefig(f"{export_folder}/figures/{dt_string}_N{num_rods}_AR{AR}_Scale{scale_factor}.png",dpi=300)
                
            fig = plt.figure()
            plt.hist(d,bins=100)
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.savefig(f"{export_folder}/figures/{dt_string}_histogram_N{num_rods}_AR{AR}_Scale{scale_factor}.png",dpi=300)
            
def example_nibox():
    num_rods = 100
    container_size = 2
    rod_diameter = 0.01
    q = create_nonintersecting_random_rods_contained_in_box(num_rods,rod_diameter,container_size,max_attempts=10000000)
    print(q.shape)
    
    fig,ax= set_3d_plot()
    plot_many_rods(q.reshape(-1,5))
    print
    
def sanity_check():
    q = create_nonintersecting_random_rods_contained_in_box(num_rods=num_rods,
                                                                    rod_diameter=rod_diameter,
                                                                    container_size=container_size,
                                                                    max_attempts=1000000)
    x = q_to_x(q)    
    x = onp.array(x)
    
    num_rods = 300
    rod_diameter = 1/20/2
    container_size = 1.5
    density_factor = num_rods/container_size**3*rod_diameter*1**2
    print(f'Density factor: {density_factor}')
    
    @numba.jit(nopython=True)    
    def fast_dist_calc(x,num_rods):
        for i in range(num_rods):
            p_i = x[i,:3]
            p_f = x[i,3:]
            for j in range(i+1,num_rods):
                q_i = x[j,:3]
                q_f = x[j,3:]
                d = dist_lin_seg_nonjax(p_i,p_f,q_i,q_f)
                if d < 0.01:
                    print(f"Distance between {i} and {j}: {d}")
                    
        return d
    
    fast_dist_calc(x,num_rods)

def McFlurry():
    batch_id = 'MacFlurry'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
    
    density_factor = 5
    container_size = 1.5
    for AR in [20,50,100,200,300,500]:
        for num_rods in [100]:
            rod_diameter = (1./AR)
            # num_rods = int(container_size**3*density_factor/rod_diameter)
            # print(f'Num rods: {num_rods}')
            density_factor = num_rods/(container_size-1/2)**3*rod_diameter*1**2
            
            q = create_nonintersecting_random_rods_contained_in_box(num_rods=num_rods,
                                                                    rod_diameter=rod_diameter,
                                                                    container_size=container_size,
                                                                    max_attempts=1000000)
            x = q_to_x(q)
            onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
            fig,ax= set_3d_plot()
            plot_edges(x,ax=ax)
            ax.set_title(f'Num rods: {x.shape[0]}')
            plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
            plt.close()
            
            # onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
            
    print
    
def Hatban():
    batch_id = 'Hatban'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
    
    density_factor = 5
    container_size = 1
    for AR in [20,50,100,200,300,500]:
        for num_rods in [100]:
            rod_diameter = (1./AR)
            # num_rods = int(container_size**3*density_factor/rod_diameter)
            # print(f'Num rods: {num_rods}')
            density_factor = num_rods/(container_size-1/2)**3*rod_diameter*1**2
            
            q = create_nonintersecting_random_rods_com_contained_sphere(num_rods=num_rods,
                                                                    rod_diameter=rod_diameter,
                                                                    container_size=container_size,
                                                                    max_attempts=1000000)
            x = q_to_x(q)
            onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
            fig,ax= set_3d_plot()
            plot_edges(x,ax=ax)
            ax.set_title(f'Num rods: {x.shape[0]}')
            plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
            plt.close()
            
            # onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
    
def Powerade():
    batch_id = 'Powerade'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
    
    density_factor = 5
    container_size = onp.array([1.2,1.2,2])
    for AR in [50,100,200,300,500]:
        for num_rods in [100,500]:
            rod_diameter = (1./AR)
            # num_rods = int(container_size**3*density_factor/rod_diameter)
            # print(f'Num rods: {num_rods}')
            container_volume = (container_size[0] - 1/2)*(container_size[1] - 1/2)*(container_size[2] - 1/2)
            density_factor = num_rods/(container_volume)**3*rod_diameter*1**2
            print(density_factor)
            
            q = create_nonintersecting_random_rods_contained_in_noncube(num_rods=num_rods,
                                                                    rod_diameter=rod_diameter,
                                                                    container_size=container_size,
                                                                    max_attempts=1000000)
            x = q_to_x(q)
            onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
            fig,ax= set_3d_plot()
            plot_edges(x,ax=ax)
            ax.set_title(f'Num rods: {x.shape[0]}')
            plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
            plt.close()
            
def CarrotCake2():
    # time for makikng a batch
    # Chuck    
    batch_id = 'CarrotCake2'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
        
    # copy this file
    import shutil
    shutil.copyfile('protocols.py',data_folder/'protocols.py')    
    
    density_factor = 5
    container_size = onp.array([0.9,0.9,2])
    log_file = data_folder/'log.txt'
    log_string = ''
    
    for AR in [50,100,200,300,500]:
        # for num_rods in [100,500]:
        container_volume = 1
        rod_diameter = (1./AR)
        num_rods = int(container_volume*density_factor/rod_diameter)
        # print(f'Num rods: {num_rods}')
        
        # density_factor = num_rods/(container_volume)**3*rod_diameter*1**2
        print(num_rods)
        
        q = create_nonintersecting_random_rods_contained_in_noncube(num_rods=num_rods,
                                                                rod_diameter=rod_diameter,
                                                                container_size=container_size,
                                                                max_attempts=1000000)
        x = q_to_x(q)
        onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
        
        log_string += "============================================\n"
        log_string += f'Num rods: {num_rods}\n'
        log_string += f'AR: {AR}\n'
        log_string += f'Density factor: {density_factor}\n'
        log_string += f'Rod diameter: {rod_diameter}\n'
        log_string += f'Container size: {container_size}\n'
        
        fig,ax= set_3d_plot()
        plot_edges(x,ax=ax)
        ax.set_title(f'Num rods: {x.shape[0]}')
        ax.view_init(0,0)
        ax.axis('equal')
        plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
        plt.close()
        
            # onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
    
    with open(log_file,'w') as f:
        f.write(log_string)
    print
    
def CarrotCake3():
    batch_id = 'CarrotCake3'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
        
    # copy this file
    import shutil
    shutil.copyfile('protocols.py',data_folder/'protocols.py')    
    
    density_factor = 5
    container_size = onp.array([1,1,1])
    log_file = data_folder/'log.txt'
    log_string = ''
    
    for AR in [50,100,200,300,500]:
        # for num_rods in [100,500]:
        container_volume = 1
        rod_diameter = (1./AR)
        num_rods = int(container_volume*density_factor/rod_diameter)
        # print(f'Num rods: {num_rods}')
        container_size = onp.array([1.3,1.3,1.3]) - rod_diameter/2*1.1
        
        # density_factor = num_rods/(container_volume)**3*rod_diameter*1**2
        print(num_rods)
        
        q = create_nonintersecting_random_rods_contained_in_noncube(num_rods=num_rods,
                                                                rod_diameter=rod_diameter,
                                                                container_size=container_size,
                                                                max_attempts=1000000)
        x = q_to_x(q)
        onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
        
        log_string += "============================================\n"
        log_string += f'Num rods: {num_rods}\n'
        log_string += f'AR: {AR}\n'
        log_string += f'Density factor: {density_factor}\n'
        log_string += f'Rod diameter: {rod_diameter}\n'
        log_string += f'Container size: {container_size}\n'
        
        fig,ax= set_3d_plot()
        plot_edges(x,ax=ax)
        ax.set_title(f'Num rods: {x.shape[0]}')
        ax.view_init(0,0)
        ax.axis('equal')
        plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
        plt.close()
        
            # onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
    
    with open(log_file,'w') as f:
        f.write(log_string)
    print
def CarrotCake4():
    # time for makikng a batch
    # Chuck    
    batch_id = 'CarrotCake4'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
        
    # copy this file
    import shutil
    shutil.copyfile('protocols.py',data_folder/'protocols.py')    
    
    density_factor = 5
    container_size = onp.array([1,1,1])
    log_file = data_folder/'log.txt'
    log_string = ''
    
    container_size = onp.array([0.9,0.9,2])
    
    for AR in [50,100,200,300,500]:
        # for num_rods in [100,500]:
        container_volume = 1
        rod_diameter = (1./AR)
        num_rods = int(container_volume*density_factor/rod_diameter)
        # print(f'Num rods: {num_rods}')
        
        
        # density_factor = num_rods/(container_volume)**3*rod_diameter*1**2
        print(num_rods)
        
        q = create_nonintersecting_random_rods_contained_in_noncube(num_rods=num_rods,
                                                                rod_diameter=rod_diameter,
                                                                container_size=container_size,
                                                                max_attempts=1000000)
        x = q_to_x(q)
        onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
        
        log_string += "============================================\n"
        log_string += f'Num rods: {num_rods}\n'
        log_string += f'AR: {AR}\n'
        log_string += f'Density factor: {density_factor}\n'
        log_string += f'Rod diameter: {rod_diameter}\n'
        log_string += f'Container size: {container_size}\n'
        
        fig,ax= set_3d_plot()
        plot_edges(x,ax=ax)
        ax.set_title(f'Num rods: {x.shape[0]}')
        ax.view_init(0,0)
        ax.axis('equal')
        plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
        plt.close()
        
            # onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
    
    with open(log_file,'w') as f:
        f.write(log_string)
    print
    

if __name__ == "__main__":
    # time for makikng a batch
    # Chuck    
    batch_id = 'CarrotCake5'
    data_folder = Path('/Users/yeonsu/Data/export/') / batch_id
    visual_folder = data_folder/'visuals'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
        
    # copy this file
    import shutil
    shutil.copyfile('protocols.py',data_folder/'protocols.py')    
    
    density_factor = 5
    container_size = onp.array([1,1,1])
    log_file = data_folder/'log.txt'
    log_string = ''
    
    container_size = onp.array([0.9,0.9,2])
    
    for AR in [25,50,75,100,125,200,300]:
        # for num_rods in [100,500]:
        container_volume = 1
        rod_diameter = (1./AR)
        num_rods = int(container_volume*density_factor/rod_diameter)
        # print(f'Num rods: {num_rods}')
        
        
        # density_factor = num_rods/(container_volume)**3*rod_diameter*1**2
        print(num_rods)
        
        q = create_nonintersecting_random_rods_contained_in_noncube(num_rods=num_rods,
                                                                rod_diameter=rod_diameter,
                                                                container_size=container_size,
                                                                max_attempts=1000000)
        x = q_to_x(q)
        onp.savetxt(data_folder/f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.txt',x)
        
        log_string += "============================================\n"
        log_string += f'Num rods: {num_rods}\n'
        log_string += f'AR: {AR}\n'
        log_string += f'Density factor: {density_factor}\n'
        log_string += f'Rod diameter: {rod_diameter}\n'
        log_string += f'Container size: {container_size}\n'
        
        fig,ax= set_3d_plot()
        plot_edges(x,ax=ax)
        ax.set_title(f'Num rods: {x.shape[0]}')
        ax.view_init(0,0)
        ax.axis('equal')
        plt.savefig(visual_folder / f'NonIntersectingBox-N{num_rods}-AR{AR}-Scale1.png',dpi=300)
        plt.close()
        
            # onp.savetxt(f'{export_folder}/{packing_id}.txt',x)
    
    with open(log_file,'w') as f:
        f.write(log_string)
    print
    