import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from optimization import optimize_fire2
from potentials import effective_potential,effective_potential_all,total_effective_potential
import numpy as onp
from matplotlib import pyplot as plt
import time

def plot_many_rods(q_pairs):
    N = q_pairs.shape[0]
    for i in range(N):
        q = q_pairs[i]
        plot_rod(q)
        
    return 1
def plot_rod(q_single):
    q_onp = onp.array(q_single)
    x1 = q_onp[0]
    y1 = q_onp[1]
    z1 = q_onp[2]
    phi1 = q_onp[3]
    theta1 = q_onp[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)
    plt.plot([x1, x11], [y1, y11], [z1, z11])
    
def plot_rods(q):
    q_onp = onp.array(q)
    x1 = q_onp[0]
    y1 = q_onp[1]
    z1 = q_onp[2]
    phi1 = q_onp[3]
    theta1 = q_onp[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)

    x2 = q_onp[5]
    y2 = q_onp[6]
    z2 = q_onp[7]
    phi2 = q_onp[8]
    theta2 = q_onp[9]

    x22 = x2 + rod_length*jnp.sin(phi2)*jnp.cos(theta2)
    y22 = y2 + rod_length*jnp.sin(phi2)*jnp.sin(theta2)
    z22 = z2 + rod_length*jnp.cos(phi2)

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([x1, x11], [y1, y11], [z1, z11])
    ax.plot([x2, x22], [y2, y22], [z2, z22])
# (q0, f, df, params, atol=1e-4, dt=0.002, logoutput=False, Nmax=10000):

def example_two_rods():    
    f = effective_potential

    rod_length = 1.0
    dist = 0.5

    num_rods = 1
    # create jnp random array
    key = random.key(0)
    p1s = random.normal(key, (num_rods,3))
    key = random.key(1)
    phi1 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(2)
    theta1 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)

    key = random.key(3)
    p2s = random.normal(key, (num_rods,3))
    key = random.key(4)
    phi2 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(5)
    theta2 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)

    q0 = jnp.concatenate([p1s, phi1, theta1, p2s, phi2, theta2], axis=1)
    q0 = q0.flatten()
    print(q0.shape)
    
    df = grad(f)
    atol = 1e-4
    dt = 0.002
    logoutput = False

    q, f_val, num_iterations = optimize_fire2(q0, f, df, atol, dt, logoutput)
     # print(f"q: {q_onp:.2f}")
    print(f"q: {q}")
    print(f"f_val: {f_val:.2f}")
    print(f"num_iterations: {num_iterations}")

    plot_rods(q0)
    plot_rods(q)
    plt.show()

def example_many_rods():    
    f = total_effective_potential # bad name...

    rod_length = 1.0
    dist = 0.5

    num_rods = 10
    # create jnp random array
    key = random.key(0)
    p1s = random.uniform(key, (num_rods,3))
    key = random.key(1)
    phi1 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(2)
    theta1 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)

    # key = random.key(3)
    # p2s = random.normal(key, (num_rods,3))
    # key = random.key(4)
    # phi2 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    # key = random.key(5)
    # theta2 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)

    q0 = jnp.concatenate([p1s, phi1, theta1], axis=1)
    # q0 = q0.flatten()
    print(q0.shape)
    
    df = grad(f)

    fval = f(q0)
    print(f"initial f_val: {fval:.2f}")

    atol = 1e-4
    dt = 0.002
    logoutput = False
    
    # indices and value for q0
    N = q0.shape[0]
    # triangular sum
    
    # time it
    start_time = time.time()

    # Your existing code here
    sz = N*(N-1)//2
    q_pairs = jnp.zeros((sz,10))        
    
    k = 0
    for i in range(N):
        q_i = q0[i]
        for j in range(i+1, N):
            q_j = q0[j]            
            q_pairs = q_pairs.at[k].set(jnp.concatenate([q_i, q_j]))            
            k += 1     
    
    q, f_val, num_iterations = optimize_fire2(q0, f, df, atol, dt, logoutput)
     # print(f"q: {q_onp:.2f}")
    print(f"q: {q}")
    print(f"f_val, initial: {fval:.2f}")
    print(f"f_val: {f_val:.2f}")
    print(f"num_iterations: {num_iterations}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_many_rods(q)
    plt.show()
    # plot_rods(q0)
    # plot_rods(q)
    # plt.show()

    


if __name__ == "__main__":
    example_many_rods()
