from numba import jit
import numpy as np
import random

# import jax.numpy as jnp
# from jax import random

from time import time

@jit(nopython=True)
def generate_non_intersecting_circles(n, max_attempts=1000):
    circles = np.zeros((n, 4))  # Initialize an array to store circle parameters
    for i in range(n):
        created = False
        attempts = 0
        while not created and attempts < max_attempts:
            # Generate random center and radius
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = random.uniform(-10, 10)
            R = random.uniform(0.5, 2.5)
            intersect = False
            
            # Check for intersection with existing circles
            for j in range(i):
                x2, y2, z2, R2 = circles[j]
                distance = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
                
                if distance < R + R2:
                    intersect = True
                    break
            
            if not intersect:
                circles[i] = np.array([x, y, z, R])
                created = True
            attempts += 1

        if attempts == max_attempts:
            print("Failed to place all circles without intersection")
            return circles[:i]  # Return only the circles that were placed successfully
            
    return circles

@jit(nopython=True)
def fixbound_nonjax(num):
    """ Ensure the number is within the bounds [0, 1]. """
    if num < 0:
        return 0
    elif num > 1:
        return 1
    return num

@jit(nopython=True)
def dist_lin_seg_nonjax(point1s, point1e, point2s, point2e):    
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)
    R = np.dot(d1, d2)

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
        u = fixbound_nonjax(-S2 / D2)
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf
    else:  # general case
        t = fixbound_nonjax((S1 * D2 - S2 * R) / den)
        u = fixbound_nonjax((t * R - S2) / D2)
        uf = fixbound_nonjax(u)
        if uf != u:
            t = fixbound_nonjax((uf * R + S1) / D1)
            u = uf

    # Compute distance
    dist = np.linalg.norm(d1 * t - d2 * u - d12)
    # vec = , (point1s + d1 * t, point2s + d2 * u)
    return dist

@jit(nopython=True)
def distance_lowerbound(point1s, point1e, point2s, point2e):    
    """ Calculate the shortest distance between two line segments. """
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    S1 = np.dot(d1, d12)
    S2 = np.dot(d2, d12)
    R = np.dot(d1, d2)

    den = D1 * D2 - R**2
    
    t = fixbound_nonjax((S1 * D2 - S2 * R) / den)
    u = fixbound_nonjax((t * R - S2) / D2)
    return np.linalg.norm(d1 * t - d2 * u - d12)

@jit(nopython=True)
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

    p_i = np.array([x_i, y_i, z_i])
    p_j = np.array([x_j, y_j, z_j])
    u_i = np.array([np.sin(phi_i)*np.cos(theta_i), np.sin(phi_i)*np.sin(theta_i), np.cos(phi_i)])
    u_j = np.array([np.sin(phi_j)*np.cos(theta_j), np.sin(phi_j)*np.sin(theta_j), np.cos(phi_j)])

    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    return dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)

# @jit(nopython=True)
def create_nonintersecting_random_rods(num_rods,rod_diameter,max_attempts):    
    q = np.zeros((num_rods,5))
    # key = random.PRNGKey(0)
    # x_all = random.uniform((max_attempts,), minval=-10, maxval=10)
    # y_all = random.uniform((max_attempts,), minval=-10, maxval=10)
    # z_all = random.uniform((max_attempts,), minval=-10, maxval=10)
    # phi_all = random.uniform((max_attempts,), minval=0, maxval=np.pi)
    # theta_all = random.uniform((max_attempts,), minval=0, maxval=2*np.pi)
    
    x_all = np.random.uniform(-10, 10, max_attempts)
    y_all = np.random.uniform(-10, 10, max_attempts)
    z_all = np.random.uniform(-10, 10, max_attempts)
    phi_all = np.random.uniform(0, np.pi, max_attempts)
    theta_all = np.random.uniform(0, 2*np.pi, max_attempts)    
    q_rand = np.array([x_all, y_all, z_all, phi_all, theta_all])
    
    for i in range(num_rods):
        created = False
        attempts = 0
        while not created and attempts < max_attempts:
            # Generate random center and radius
            # key = random.key(0
            # Generate random center and solid angle
            # key = random.PRNGKey(0)            
            # random key
            
            x = x_all[attempts]
            y = y_all[attempts]
            z = z_all[attempts]
            phi = phi_all[attempts]
            theta = theta_all[attempts]
            
            
            intersect = False
            
            if i == 0:
                q[i] = np.array([x, y, z, phi, theta])
                created = True
                break
            
            # Check for intersection with existing circles
            distance_lowerbound(q[:i],q[i])
            
            for j in range(i):
                x2, y2, z2, phi2,theta2 = q[j]
                # q_pair = np.array([q[i],q[j]])
                q_pair = np.array([x, y, z, phi, theta,x2, y2, z2, phi2, theta2])
                distance = pairwise_distance(q_pair);
                if distance < rod_diameter:
                    intersect = True
                    break
            
            if not intersect:
                # print(np.array([x, y, z, phi, theta]))
                q[i] = np.array([x, y, z, phi, theta])
                created = True
            attempts += 1
            
        if np.mod(attempts,100) == 0:
            print(f"attempts: {attempts}")

        if attempts == max_attempts:
            print("Failed to place all circles without intersection")
            return q[:i]  # Return only the circles that were placed successfully
            
    return q

def check_circle_intersection(circles):
    for i in range(len(circles)):
        x1, y1, z1, R1 = circles[i]
        for j in range(i + 1, len(circles)):
            x2, y2, z2, R2 = circles[j]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            if distance < R1 + R2:
                return False
    return True

def plot_circles(circles):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, y, z, R in circles:
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = R * np.outer(np.cos(u), np.sin(v)) + x
        y = R * np.outer(np.sin(u), np.sin(v)) + y
        z = R * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x, y, z, color='b', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return 0

# # Example usage
# n = 1000  # Number of circles to generate
# circles = generate_non_intersecting_circles(n)
# print(circles)

# # plot_circles(circles)


# t_start = time()
# TF = check_circle_intersection(circles)
# t_end = time()
# print(TF)
# print(f"Time taken: {t_end - t_start:.6f} seconds")

if __name__ == '__main__':
    num_rods = 1000
    rod_diameter = 0.2
    max_attempts = 1000
    q =create_nonintersecting_random_rods(num_rods,rod_diameter,max_attempts)
    
    print(q)
    
