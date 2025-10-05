# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# %%
pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240813-1651_COMPILE_N1/log_files/SingleRod-N1-AR300-Scale1-mu0.20-visc10.0-amp10.0_allLog_20240813-165125.csv'
time_line, node_list, contact_list = import_all_log(pth,max_rows=10000000)

# pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240811-1644_COMPILE_Entangled/log_files/SingleRod-N1-AR300-Scale1-mu0.20-visc1000-amp10.0_allLog_20240811-164429.csv'
# dta = np.loadtxt(pth,delimiter=',')
# print(len(dta))

# dta.shape
# time_line = dta[:,0]
# node_list = dta[:,1:]


length_fluctuation = []
centroid_evolution = []
orientation_evolution = [] 
for i in range(len(node_list)):
    r1 = node_list[i][0:3]
    r2 = node_list[i][3:6]
    
    centroid = (r1+r2)/2
    centroid_evolution.append(centroid)    
    
    orientation = r2-r1
    length_fluctuation.append(np.linalg.norm(orientation))
    orientation = orientation/np.linalg.norm(orientation)
    orientation_evolution.append(orientation)
# %%
# plt.plot(time_line,length_fluctuation)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [orientation[0] for orientation in orientation_evolution]
y = [orientation[1] for orientation in orientation_evolution]
z = [orientation[2] for orientation in orientation_evolution]

ax.plot(x, y, z,'-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add sphere for visual aid
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='r', alpha=0.2)
ax.axis('equal')

plt.show()
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [centroid[0] for centroid in centroid_evolution]
y = [centroid[1] for centroid in centroid_evolution]
z = [centroid[2] for centroid in centroid_evolution]

ax.plot(x, y, z,'-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
# %%
def mean_square_displacement_autocorrelation(u_list, dt, max_lag):
    autocorrelation = []
    time_lags = np.arange(1, max_lag + 1) * dt
    
    for lag in range(1, max_lag + 1):
        autocorrelation.append(np.mean(np.sum(u_list[lag:] * u_list[:-lag], axis=1)))
    
    return time_lags, np.array(autocorrelation)

# %%
delta_t = time_line[1] - time_line[0]
total_time = time_line[-1] - time_line[0]
max_lag = int(total_time/delta_t/10)  # Adjust based on simulation length

time_lags, acf = mean_square_displacement_autocorrelation(np.array(orientation_evolution), delta_t, max_lag)

# %% Plot the autocorrelation function
D_rot = 1
plt.figure(figsize=(5, 4))
plt.semilogx(time_lags*D_rot, acf,'.', label='Simulation')

xs = np.linspace(0, max_lag*delta_t*D_rot, 10000)
ys = np.exp(-2*xs)
plt.plot(xs, ys, 'k--', label=r'Exponential Decay $\exp(-2 D_{rot} \tau)$')

plt.title('Autocorrelation Function of the Orientation Vectors')
plt.xlabel(r'$D_{rot} \tau$')
plt.ylabel(r'Autocorrelation, $\langle u(t+\tau)\cdot u(t) \rangle$')
plt.legend()
plt.savefig('acf.png', dpi=300, bbox_inches='tight')
# %%

def mean_square_displacement_3d(x, dt, max_lag):
    msd = []
    time_lags = np.arange(1, max_lag + 1) * dt
    
    for lag in range(1, max_lag + 1):
        squared_displacement = np.mean(np.sum((x[lag:] - x[:-lag])**2, axis=1))
        msd.append(squared_displacement)
    
    return time_lags, np.array(msd)

# %%
max_lag = int(total_time/delta_t/10)  # Adjust based on simulation length
time_lags, acf = mean_square_displacement_autocorrelation(np.array(centroid_evolution), delta_t, max_lag)

# %% Plot the autocorrelation function
plt.figure(figsize=(5, 4))
plt.semilogx(time_lags*D_rot, acf,'.', label='Simulation')

# xs = np.linspace(0, max_lag*delta_t*D_rot, 10000)
# ys = np.exp(-2*xs)
# plt.plot(xs, ys, 'k--', label=r'Exponential Decay $\exp(-2 D_{rot} \tau)$')

plt.title('Autocorrelation Function of the Orientation Vectors')
plt.xlabel(r'$D_{rot} \tau$')
plt.ylabel(r'Autocorrelation, $\langle r(t+\tau) - r(t) \rangle$')
plt.legend()
plt.savefig('displacement_acf.png', dpi=300, bbox_inches='tight')

# %%
