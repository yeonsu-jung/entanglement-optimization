# import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from potentials import lj, lj_collective, grad_lj_collective
# Original Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-05
# Updated: 2021-05-31


# Modified by Yeonsu Jung

" ==== FIRE: Fast Inertial Relaxation Engine ===== "

" References: "
"- Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Gumbsch, P. (2006). Structural relaxation made simple. Physical Review Letters, 97(17), 1–4. https://doi.org/10.1103/PhysRevLett.97.170201"
"- Guénolé, J., Nöhring, W. G., Vaid, A., Houllé, F., Xie, Z., Prakash, A., & Bitzek, E. (2020). Assessment and optimization of the fast inertial relaxation engine (FIRE) for energy minimization in atomistic simulations and its implementation in LAMMPS. Computational Materials Science, 175. https://doi.org/10.1016/j.commatsci.2020.109584"
def my_fire_for_jax():

" Global variables for the FIRE algorithm"
alpha0 = 0.1
Ndelay = 5
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 200000

def optimize_fire(q0,f,df,params,atol=1e-4,dt = 0.002,logoutput=False):
    error = 10*atol 
    dtmax = 10*dt
    dtmin = 0.02*dt
    alpha = alpha0
    Npos = 0

    q = q0.copy()    
    V = np.zeros(q.shape)
    F = -df(q,params)

    for i in range(Nmax):

        P = (F*V).sum() # dissipated power
        
        if (P>0):
            Npos = Npos + 1
            if Npos>Ndelay:
                dt = np.min(np.array([dt*finc,dtmax]))
                alpha = alpha*fa
        else:
            Npos = 0
            dt = np.max(np.array([dt*fdec,dtmin]))
            alpha = alpha0
            V = np.zeros(q.shape)        
        
        V = V + 0.5*dt*F
        V = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
        q = q + dt*V
        F = -df(x,params)
        V = V + 0.5*dt*F

        error = np.max(np.abs(F))
        if error < atol: break

        if logoutput: print(f(x,params),error)

    del V, F  
    return [q,f(q,params),i]

def optimize_fire2(q0,f,df,params,atol=1e-4,dt = 0.002,logoutput=False):
    error = 10*atol 
    dtmax = 10*dt
    dtmin = 0.02*dt
    alpha = alpha0
    Npos = 0
    Nneg = 0

    x = q0.copy()
    V = np.zeros(x.shape)
    F = -df(x,params)

    for i in range(Nmax):

        P = (F*V).sum() # dissipated power
        
        if (P>0):
            Npos = Npos + 1
            Nneg = 0
            if Npos>Ndelay:
                dt = np.min(dt*finc,dtmax)
                alpha = alpha*fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax: break
            if i> Ndelay:
                dt = np.max(dt*fdec,dtmin)
                alpha = alpha0
            x = x - 0.5*dt*V
            V = np.zeros(x.shape)
            
        V = V + 0.5*dt*F
        V = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
        x = x + dt*V
        F = -df(x,params)
        V = V + 0.5*dt*F

        error = np.max(np.abs(F))
        if error < atol: break

        if logoutput: print(f(x,params),error)

    del V, F  
    return [x,f(x,params),i]

############################################

if __name__ == "__main__":
    params = {"K": 1/15,
              "collision_radius": 0.00001}
    
    # F = -grad_potential_optimized(x0, params)
    # print(np.abs(F))
    from potentials import point_potential_optimized, grad_potential_optimized

    N = 10
    q0 = np.random.rand(N*2)
    
    [xmin,fmin,Niter] = optimize_fire(q0,point_potential_optimized,grad_potential_optimized,params,1e-6,logoutput=True)
    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ",Niter)

    print("Optimized potential: ", point_potential_optimized(xmin, params))
    plt.plot(xmin[:, 0], xmin[:, 1], 'ro')
    plt.show()

    # from potentials import central_force, central_force_potential
    # params = {"collision_radius":0.05}

    # N = 10
    # # q = [x1,y1,x2,y2,x3,y3, ..., xn, yn]
    # # regular grid on [0,1]x[0,1]
    # x = np.linspace(0, 1, N)
    # y = np.linspace(0, 1, N)    
    # xx, yy = np.meshgrid(x, y)
    # q0 = np.vstack((xx.flatten(), yy.flatten())).transpose().flatten()

    # q_opt, phi_opt, n_iter = optimize_fire(q0, central_force, central_force_potential, params, atol=1e-4, dt=0.0002, logoutput=False)

    # plt.plot(q0[::2], q0[1::2], 'bo')
    # plt.plot(q_opt[::2], q_opt[1::2], 'ro')
    # plt.show()


    





