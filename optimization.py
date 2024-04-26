import jax
import jax.numpy as jnp
import numpy as onp
from potentials import compute_linking_number_vectorized, effective_potential, simple_harmonic_line
from matplotlib import pyplot as plt

def optimize_fire_nonjax(q0,f,df,Nmax,atol=1e-4,dt = 0.002,logoutput=False):
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
    dtmax = 10*dt
    dtmin = 0.02*dt
    alpha = alpha0
    Npos = 0

    q = q0.copy()    
    V = jnp.zeros(q.shape)
    F = -df(q)

    for i in range(Nmax):

        # P = (F*V).sum() # dissipated power
        P = jnp.sum(F*V)
        
        if (P>0):
            Npos = Npos + 1
            if Npos>Ndelay:
                dt = jnp.min(jnp.array([dt*finc,dtmax]))
                alpha = alpha*fa
        else:
            Npos = 0
            dt = jnp.max(jnp.array([dt*fdec,dtmin]))
            alpha = alpha0
            V = jnp.zeros(q.shape)        
        
        V = V + 0.5*dt*F
        V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
        q = q + dt*V
        F = -df(q)
        V = V + 0.5*dt*F        

        error = jnp.max(jnp.abs(F))
        if onp.mod(i,10) == 0:
            print(f"Iteration: {i}, fval: {f(q):.7f},error: {error:.7f}")
        
        if onp.isnan(F).any():            
            print("NaN detected in variables")
            
        if error < atol: break

        if logoutput: print(f(q),error)

    # del V, F  
    return q, f(q), Npos, error


def optimize_fire(q0, f, df, params, atol=1e-4, dt=0.002, logoutput=False):
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1  # example starting value for alpha
    alpha = alpha0
    Nmax = 10000  # maximum number of iterations
    Ndelay = 10   # example delay for adjusting dt
    finc = 1.1    # factor to increase dt
    fdec = 0.5    # factor to decrease dt
    fa = 0.99     # factor to adjust alpha

    def body_fun(carry, i):
        q, V, alpha, dt, Npos = carry
        F = -df(q, params)
        P = jnp.sum(F * V)  # dissipated power

        dt = jax.lax.cond(
            P > 0,
            lambda _: jnp.minimum(dt * finc, dtmax),
            lambda _: jnp.maximum(dt * fdec, dtmin),
            None
        )
        alpha = jax.lax.cond(
            P > 0,
            lambda _: alpha * fa,
            lambda _: alpha0,
            None
        )
        Npos = jax.lax.cond(P > 0, lambda _: Npos + 1, lambda _: 0, None)
        
        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * jnp.linalg.norm(V) / jnp.linalg.norm(F)
        q = q + dt * V
        F = -df(q, params)
        V = V + 0.5 * dt * F

        error = jnp.max(jnp.abs(F))
        return (q, V, alpha, dt, Npos), error

    def cond_fun(carry, error):
        return (error > atol) & (carry[4] < Nmax)

    q = q0
    V = jnp.zeros_like(q0)
    Npos = 0

    (q, V, alpha, dt, Npos), error = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (q, V, alpha, dt, Npos, 0)        
    )

    if logoutput:
        print(f(q, params), error)

    return q, f(q, params), Nmax

import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp

# @jax.jit
def optimize_fire2(q0, f, df, Nmax = 1e4, atol=1e-4, dt=0.002, logoutput=False):
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1  # example starting value for alpha
    alpha = alpha0    
    Ndelay = 10   # example delay for adjusting dt
    finc = 1.1    # factor to increase dt
    fdec = 0.5    # factor to decrease dt
    fa = 0.99     # factor to adjust alpha

    def body_fun(carry):
        q, V, alpha, dt, Npos, error = carry        
        F = -df(q)
        P = jnp.sum(F * V)  # dissipated power

        dt = jax.lax.cond(
            P > 0,
            lambda _: jax.lax.cond(Npos > Ndelay,
                                    lambda _: jnp.minimum(dt * finc, dtmax),
                                    lambda _: dt,
                                    None),                    
            lambda _: jnp.maximum(dt * fdec, dtmin),
            None
        )
        
        alpha = jax.lax.cond(
            P > 0,
            lambda _: alpha * fa,
            lambda _: alpha0,
            None
        )
        Npos = jax.lax.cond(P > 0, lambda _: Npos + 1,
                                lambda _: 0, None)

        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * jnp.linalg.norm(V) / jnp.linalg.norm(F)
        q = q + dt * V
        
        F = -df(q)
        V = V + 0.5 * dt * F
        
        Npos = Npos + 1

        error = jnp.max(jnp.abs(F))

        return q, V, alpha, dt, Npos, error

    def cond_fun(carry):
        _, _, _, _, Npos, error = carry
        return (error > atol) & (Npos < Nmax)

    q = q0
    V = jnp.zeros_like(q0)
    Npos = 0
    error = 10*atol

    carry_init = (q, V, alpha, dt, Npos, error)
    # error_init = jnp.array(10 * atol)  # Initial error to start the loop

    # q, V, alpha, dt, Npos, error = jax.lax.while_loop(
    #     cond_fun,
    #     body_fun,
    #     carry_init
    # )
    
    q, V, alpha, dt, Npos, error = jax.lax.fori_loop(0, Nmax,
                                                     lambda i, carry: body_fun(carry),
                                                     carry_init)

    return q, f(q), Npos, error

def optimize_fire_debug(q0, f, df, atol=1e-4, dt=0.002, logoutput=False):
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1  # example starting value for alpha
    alpha = alpha0
    Nmax = 10000000  # maximum number of iterations
    Ndelay = 10   # example delay for adjusting dt
    finc = 1.1    # factor to increase dt
    fdec = 0.5    # factor to decrease dt
    fa = 0.99     # factor to adjust alpha

    def body_fun(carry):
        q, V, alpha, dt, Npos, error = carry        
        F = -df(q)
        P = jnp.sum(F * V)  # dissipated power

        dt = jax.lax.cond(
            P > 0,
            lambda _: jnp.minimum(dt * finc, dtmax),
            lambda _: jnp.maximum(dt * fdec, dtmin),
            None
        )
        alpha = jax.lax.cond(
            P > 0,
            lambda _: alpha * fa,
            lambda _: alpha0,
            None
        )
        Npos = jax.lax.cond(P > 0, lambda _: Npos + 1, lambda _: 0, None)

        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * jnp.linalg.norm(V) / jnp.linalg.norm(F)
        q = q + dt * V
        
        F = -df(q)
        V = V + 0.5 * dt * F

        error = jnp.max(jnp.abs(F))
        
        # Check for NaNs in new positions and forces
        if jnp.any(jnp.isnan(q)) or jnp.any(jnp.isnan(F)):
            if logoutput:
                print("NaN detected in variables")
            return jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan  # Propagate NaNs
        
        return q, V, alpha, dt, Npos, error

    def cond_fun(carry):
        _, _, _, _, Npos, error = carry
        return (error > atol) & (Npos < Nmax) & (~jnp.any(jnp.isnan(carry)))

    q = q0
    V = jnp.zeros_like(q0)
    Npos = 0
    error = 10*atol

    carry_init = (q, V, alpha, dt, Npos, error)

    q, V, alpha, dt, Npos, error = jax.lax.while_loop(
        cond_fun,
        body_fun,
        carry_init
    )

    if jnp.any(jnp.isnan(carry_init)):
        if logoutput:
            print("NaN detected at initialization")
        return jnp.nan, jnp.nan, jnp.nan, jnp.nan  # Return NaNs if initial values are problematic

    return q, f(q), Npos, error

# scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None, tol=None, options=None)

# (q0, f, df, params, atol=1e-4, dt=0.002, logoutput=False, Nmax=10000):
def example1():
    # q0 = jnp.array([1.0, 2.0, 3.0])
    # r0 = jnp.array([1.0, 2.0, 1.0])
    # f = lambda q, params: jnp.sum((q-r0 + 1) ** 2)
    # df = lambda q, params: 2 * (q - r0)

    # f = compute_linking_number_vectorized
    f = effective_potential
    # f = simple_harmonic_line

    rod_length = 1.0
    dist = 0.5

    q0 = jnp.array([-rod_length/2,0,0,jnp.pi/2,0,0,-rod_length/2,dist,jnp.pi/2,jnp.pi/2])
    df = jax.grad(f)
    atol = 1e-4
    dt = 0.002
    logoutput = True

    q, f_val, num_iterations = optimize_fire2(q0, f, df, atol, dt, logoutput)

    # cast q to numpy array
    q_onp = onp.array(q)

    # print(f"q: {q_onp:.2f}")
    print(f"q: {q_onp}")
    print(f"f_val: {f_val:.2f}")
    print(f"num_iterations: {num_iterations}")

    x1 = q_onp[0]
    y1 = q_onp[1]
    z1 = q_onp[2]
    phi1 = q_onp[3]
    theta1 = q_onp[4]

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

    dist = jnp.linalg.norm(jnp.array([x1, y1, z1]) - jnp.array([x2, y2, z2]))
    print(f"Distance between the two rods: {dist:.2f}")

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([x1, x11], [y1, y11], [z1, z11], 'b')
    ax.plot([x2, x22], [y2, y22], [z2, z22], 'r')
    plt.show()
    
def example2():
    pass
    
def main():
    example2()
    

if __name__ == "__main__":
    main()
