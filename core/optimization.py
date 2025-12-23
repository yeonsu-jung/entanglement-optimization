import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as onp

from potentials import compute_linking_number_vectorized, effective_potential, total_harmonic_line
from matplotlib import pyplot as plt
from potentials import create_pairs, all_pairwise_distances

import polyscope as ps
from transforms import q_to_x
from visualizations import prep_for_polyscope


# %%
# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple, Optional, List, Dict, Any

# Third-party libraries
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, random, jit
from matplotlib import pyplot as plt

# Local application/library specific imports
# Note: Ensure these modules are in your Python path
# from optimization import optimize_fire_nonjax_individual
from potentials import (total_effective_potential, create_pairs,
                        total_harmonic_line, all_pairwise_distances)
from transforms import q_to_x
from visualizations import set_3d_plot, plot_many_rods

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Enable 64-bit precision for JAX, crucial for scientific computing
jax.config.update("jax_enable_x64", True)

CURRENT_FILE = Path(__file__).resolve()
CURRENT_FILE_NAME = CURRENT_FILE.stem

# HOME_DIR = Path.home()
# find project root (assuming this script is in a subdirectory)
# by while loop
# find entanglement-optimization directory
HOME_DIR = CURRENT_FILE.parent
while HOME_DIR.name != "entanglement-optimization":
    HOME_DIR = HOME_DIR.parent
    if HOME_DIR == HOME_DIR.parent:  # reached root without finding
        raise FileNotFoundError("Could not find 'entanglement-optimization' directory in path.")

DATA_DIR = HOME_DIR / CURRENT_FILE_NAME / "data"
FIGURES_DIR = HOME_DIR / CURRENT_FILE_NAME / "figures"
EXPORT_DIR = HOME_DIR / CURRENT_FILE_NAME / "export"
MOVIE_DIR = HOME_DIR / CURRENT_FILE_NAME / "movie"

# make directories
for directory in [DATA_DIR, FIGURES_DIR, EXPORT_DIR, MOVIE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f'Data directory: {DATA_DIR}')

def optimize_fire_nonjax_individual_with_constraint(q0,f,df,g,dg,Nmax,atol=1e-4,dt = 0.002,logoutput=False,callback=None):
    # q0: initial degrees of freedom
    # f: function to minimize
    # df: gradient of f
    # g: constraint function
    # dg: gradient of g
    # Nmax: maximum number of iterations
    # atol: absolute tolerance
    # dt: initial time step

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
    num_existing_files = len(list(Path('qs2').glob('*.npy')))
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

        # Project q onto the constraint surface
        g_val = g(q)
        # dg = jax.grad(g)
        dg_val = dg(q)
        q = q - jnp.dot(q - g_val,dg_val)*dg_val

        # q = jnp.where(g_val > 0, q - jnp.dot(g_val,dg_val)/jnp.dot(dg_val,dg_val)*dg_val, q)

        F = -df(q)
        V = V + 0.25*dt*F

        error = jnp.max(jnp.abs(F))
        if onp.mod(i,10) == 0:
            

            # TODO: add this to the callback
            from potentials import create_pairs, all_pairwise_distances

            q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
            d = all_pairwise_distances(q_pairs)

            print(f"Iteration: {i}, fval: {f(q):.7f},error: {error:.7f}, min. distance: {jnp.min(d)}")

            if callback is not None:
                callback(q)
            onp.save(f"qs2/q_{k+num_existing_files}.npy",onp.array(q))
            k += 1
        
        if jnp.isnan(F).any():
            print("NaN detected in variables")
            
        if error < atol: break

        if logoutput: print(f(q),error)

    # del V, F  
    return q, f(q), Npos, error

def optimize_fire_nonjax_individual_with_constraint2(q0,f,df,g,Nmax,atol=1e-4,dt = 0.002,logoutput=False,callback=None):
    # q0: initial degrees of freedom
    # f: function to minimize
    # df: gradient of f
    # g: constraint function
    # dg: gradient of g
    # Nmax: maximum number of iterations
    # atol: absolute tolerance
    # dt: initial time step

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

        # Project q onto the constraint surface
        g_val = g(q)
        dg = jax.grad(g)
        dg_val = dg(q)
        # q = q - jnp.dot(q - g_val,dg_val)*dg_val
        q = tree_map(lambda q: (g(q) >= 0) * q, q)


        F = -df(q)
        V = V + 0.25*dt*F

        error = jnp.max(jnp.abs(F))
        if onp.mod(i,10) == 0:
            print(f"Iteration: {i}, fval: {f(q):.7f},error: {error:.7f}")
            if callback is not None:
                callback(q)
            onp.save(f"qs/q_{k+num_existing_files}.npy",onp.array(q))
            k += 1
        
        if jnp.isnan(F).any():
            print("NaN detected in variables")
            
        if error < atol: break

        if logoutput: print(f(q),error)

    # del V, F  
    return q, f(q), Npos, error

import jax.numpy as jnp
from jax import jit, grad, tree_map
import numpy as np
from functools import partial

@partial(jit, static_argnames=['f', 'df', 'callback'])
def optimize_fire_jax_individual(q0, f, df, Nmax, atol=1e-4, dt=0.002, logoutput=False, callback=None):
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1
    alpha = alpha0
    Ndelay = 10  
    finc = 1.1  
    fdec = 0.5  
    fa = 0.99  
    Nnegmax = 200000
    error = 10 * atol  

    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)

    Npos = jnp.zeros(q0.shape[0])
    dt_array = jnp.full_like(q, dt)
    dtmax_array = jnp.full_like(q, dtmax)

    for i in range(Nmax):
        # Compute power P = F ⋅ V
        P = jnp.sum(F * V, axis=-1)

        # Update velocity using FIRE mixing rule
        V = (1 - alpha) * V + alpha * F * (jnp.linalg.norm(V) / jnp.linalg.norm(F))

        # Update dt and alpha based on power P
        dt_array = jnp.where(P >= 0, 
                             jnp.where(Npos > Ndelay, jnp.minimum(dt_array * finc, dtmax_array), dt_array),
                             dt * fdec)
        alpha = jnp.where(P >= 0, 
                          jnp.where(Npos > Ndelay, alpha * fa, alpha), 
                          alpha0)
        Npos = jnp.where(P >= 0, Npos + 1, 0)

        # Integrate positions and velocities (Verlet-like scheme)
        V += 0.25 * dt_array * F
        q += 0.5 * dt_array * V
        F = -df(q)
        V += 0.25 * dt_array * F

        # Error evaluation
        error = jnp.max(jnp.abs(F))
        if i % 10 == 0:
            print(f"Iteration: {i}, fval: {f(q):.7f}, error: {error:.7f}")
            if callback is not None:
                callback(q, {"numbering": i // 10})

        # Check for termination or NaNs
        if jnp.isnan(F).any():
            print("NaN detected in variables")
            break
        if error < atol:
            break

        if logoutput:
            print(f(q), error)

    return q, f(q), Npos, error


def optimize_fire_nonjax_individual(q0,f,df,Nmax,atol=1e-4,dt = 0.002,logoutput=False,callback=None,allow_callback_break=False):
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
    k = 0
    
    break_or_continue = False
    for i in range(Nmax):

        # P = (F*V).sum() # dissipated power
        P = F*V
        V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
        Npos = jnp.where(P>0,Npos+1,0)
        
        dt_choice = jnp.array([dt_array * finc, dtmax])
        
        dt_array = jnp.where(P > 0, jnp.where(Npos > Ndelay,jnp.min(dt_choice),dt_array),dt_array)
        dt_array = jnp.where(P <= 0, dt * fdec, dt)
        
        alpha = jnp.where(P > 0,jnp.where(Npos > Ndelay,
                                alpha * fa,
                                alpha),alpha)
        
        alpha = jnp.where(P <= 0, alpha0, alpha)
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
        if onp.mod(i,100) == 0:
            q_pairs = create_pairs(jnp.reshape(q,(-1,5)))
            d = all_pairwise_distances(q_pairs)
            print(f"Iteration: {i:3d}, "
                    f"fval: {f(q):12.7f}, "
                    f"error: {error:12.7f}, "
                    f"min. dist.: {jnp.min(d):12.7f}")

            callback_params = {"numbering": k, "min_distance": jnp.min(d)}
            if callback is not None:
                break_or_continue = callback(q, callback_params)

            if allow_callback_break and break_or_continue:
                print("Callback requested to break the loop")
                break

            k += 1
        
        # if jnp.isnan(F).any():
        #     print("NaN detected in variables")
            
        # if error < atol: break

        if logoutput: print(f(q),error)

    # del V, F  
    return q, f(q), Npos, error

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
        
        # V = V + 0.5*dt*F
        # V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
        # q = q + dt*V
        # F = -df(q)
        # V = V + 0.5*dt*F        
        
        V = V + 0.25*dt*F
        q = q + dt*V
        F = -df(q)
        V = V + 0.25*dt*F
        
        V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
        q = q + dt*V
        F = -df(q)
        V = V + 0.5*dt*F

        onp.save(f"qs/q_{i}.npy",onp.array(q))
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

def testing_individual_fire(q = None):
    from protocols import create_random_rods, create_nonintersecting_random_rods_contained
    from potentials import total_effective_potential
    from jax import grad
    import os
    
    num_rods = 100
    
    # rod length = 1
    rod_diameter = 0.001
    container_size = 1.2
    
    if q is None:
        cachefile = 'random-rod-cache.txt'
        if os.path.exists(cachefile):
            q0 = onp.loadtxt(cachefile)
        else:
            q0 = create_nonintersecting_random_rods_contained(num_rods,rod_diameter,container_size,max_attempts=10000)
            onp.savetxt(cachefile,q0)

        cachefile = 'entangled-rod-cache.txt'
        if os.path.exists(cachefile):
            q = onp.loadtxt(cachefile)
        else:
            f = total_effective_potential
            df = grad(f)
                
            Nmax = 500
            atol = 1e-4
            dt = 1e-3
            
            q0 = jnp.array(q0.flatten(),dtype=jnp.float64)
            # q, f_val, num_iterations, error = optimize_fire_nonjax_individual(q0, f, df, Nmax,atol, dt)
            q, f_val, num_iterations, error = optimize_fire_nonjax_individual(q0, f, df, Nmax,atol, dt)
            onp.savetxt(cachefile,q)
            
    from visualizations import plot_many_rods,set_3d_plot
    # fig,ax=set_3d_plot()
    # plot_many_rods(q.reshape(num_rods,-1),ax=ax)
    
    params = {"col_rad":rod_diameter/2,"amp":1.0}
    f = lambda q: total_harmonic_line(q,params)
    df = grad(f)
    
    Nmax = 500
    atol = 1e-7
    dt = 1e-3
    q, f_val, num_iterations, error = optimize_fire_nonjax_individual(q, f, df, Nmax,atol, dt)
    
    from potentials import create_pairs,all_pairwise_distances_xyz
    from transforms import q_to_x
    
    x = q_to_x(q)
    pairs = create_pairs(x)
    d = all_pairwise_distances_xyz(pairs)
    
    print(f'Rod diameter: {rod_diameter}')
    print(f'Minimum distance: {d.min()}')
    
    
    fig,ax=set_3d_plot()
    plot_many_rods(q0.reshape(num_rods,-1),ax=ax,opt_dict={'color':'k'})
    plot_many_rods(q.reshape(num_rods,-1),ax=ax)
    plt.show()
    print()
    
def main():
    from protocols import create_large_entangled_packing, create_random_rods, total_effective_potential
    from transforms import q_to_x
    from potentials import dist_lin_seg, dist_lin_seg_nonjax
    from jax import grad
    from visualizations import plot_many_rods
    # q = create_large_entangled_packing(500)

    q0 = create_random_rods(2)
    x = q_to_x(q0)
    d = dist_lin_seg_nonjax(x[0,:3],x[0,3:],x[1,:3],x[1,3:])

    params = {"col_rad":0.001,"amp":10.0}
    
    f = total_effective_potential # which is actually just linking number
    df = grad(f)

    g = lambda q: total_harmonic_line(q,params)
    dg = grad(g)


    Nmax = 1000
    q_opt, f_val, Npos, error = optimize_fire_nonjax_individual_with_constraint(q0,f,df,g,dg,Nmax,atol=1e-4,dt = 0.002,logoutput=False,callback=None)

    

    x = q_to_x(q_opt)
    d = dist_lin_seg_nonjax(x[0,:3],x[0,3:],x[1,:3],x[1,3:])
    print(d)
    
    plot_many_rods(q_opt.reshape(-1,5))
    plt.show()
    
    # testing_individual_fire(q)

# @jit
# def onestep_fire(q,f,df,atol=1e-4,dt = 0.002):
#     dtmax = 10 * dt
#     dtmin = 0.02 * dt

#     alpha0 = 0.1  # example starting value for alpha
#     alpha = alpha0
#     Ndelay = 10   # example delay for adjusting dt
#     finc = 1.1    # factor to increase dt
#     fdec = 0.5    # factor to decrease dt
#     fa = 0.99     # factor to adjust alpha
    
#     alpha0 = 0.1
#     Ndelay = 5
#     finc = 1.1
#     fdec = 0.5
#     fa = 0.99
#     Nnegmax = 200000
    
#     error = 10*atol 
    
#     dtmin = 0.02*dt
#     alpha = alpha0
    
#     Npos = jnp.zeros(q.shape[0])
#     V = jnp.zeros(q.shape)
#     F = -df(q)
#     dt_array = jnp.ones(q.shape)*dt
#     dtmax = 10*dt_array

#     # disgusting hack to save the q values
#     P = F*V
#     V = (1-alpha)*V + alpha*F*jnp.linalg.norm(V)/jnp.linalg.norm(F)
#     Npos = jnp.where(P>0,Npos+1,0)
    
#     dt_choice = jnp.array([dt_array * finc, dtmax])
#     dt_array = jnp.where(P > 0, jnp.where(Npos > Ndelay,jnp.min(dt_choice),dt_array),dt_array)
#     dt_array = jnp.where(P <= 0, dt * fdec, dt)
    
#     alpha = jnp.where(P > 0,jnp.where(Npos > Ndelay,
#                             alpha * fa,
#                             alpha),alpha)
#     alpha = jnp.where(P <= 0, alpha0, alpha)

#     V = V + 0.25*dt*F
#     q = q + 0.5*dt*V
#     F = -df(q)
#     V = V + 0.25*dt*F
#     V = tree_map(lambda v: (P >= 0) * v, V)
    
#     V = V + 0.25*dt*F
#     q = q + 0.5*dt*V
#     F = -df(q)
#     V = V + 0.25*dt*F

#     # error = jnp.max(jnp.abs(F))

#     # del V, F  
#     return q

from jax import jit, lax
import jax.numpy as jnp


def onestep_fire(q, df, atol=1e-4, dt=0.002):
    """
    Perform one step of the FIRE (Fast Inertial Relaxation Engine) algorithm.

    Parameters:
    - q: Current positions (array-like)
    - f: Force function f(q)
    - df: Function to compute the gradient of f
    - atol: Absolute tolerance for convergence
    - dt: Initial time step

    Returns:
    - q: Updated positions after one FIRE step
    """
    # Constants
    alpha0 = 0.1
    finc = 1.1
    fdec = 0.5
    fa = 0.99
    Ndelay = 10
    dtmax = 10 * dt
    dtmin = 0.02 * dt

    # Initialize variables
    alpha = alpha0
    Npos = 0
    V = jnp.zeros_like(q)
    F = -df(q)
    P = jnp.dot(F.ravel(), V.ravel())
    error = 10 * atol

    def update_velocity(V, F, alpha):
        """Update velocity using the FIRE algorithm."""
        norm_F = jnp.linalg.norm(F)
        norm_V = jnp.linalg.norm(V)
        safe_norm_F = jnp.where(norm_F > 0, norm_F, 1.0)
        return (1 - alpha) * V + alpha * F * norm_V / safe_norm_F

    def conditionally_update_parameters(P, Npos, dt, alpha):
        """Update time step and alpha based on power P."""
        def positive_power_updates():
            new_dt = jnp.minimum(dt * finc, dtmax)
            new_alpha = alpha * fa if Npos > Ndelay else alpha
            return new_dt, new_alpha

        def negative_power_updates():
            new_dt = dt * fdec
            new_alpha = alpha0
            return new_dt, new_alpha

        dt, alpha = lax.cond(
            P > 0,
            positive_power_updates,
            negative_power_updates
        )
        return dt, alpha

    # FIRE algorithm loop
    V = update_velocity(V, F, alpha)
    dt, alpha = conditionally_update_parameters(P, Npos, dt, alpha)

    # Update positions and velocities
    V = V + 0.5 * dt * F
    q = q + dt * V
    F = -df(q)
    V = V + 0.5 * dt * F

    # Reset velocity if P is negative
    V = jnp.where(P < 0, 0.0, V)

    # Convergence check (optional: uncomment if needed)
    # error = jnp.max(jnp.abs(F))
    # if error < atol:
    #     return q

    return q

    
def test_onestep_fire():
    from protocols import create_random_rods, create_nonintersecting_random_rods_contained
    from potentials import total_effective_potential

    q0 = create_random_rods(100,[25,32,12])
    f = jit(total_effective_potential)
    df = jit(grad(f))

    from potentials import total_harmonic_line

    col_rad = 0.0001
    params = {
        "col_rad": col_rad,
        "amp": 1
    }
    g = lambda q: total_harmonic_line(q,params)
    dg = grad(g)

    g(q0)

    q = q0.copy()
    q = jnp.array(q.flatten(),dtype=jnp.float64)

    entangle_step = lambda q: onestep_fire(q,df,atol=1e-4,dt=1.e-2)
    entangle_step = jit(entangle_step)

    project_step = lambda q: onestep_fire(q,dg,atol=1e-4,dt=0.001)
    project_step = jit(project_step)

    for i in range(10000):
        # q, df, atol=1e-4, dt=0.002
        q = entangle_step(q)

        # projection
        while 1:
            q_pairs = create_pairs(q.reshape(-1,5))
            distances = all_pairwise_distances(q_pairs)
            q = project_step(q)
            if (jnp.abs(col_rad - jnp.min(distances)) < 1e-1*col_rad):
                break
        # print(params["col_rad"] - jnp.min(distances))
        if i % 500 == 0:
            print(f"Iteration: {i}, min. distance: {jnp.min(distances)}")
    
    from matplotlib import pyplot as plt
    from visualizations import plot_many_rods
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    plot_many_rods(q0.reshape(-1,5),ax=ax,opt_dict={'color':'k'})
    plot_many_rods(q.reshape(-1,5),ax=ax,opt_dict={'color':'r'})
    plt.show()    

    return

def entangle_and_release():
    from protocols import create_random_rods, create_nonintersecting_random_rods_contained
    from potentials import total_effective_potential

    q0 = create_random_rods(100,[25,32,12])
    f = jit(total_effective_potential)
    df = jit(grad(f))

    from potentials import total_harmonic_line

    col_rad = 1/500/2
    params = {
        "col_rad": col_rad,
        "amp": 100
    }
    g = lambda q: total_harmonic_line(q,params)
    dg = jit(grad(g))

    q = q0.copy()
    q = jnp.array(q.flatten(),dtype=jnp.float64)

    entangle_step = lambda q: onestep_fire(q,df,atol=1e-4,dt=1.e-2)
    entangle_step = jit(entangle_step)

    project_step = lambda q: onestep_fire(q,dg,atol=1e-4,dt=1.e-2)
    project_step = jit(project_step)

    for i in range(1000):
        # q, df, atol=1e-4, dt=0.002
        q = entangle_step(q)

    for i in range(3000):
        q = project_step(q)
        if i % 500 == 0:
            q_pairs = create_pairs(q.reshape(-1,5))
            distances = all_pairwise_distances(q_pairs)
            print(f"Iteration: {i}, min. distance: {jnp.min(distances)}")
    
    from matplotlib import pyplot as plt
    from visualizations import plot_many_rods
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    plot_many_rods(q0.reshape(-1,5),ax=ax,opt_dict={'color':'k'})
    plot_many_rods(q.reshape(-1,5),ax=ax,opt_dict={'color':'r'})
    plt.show()    
    plt.savefig(FIGURES_DIR / "entangled_and_released.png", dpi=300)

    return

def optimize_two_rods():
    from potentials import total_harmonic_line, total_effective_potential

    # create two rods
    rod_length = 1.0
    alpha = 10

    rod_diameter = rod_length/alpha

    # q: x,y,z,phi,theta

    epsilon = 0.001

    xm_i = jnp.array([0.2,0.4,0])
    xm_j = jnp.array([0.4,0.1,rod_diameter + 2*epsilon])

    x_i = xm_i - jnp.array([rod_length/2,0,0])
    x_j = xm_j - jnp.array([0,rod_length/2,0])

    qi = jnp.array([x_i[0],x_i[1],x_i[2],jnp.pi/2,0])
    qj = jnp.array([x_j[0],x_j[1],x_j[2],jnp.pi/2,jnp.pi/8])


    num_rods = 2
    q = jnp.array([*qi, *qj])
    q = jnp.array(q.flatten(), dtype=jnp.float64)

    # visualize_rods_with_polyscope(q, num_rods, rod_diameter)
    # visualize with polyscope. END

    f = jit(total_effective_potential)
    df = jit(grad(f))

    
    params = {
        "col_rad": rod_diameter/2,
        "amp": 100
    }
    g = lambda q: total_harmonic_line(q,params)
    dg = jit(grad(g))

    time_step = 1.e-1
    entangle_step = lambda q: onestep_fire(q,df,atol=1e-4,dt=time_step)
    entangle_step = jit(entangle_step)

    project_step = lambda q: onestep_fire(q,dg,atol=1e-4,dt=1e-1)
    project_step = jit(project_step)


    def init_polyscope(a_list_of_curves, num_rods, rod_diameter):
        import polyscope as ps
        from transforms import q_to_x
        from visualizations import prep_for_polyscope

        ps.init()
        ps.set_autoscale_structures(False)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_ground_plane_mode("none")

        a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
        nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)
        min_z = np.min(nodes[:, 2])
        ps_curves = ps.register_curve_network( "filaments", nodes, edges )
        ps_curves.add_color_quantity( "edge_colors", edge_colors, defined_on='edges', enabled=True )
        ps_curves.set_radius( rod_diameter / 2, relative=False )

        ps.set_length_scale(2.)
        sz = 2.
        low = np.array((-sz, -sz, -sz))
        high = np.array((sz, sz, sz))
        ps.set_bounding_box(low, high)
        ps.set_up_dir("z_up")

        nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)

        return ps_curves, nodes, edges, edge_colors
    
    ps_curves, nodes, edges, edge_colors = init_polyscope(q, num_rods, rod_diameter)

    # distance function: 

    def get_dist(q):
        from potentials import create_pairs, all_pairwise_distances
        q_pairs = create_pairs(q.reshape(-1,5))
        distances = all_pairwise_distances(q_pairs)
        return jnp.min(distances)
    
    # grad_dist = jit(grad(get_dist))
    grad_dist_fn = jit(grad(get_dist))

    # test
    distances = get_dist(q)
    print(distances)
    grad_dist = grad_dist_fn(q)
    print(grad_dist)

    k = 0
    for i in range(30000):
        q = entangle_step(q)
        # q = project_step(q)
        # project by "implicit whatever"

        # # compute distances
        q_pairs = create_pairs(q.reshape(-1,5))
        distances = all_pairwise_distances(q_pairs)
        
        if jnp.min(distances) < rod_diameter:
            # time_step for projection

            # grad_dist = grad_dist_fn(q)
            # dt_proj = 1. * rod_diameter / jnp.linalg.norm(grad_dist)
            # print(f"dt_proj: {dt_proj}")
            # project_step = lambda q: onestep_fire(q,lambda q: -grad_dist_fn(q),atol=1e-4,dt=dt_proj)
            # project_step = jit(project_step)
            # q = project_step(q)
            # q_pairs = create_pairs(q.reshape(-1,5))
            # distances = all_pairwise_distances(q_pairs)

            while 1:
                q = project_step(q)
                q_pairs = create_pairs(q.reshape(-1,5))
                distances = all_pairwise_distances(q_pairs)
                if (jnp.abs(rod_diameter - jnp.min(distances)) < 1e-2*rod_diameter):
                    break

            # print(f"Projection done. Min. distance: {jnp.min(distances)}")

        if i % 300 == 0:
            a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
            ps_curves.update_node_positions(a_list_of_curves.reshape(-1,3))
            # ps_curves.get_color_quantity("edge_colors").update_values(edge_colors)
            pth = MOVIE_DIR / f"frame_{k:05d}.png"
            ps.screenshot(str(pth))
            print(f"Iteration: {i}, min. distance: {jnp.min(distances)}")
            k+=1



if __name__ == "__main__":
    # main()
    # test_onestep_fire()
    # entangle_and_release()

    optimize_two_rods()

