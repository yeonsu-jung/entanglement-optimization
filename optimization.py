import jax
import jax.numpy as jnp
from jax import jit, grad, lax
from jax.tree_util import tree_map
import numpy as onp
from functools import partial

@partial(jit, static_argnames=['f', 'df', 'dist_fn'])
def optimize_fire_jax_individual(q0, f, df, Nmax, atol=1e-4, dt=0.002, dist_fn=None, target_dist=-1.0):
    """
    Highly optimized per-DOF (degree of freedom) FIRE optimizer using JAX while_loop.
    
    Terminates when:
    - max force < atol
    - OR step >= Nmax
    - OR (if dist_fn is provided) min_dist >= target_dist
    """
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1
    Ndelay = 10
    finc = 1.1
    fdec = 0.5
    fa = 0.99

    def body_fun(carry):
        q, V, alpha, dt_array, Npos, step, _, _ = carry
        F = -df(q)
        P = F * V  # Element-wise power

        # Per-DOF time-step update
        dt_array = jnp.where(
            P > 0,
            jnp.where(Npos > Ndelay, jnp.minimum(dt_array * finc, dtmax), dt_array),
            jnp.maximum(dt_array * fdec, dtmin)
        )
        
        # Per-DOF alpha update
        alpha_array = jnp.where(
            P > 0,
            jnp.where(Npos > Ndelay, alpha * fa, alpha),
            alpha0
        )
        # We'll carry the average alpha for the next step's default
        new_alpha = jnp.mean(alpha_array)
        
        # Reset Npos for each DOF
        Npos = jnp.where(P > 0, Npos + 1, 0)

        # Steering rule - per DOF
        norm_V = jnp.linalg.norm(V)
        norm_F = jnp.linalg.norm(F)
        V = (1 - alpha_array) * V + alpha_array * F * (norm_V / (norm_F + 1e-14))
        
        # Velocity Verlet-like update
        V = V + 0.5 * dt_array * F
        q = q + dt_array * V
        F_new = -df(q)
        V = V + 0.5 * dt_array * F_new
        
        # Momentum reset: if P < 0, zero out velocity for that DOF
        V = jnp.where(P < 0, 0.0, V)

        error = jnp.max(jnp.abs(F_new))
        
        if dist_fn is not None:
            min_dist = dist_fn(q)
        else:
            min_dist = -1.0

        return q, V, new_alpha, dt_array, Npos, step + 1, error, min_dist

    def cond_fun(carry):
        _, _, _, _, _, step, error, min_dist = carry
        
        # Termination conditions
        not_converged = (error > atol)
        within_steps = (step < Nmax)
        
        # Distance-based termination:
        # If target_dist > 0, we also check if we've reached it.
        # "Strictly distance based" means we stop if we clear the distance.
        is_too_close = jnp.where(target_dist > 0,
                                 min_dist < target_dist,
                                 True)

        return not_converged & within_steps & is_too_close

    q = q0
    V = jnp.zeros_like(q0)
    dt_array = jnp.full_like(q0, dt)
    Npos = jnp.zeros_like(q0)
    step = 0
    error = 10.0 * atol
    min_dist = -1.0 # Initialize

    carry_init = (q, V, alpha0, dt_array, Npos, step, error, min_dist)

    q, V, alpha, dt_array, Npos, step, error, min_dist = lax.while_loop(
        cond_fun,
        body_fun,
        carry_init
    )
    
    return q, f(q), step, error

def optimize_fire_nonjax_individual(q0, f, df, Nmax, atol=1e-4, dt=0.002, logoutput=False, callback=None, allow_callback_break=False):
    """Per-DOF FIRE optimizer for non-JAX (CPU/Debugging) use."""
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1
    Ndelay = 5
    finc = 1.1
    fdec = 0.5
    fa = 0.99
    
    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)
    Npos = jnp.zeros(q0.shape[0])
    dt_array = jnp.full_like(q, dt)
    alpha = alpha0
    
    k = 0
    for i in range(Nmax):
        P = F * V
        norm_V = jnp.linalg.norm(V)
        norm_F = jnp.linalg.norm(F)
        V = (1 - alpha) * V + alpha * F * norm_V / (norm_F + 1e-14)
        V = jnp.clip(V, -10.0, 10.0) # Stability clip for non-jax version
        
        Npos = jnp.where(P > 0, Npos + 1, 0)
        dt_array = jnp.where(P > 0, jnp.where(Npos > Ndelay, jnp.minimum(dt_array * finc, dtmax), dt_array), dt * fdec)
        alpha = jnp.where(P > 0, jnp.where(Npos > Ndelay, alpha * fa, alpha), alpha0)
        
        V = V + 0.25 * dt_array * F
        q = q + 0.5 * dt_array * V
        F = -df(q)
        V = V + 0.25 * dt_array * F
        V = jnp.where(P < 0, 0.0, V)
        V = V + 0.25 * dt_array * F
        q = q + 0.5 * dt_array * V
        F = -df(q)
        V = V + 0.25 * dt_array * F

        error = jnp.max(jnp.abs(F))
        if i % 100 == 0:
            if logoutput:
                print(f"Iteration: {i:3d}, fval: {f(q):12.7f}, error: {error:12.7f}")
            if callback is not None:
                if callback(q, {"numbering": k}):
                    if allow_callback_break: break
            k += 1
            
        if error < atol: break
        if jnp.isnan(F).any(): break

    return q, f(q), Npos, error

@partial(jit, static_argnames=['f', 'df'])
def optimize_fire2(q0, f, df, Nmax=1e4, atol=1e-4, dt=0.002):
    """Standard global-dt FIRE optimizer using JAX while_loop."""
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha0 = 0.1
    Ndelay = 10
    finc = 1.1
    fdec = 0.5
    fa = 0.99

    def body_fun(carry):
        q, V, alpha, dt, Npos, step, _ = carry        
        F = -df(q)
        P = jnp.sum(F * V)

        dt = lax.cond(P > 0,
                      lambda d: lax.cond(Npos > Ndelay, lambda _: jnp.minimum(d * finc, dtmax), lambda _: d, None),
                      lambda d: jnp.maximum(d * fdec, dtmin),
                      dt)
        
        alpha = lax.cond(P > 0,
                         lambda a: lax.cond(Npos > Ndelay, lambda _: a * fa, lambda _: a, None),
                         lambda _: alpha0,
                         alpha)
        
        Npos = lax.cond(P > 0, lambda n: n + 1, lambda _: 0, Npos)

        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * jnp.linalg.norm(V) / (jnp.linalg.norm(F) + 1e-14)
        q = q + dt * V
        F_new = -df(q)
        V = V + 0.5 * dt * F_new
        
        error = jnp.max(jnp.abs(F_new))
        return q, V, alpha, dt, Npos, step + 1, error

    def cond_fun(carry):
        return (carry[6] > atol) & (carry[5] < Nmax)

    carry_init = (q0, jnp.zeros_like(q0), alpha0, dt, 0, 0, 10.0*atol)
    q, V, alpha, dt, Npos, step, error = lax.while_loop(cond_fun, body_fun, carry_init)
    return q, f(q), step, error

# Fallbacks for older code
def optimize_fire_nonjax_individual_with_constraint(q0,f,df,g,dg,Nmax,atol=1e-4,dt = 0.002,logoutput=False,callback=None):
    # Minimal version to keep protocols.py happy
    return optimize_fire_nonjax_individual(q0,f,df,Nmax,atol,dt,logoutput,callback)