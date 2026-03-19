"""FIRE (Fast Inertial Relaxation Engine) optimizers.

Two modes:
  optimize_fire      -- single lax.while_loop (fastest, no trajectory)
  make_fire_runner   -- chunked loop (slower, captures snapshots)

Carry state tuple:
  (q, V, alpha, dt_array, Npos, step, error, min_dist)
  where alpha, dt_array, Npos are per-DOF arrays.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, lax, grad
from functools import partial


# FIRE hyper-parameters (standard values)
_NDELAY = 10
_FINC   = 1.1
_FDEC   = 0.5
_FA     = 0.99
_ALPHA0 = 0.1


def _fire_body(df, dtmax, dtmin):
    """Return a single-step body_fun for use in lax.while_loop / fori_loop."""

    def body(carry):
        q, V, alpha, dt_arr, Npos, step, _err, _md = carry
        F = -df(q)
        P = F * V

        dt_arr = jnp.where(
            P > 0,
            jnp.where(Npos > _NDELAY, jnp.minimum(dt_arr * _FINC, dtmax), dt_arr),
            jnp.maximum(dt_arr * _FDEC, dtmin),
        )
        alpha_arr = jnp.where(
            P > 0,
            jnp.where(Npos > _NDELAY, alpha * _FA, alpha),
            _ALPHA0,
        )
        new_alpha = jnp.mean(alpha_arr)
        Npos = jnp.where(P > 0, Npos + 1, 0)

        nV = jnp.linalg.norm(V)
        nF = jnp.linalg.norm(F)
        V = (1 - alpha_arr) * V + alpha_arr * F * (nV / (nF + 1e-14))

        V = V + 0.5 * dt_arr * F
        q = q + dt_arr * V
        F2 = -df(q)
        V = V + 0.5 * dt_arr * F2
        V = jnp.where(P < 0, 0.0, V)

        error = jnp.max(jnp.abs(F2))
        return q, V, new_alpha, dt_arr, Npos, step + 1, error, -1.0

    return body


def fire_init_carry(q0, dt):
    """Initial FIRE carry compatible with both optimize_fire and make_fire_runner."""
    return (
        q0,
        jnp.zeros_like(q0),          # V
        _ALPHA0,                       # alpha (scalar mean)
        jnp.full_like(q0, dt),        # dt_arr (per-DOF)
        jnp.zeros_like(q0),           # Npos  (per-DOF)
        0,                             # step
        1.0,                           # error (initial large value)
        -1.0,                          # min_dist (unused internally)
    )


@partial(jit, static_argnames=["f", "df", "dist_fn"])
def optimize_fire(q0, f, df, Nmax, atol=1e-8, dt=1e-4,
                  dist_fn=None, target_dist=-1.0):
    """Full lax.while_loop FIRE — fastest, no Python interaction during run.

    Terminates when:
      max|force| < atol  OR  step >= Nmax
      OR (if dist_fn given) min_dist >= target_dist
    """
    dtmax = 10.0 * dt
    dtmin = 0.02 * dt

    body = _fire_body(df, dtmax, dtmin)

    def body_fn(carry):
        return body(carry)

    def cond_fn(carry):
        _, _, _, _, _, step, error, min_dist = carry
        not_conv  = error > atol
        in_steps  = step  < Nmax
        too_close = jnp.where(target_dist > 0, min_dist < target_dist, True)
        return not_conv & in_steps & too_close

    carry = fire_init_carry(q0, dt)
    # Inject initial min_dist if dist_fn provided
    if dist_fn is not None:
        carry = carry[:7] + (dist_fn(q0),)

    # patch body to update min_dist when dist_fn is given
    if dist_fn is not None:
        def body_fn(carry):
            new = body(carry)
            return new[:7] + (dist_fn(new[0]),)

    carry = lax.while_loop(cond_fn, body_fn, carry)
    q = carry[0]
    return q, f(q), carry[5], carry[6]


def make_fire_runner(f, df, dt):
    """Return run_chunk(carry, n) that advances FIRE by exactly n steps.

    Use this for trajectory export: call run_chunk in a Python loop,
    saving carry[0] (q) after each chunk.

    Example::

        run_chunk = make_fire_runner(potential, grad_fn, dt)
        carry = fire_init_carry(q0, dt)
        snapshots = []
        while not_converged:
            carry = run_chunk(carry, stride)
            snapshots.append(np.asarray(carry[0]))
    """
    dtmax = 10.0 * dt
    dtmin = 0.02 * dt

    body = _fire_body(df, dtmax, dtmin)

    # fori_loop body ignores loop index
    def _fori_body(_, carry):
        return body(carry)

    @partial(jit, static_argnums=[1])
    def run_chunk(carry, n):
        return lax.fori_loop(0, n, _fori_body, carry)

    return run_chunk
