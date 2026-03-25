"""FIRE (Fast Inertial Relaxation Engine) optimizers.

Two modes:
  optimize_fire      -- single lax.while_loop (fastest, no trajectory)
  make_fire_runner   -- chunked loop (slower, captures snapshots)

Carry state tuple:
  (q, V, alpha, dt, Npos, step, error, min_dist)
  where alpha, dt, Npos are global scalars — standard FIRE, not per-DOF.
  This matches the benchmark gpu_relax_collision pattern.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


# FIRE hyper-parameters (standard values)
_NDELAY = 10
_FINC   = 1.1
_FDEC   = 0.5
_FA     = 0.99
_ALPHA0 = 0.1


def _fire_body(df, dtmax, dtmin):
    """Return a single-step body_fun — global FIRE with scalar dt/alpha/Npos.

    Algorithm (matches benchmark gpu_relax_collision):
      1. Compute F = -df(q)
      2. P = sum(F·V)  (global power, scalar)
      3. Reset V <- 0 if P <= 0
      4. Update dt, alpha, Npos based on P
      5. Half-step V, apply FIRE mixing, full position step, second half-step
    """
    def body(carry):
        q, V, alpha, dt, Npos, step, _err, _md = carry
        F = -df(q)
        P = jnp.sum(F * V)
        P_pos = P > 0

        # Velocity reset when P <= 0
        V = jnp.where(P_pos, V, jnp.zeros_like(V))

        # Update FIRE scalars
        dt    = jnp.where(P_pos,
                    jnp.where(Npos > _NDELAY,
                              jnp.minimum(dt * _FINC, dtmax), dt),
                    jnp.maximum(dt * _FDEC, dtmin))
        alpha = jnp.where(P_pos,
                    jnp.where(Npos > _NDELAY, alpha * _FA, alpha),
                    jnp.float64(_ALPHA0))
        Npos  = jnp.where(P_pos, Npos + 1, jnp.int32(0))

        # Leapfrog half-step, FIRE mixing, position update
        V_half = V + 0.5 * dt * F
        nV     = jnp.linalg.norm(V_half)
        nF     = jnp.linalg.norm(F)
        V_mix  = jnp.where(nF > 1e-12,
                            (1.0 - alpha) * V_half + alpha * F * (nV / nF),
                            V_half)
        q  = q + dt * V_mix
        F2 = -df(q)
        V  = V_mix + 0.5 * dt * F2

        error = jnp.max(jnp.abs(F2))
        return q, V, alpha, dt, Npos, step + 1, error, _md

    return body


def fire_init_carry(q0, dt):
    """Initial FIRE carry compatible with both optimize_fire and make_fire_runner."""
    return (
        q0,
        jnp.zeros_like(q0),          # V
        jnp.float64(_ALPHA0),         # alpha (scalar)
        jnp.float64(dt),              # dt    (scalar)
        jnp.int32(0),                 # Npos  (scalar)
        jnp.int32(0),                 # step
        jnp.float64(1.0),             # error
        jnp.float64(-1.0),            # min_dist (unused until dist_fn injected)
    )


@partial(jit, static_argnames=["f", "df", "dist_fn", "dtmax_factor"])
def optimize_fire(q0, f, df, Nmax, atol=1e-8, dt=1e-4,
                  dtmax_factor=10.0,
                  dist_fn=None, target_dist=-1.0):
    """Full lax.while_loop FIRE — fastest, no Python interaction during run.

    Terminates when:
      max|force| < atol  OR  step >= Nmax
      OR (if dist_fn given) min_dist >= target_dist
    """
    dtmax = dtmax_factor * dt
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

    if dist_fn is not None:
        carry = carry[:7] + (dist_fn(q0),)
        def body_fn(carry):
            new = body(carry)
            return new[:7] + (dist_fn(new[0]),)

    carry = lax.while_loop(cond_fn, body_fn, carry)
    q = carry[0]
    return q, f(q), carry[5], carry[6]


def make_fire_runner(f, df, dt, dtmax_factor=10.0, dist_fn=None, target_dist=-1.0):
    """Return run_chunk(carry, n) that advances FIRE by up to n steps.

    If dist_fn and target_dist are provided, each chunk also stops early
    (within the n steps) as soon as min_dist >= target_dist.
    carry[7] is then the live min_dist.
    """
    dtmax = dtmax_factor * dt
    dtmin = 0.02 * dt

    body = _fire_body(df, dtmax, dtmin)

    if dist_fn is None:
        def _fori_body(_, carry):
            return body(carry)

        @partial(jit, static_argnums=[1])
        def run_chunk(carry, n):
            return lax.fori_loop(0, n, _fori_body, carry)
    else:
        def body_fn(carry):
            new = body(carry)
            return new[:7] + (dist_fn(new[0]),)

        @jit
        def run_chunk(carry, n):
            step_start = carry[5]
            def cond_fn(c):
                _, _, _, _, _, step, _, min_dist = c
                return (step - step_start < n) & (min_dist < target_dist)
            return lax.while_loop(cond_fn, body_fn, carry)

    return run_chunk
