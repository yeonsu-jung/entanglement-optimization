"""Smoke test for FIRE on a trivial convex objective."""
import jax.numpy as jnp

from entanglement_optimization.core.fire import optimize_fire


def test_fire_converges_on_quadratic():
    # f(q) = 0.5 * sum(q**2),  grad = q.  Minimum at q = 0.
    f  = lambda q: 0.5 * jnp.sum(q * q)
    df = lambda q: q

    q0 = jnp.array([1.0, -0.7, 0.3])
    q, fval, _step, err = optimize_fire(q0, f, df, Nmax=2000, atol=1e-7, dt=1e-2)

    assert float(err) < 1e-6
    assert float(fval) < 1e-12
    assert float(jnp.max(jnp.abs(q))) < 1e-6
