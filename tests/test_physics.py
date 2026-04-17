"""Smoke tests for core/physics geometry primitives."""
import jax.numpy as jnp
import numpy as np
import pytest

from entanglement_optimization.core.physics import (
    dist_lin_seg, sph2cart, q_to_x,
)


def test_dist_lin_seg_parallel_offset():
    # Two unit segments along x, offset by 1 unit in y.
    p1s = jnp.array([0.0, 0.0, 0.0]); p1e = jnp.array([1.0, 0.0, 0.0])
    p2s = jnp.array([0.0, 1.0, 0.0]); p2e = jnp.array([1.0, 1.0, 0.0])
    d = float(dist_lin_seg(p1s, p1e, p2s, p2e))
    assert d == pytest.approx(1.0, abs=1e-6)


def test_sph2cart_known_angles():
    v = np.asarray(sph2cart(jnp.pi / 2, 0.0))
    np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-7)
    v = np.asarray(sph2cart(0.0, 0.0))
    np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-7)


def test_q_to_x_unit_rod():
    # One rod at origin, pointing along +x (theta=pi/2, phi=0), length 1.
    q = jnp.array([[0.0, 0.0, 0.0, jnp.pi / 2, 0.0]])
    x = np.asarray(q_to_x(q))
    assert x.shape == (1, 6)
    np.testing.assert_allclose(x[0, :3], [-0.5, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(x[0, 3:], [ 0.5, 0.0, 0.0], atol=1e-7)
