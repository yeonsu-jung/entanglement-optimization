"""Smoke tests for core/physics geometry primitives."""
import jax.numpy as jnp
import numpy as np
import pytest

from entanglement_optimization.core.physics import (
    all_pairwise_distances, create_pairs, dist_lin_seg, min_pairwise_distance,
    sph2cart, q_to_x,
)


def test_dist_lin_seg_parallel_offset():
    # Two unit segments along x, offset by 1 unit in y.
    p1s = jnp.array([0.0, 0.0, 0.0]); p1e = jnp.array([1.0, 0.0, 0.0])
    p2s = jnp.array([0.0, 1.0, 0.0]); p2e = jnp.array([1.0, 1.0, 0.0])
    d = float(dist_lin_seg(p1s, p1e, p2s, p2e))
    assert d == pytest.approx(1.0, abs=1e-6)


def test_dist_lin_seg_crossing_segments():
    # Two segments cross at the origin, so the shortest distance is zero.
    p1s = jnp.array([-1.0, 0.0, 0.0]); p1e = jnp.array([1.0, 0.0, 0.0])
    p2s = jnp.array([0.0, -1.0, 0.0]); p2e = jnp.array([0.0, 1.0, 0.0])
    d = float(dist_lin_seg(p1s, p1e, p2s, p2e))
    assert d == pytest.approx(0.0, abs=1e-6)


def test_dist_lin_seg_endpoint_gap():
    # The closest points are the first segment's right endpoint and the second
    # segment's lower endpoint, one unit apart along x.
    p1s = jnp.array([0.0, 0.0, 0.0]); p1e = jnp.array([1.0, 0.0, 0.0])
    p2s = jnp.array([2.0, 0.0, 0.0]); p2e = jnp.array([2.0, 1.0, 0.0])
    d = float(dist_lin_seg(p1s, p1e, p2s, p2e))
    assert d == pytest.approx(1.0, abs=1e-6)


def test_sph2cart_known_angles():
    v = np.asarray(sph2cart(jnp.pi / 2, 0.0))
    np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-7)
    v = np.asarray(sph2cart(jnp.pi / 2, jnp.pi / 2))
    np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-7)
    v = np.asarray(sph2cart(0.0, 0.0))
    np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-7)


def test_q_to_x_unit_rod():
    # One rod at origin, pointing along +x (theta=pi/2, phi=0), length 1.
    q = jnp.array([[0.0, 0.0, 0.0, jnp.pi / 2, 0.0]])
    x = np.asarray(q_to_x(q))
    assert x.shape == (1, 6)
    np.testing.assert_allclose(x[0, :3], [-0.5, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(x[0, 3:], [ 0.5, 0.0, 0.0], atol=1e-7)


def test_q_to_x_multiple_rods_preserves_centers_and_directions():
    q = jnp.array([
        [0.0, 0.0, 0.0, jnp.pi / 2, 0.0],
        [1.0, 2.0, 3.0, 0.0, 0.0],
    ])

    x = np.asarray(q_to_x(q))

    np.testing.assert_allclose(x[0], [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(x[1], [1.0, 2.0, 2.5, 1.0, 2.0, 3.5], atol=1e-7)


def test_pairwise_distance_helpers_match_expected_values():
    q = jnp.array([
        [0.0, 0.0, 0.0, jnp.pi / 2, 0.0],
        [0.0, 1.0, 0.0, jnp.pi / 2, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])

    pairs = create_pairs(q)
    distances = np.asarray(all_pairwise_distances(pairs))

    assert pairs.shape == (3, 10)
    np.testing.assert_allclose(distances, [1.0, 0.5, np.sqrt(1.25)], atol=1e-6)
    assert float(min_pairwise_distance(q)) == pytest.approx(0.5, abs=1e-6)
