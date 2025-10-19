# single_tentacle_jitter_fast.py
import jax
import jax.numpy as jnp
from jax import jit, vmap, config, lax
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
import polyscope as ps
import sys

# (optional) project-local util that makes output dirs; keep if you already have it
sys.path.append('core')
try:
    from utils import setup_directories
    output_dir, movie_dir = setup_directories(__file__)
    print(f"✅ Outputting to: {output_dir}")
except Exception:
    output_dir = "."
    movie_dir = "./frames"

config.update("jax_enable_x64", True)

# =========================
# Global config
# =========================
TIME_HORIZON = 10001
SOFTMIN_BETA = 60.0
MAX_HALVINGS = 80
SAVE_EVERY   = 30           # capture a frame every N steps
TIME_STEP = 5e-5

# jitter (annealed) to encourage out-of-plane motion
def jitter_schedule(it, iters, sigma0=5e-3, sigma1=5e-4, warmup_frac=0.25):
    t = jnp.clip(it / (iters * warmup_frac), 0.0, 1.0)
    # cosine decay sigma0 -> sigma1 over warmup
    return sigma1 + 0.5 * (sigma0 - sigma1) * (1.0 + jnp.cos(jnp.pi * t))

def fixbound(num): return jnp.clip(num, 0.0, 1.0)

# =========================
# Segment-segment distance
# =========================
def dist_lin_seg_flat(points):
    d1 = points[3:6]  - points[0:3]
    d2 = points[9:12] - points[6:9]
    d12= points[6:9]  - points[0:3]

    D1 = jnp.dot(d1, d1); D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12); S2 = jnp.dot(d2, d12)
    R  = jnp.dot(d1, d2)
    den = D1 * D2 - R**2

    def case1():
        (t,u) = lax.cond(D1 != 0.,
            lambda _: (fixbound(S1/D1), 0.),
            lambda _: lax.cond(D2 != 0., lambda _: (0., fixbound(-S2/D2)), lambda _: (0.,0.), None),
            None)
        return (t,u)

    def case2_1():
        t = 0.; u = -S2/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2_2():
        t = fixbound((S1*D2 - S2*R)/den)
        u = (t*R - S2)/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2():
        return lax.cond(den == 0., lambda _: case2_1(), lambda _: case2_2(), None)

    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.), lambda _: case1(), lambda _: case2(), None)
    return jnp.linalg.norm(d1 * t - d2 * u - d12)

@jit
def dist_lin_seg(p1s, p1e, p2s, p2e):
    d1 = p1e - p1s
    d2 = p2e - p2s
    d12= p2s - p1s

    D1 = jnp.dot(d1, d1); D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12); S2 = jnp.dot(d2, d12)
    R  = jnp.dot(d1, d2)
    den = D1 * D2 - R**2

    def case1():
        (t,u) = lax.cond(D1 != 0.,
            lambda _: (fixbound(S1/D1), 0.),
            lambda _: lax.cond(D2 != 0., lambda _: (0., fixbound(-S2/D2)), lambda _: (0.,0.), None),
            None)
        return (t,u)

    def case2_1():
        t = 0.; u = -S2/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2_2():
        t = fixbound((S1*D2 - S2*R)/den)
        u = (t*R - S2)/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2():
        return lax.cond(den == 0., lambda _: case2_1(), lambda _: case2_2(), None)

    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.), lambda _: case1(), lambda _: case2(), None)
    return jnp.linalg.norm(d1 * t - d2 * u - d12)

@jit
def dist_lin_seg_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

def min_distance_between_polylines(A0, A1, B0, B1):
    D = jax.vmap(lambda a0,a1: jax.vmap(lambda b0,b1: dist_lin_seg(a0,a1,b0,b1))(B0,B1))(A0,A1)
    return jnp.min(D)

# =========================
# Curvature-only kinematics
# =========================
@jit
def rotate(u_i, kappa):
    k2 = jnp.dot(kappa, kappa)
    num = (4 - k2) * u_i + 4 * jnp.cross(kappa, u_i) + 2 * kappa * jnp.dot(kappa, u_i)
    return num / (4 + k2 + 1e-18)

@jit
def reconstruct_curve_from_curvature_and_length(first_point, first_edge, curvatures, reference_lengths):
    num_segments = len(curvatures) + 1
    nodes = jnp.zeros((num_segments + 1, 3))
    nodes = nodes.at[0].set(first_point)
    tangents = jnp.zeros((num_segments, 3))
    tangents = tangents.at[0].set(first_edge / (jnp.linalg.norm(first_edge) + 1e-18))

    def tangent_body(i, tangents):
        next_tangent = rotate(tangents[i], curvatures[i])
        return tangents.at[i+1].set(next_tangent)

    tangents = jax.lax.fori_loop(0, num_segments - 1, tangent_body, tangents)

    def node_body(i, nodes):
        return nodes.at[i+1].set(nodes[i] + reference_lengths[i] * tangents[i])

    nodes = jax.lax.fori_loop(0, num_segments, node_body, nodes)
    return nodes, tangents

def get_edge_lengths(P):
    return jnp.linalg.norm(P[1:] - P[:-1], axis=-1)

def get_der_curvature(x):
    t = x[1:] - x[:-1]
    t = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-18)
    return 2 * jnp.cross(t[:-1], t[1:]) / (1 + jnp.sum(t[:-1] * t[1:], axis=-1, keepdims=True))

# =========================
# Linking (your formula)
# =========================
def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
    r_ij   = p_i  - p_j
    r_ijj  = p_i  - p_jj
    r_iij  = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij,  r_ijj);  n1 = n1 / (jnp.linalg.norm(n1)  + tol)
    n2 = jnp.cross(r_ijj, r_iijj); n2 = n2 / (jnp.linalg.norm(n2) + tol)
    n3 = jnp.cross(r_iijj, r_iij); n3 = n3 / (jnp.linalg.norm(n3) + tol)
    n4 = jnp.cross(r_iij,  r_ij);  n4 = n4 / (jnp.linalg.norm(n4)  + tol)

    tol = 0.0
    return -1.0/(4.0*jnp.pi) * jnp.abs(
        jnp.arcsin(jnp.clip(jnp.dot(n1,n2), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n2,n3), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n3,n4), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n4,n1), -1.0+tol, 1.0-tol))
    )

def _segments(P): return P[:-1], P[1:]

def _safe_seg_contrib(p_i, p_ii, p_j, p_jj, eps=1e-12):
    ok_i = jnp.linalg.norm(p_ii - p_i)  > eps
    ok_j = jnp.linalg.norm(p_jj - p_j)  > eps
    def _do(): return compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj)
    return lax.cond(ok_i & ok_j, _do, lambda: 0.0)

_pair_vmap = vmap(vmap(_safe_seg_contrib, in_axes=(None, None, 0, 0)), in_axes=(0, 0, None, None))

def total_linking_number_polyline(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    A0, A1 = _segments(A)
    B0, B1 = _segments(B)
    contribs = _pair_vmap(A0, A1, B0, B1)
    contribs = jnp.nan_to_num(contribs, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.sum(jnp.abs(contribs))

# =========================
# Soft-min + projection
# =========================
@jit
def softmin_pairwise_distance(A0, A1, B0, B1, beta):
    pair = jax.vmap(lambda a0,a1: jax.vmap(lambda b0,b1: dist_lin_seg(a0,a1,b0,b1))(B0,B1))(A0,A1)
    m = jnp.min(pair)
    return -(1.0/beta) * (jnp.log(jnp.sum(jnp.exp(-beta*(pair - m)))) - (-beta*m))

@jit
def softmin_post_distance_from_kappa(kappa, first_point, first_edge, lengths, post, beta):
    q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, lengths)
    return softmin_pairwise_distance(q[:-1], q[1:], post[:-1], post[1:], beta)

def project_update_against_constraint(update_pytree, gradc_pytree, eps=1e-12):
    u_flat, unflatten = ravel_pytree(update_pytree)
    g_flat, _         = ravel_pytree(gradc_pytree)
    dot   = jnp.vdot(u_flat, g_flat)
    denom = jnp.vdot(g_flat, g_flat) + eps
    u_proj_flat = u_flat - (dot/denom) * g_flat
    return unflatten(u_proj_flat)

# =========================
# Trainer (fully JIT)
# =========================
def make_trainer(first_point, first_edge, edge_lengths, post,
                 d_self_min, d_post_min,
                 i_indices, j_indices,
                 lr=2e-4, momentum=0.9, nesterov=True,
                 w_kappa=1e-3, beta_softmin=SOFTMIN_BETA,
                 max_halvings=MAX_HALVINGS,
                 time_horizon=TIME_HORIZON,
                 save_every=SAVE_EVERY):

    def loss_and_aux(kappa):
        q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, edge_lengths)
        lk  = total_linking_number_polyline(q, post)
        reg = w_kappa * jnp.sum(kappa * kappa)
        return (-(lk) + reg), (lk, reg, q)

    loss_grad = jax.value_and_grad(lambda k: loss_and_aux(k), has_aux=True)
    opt = optax.sgd(learning_rate=lr, momentum=momentum, nesterov=nesterov)

    @jit
    def min_self_distance_q(q):
        r1, r2 = q[:-1], q[1:]
        pdist = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
        return jnp.min(pdist)

    @jit
    def min_post_distance_q(q):
        A0, A1 = q[:-1], q[1:]
        B0, B1 = post[:-1], post[1:]
        return min_distance_between_polylines(A0, A1, B0, B1)

    def grad_c_post(kappa):
        return jax.grad(lambda kk: d_post_min - softmin_post_distance_from_kappa(
            kk, first_point, first_edge, edge_lengths, post, beta_softmin))(kappa)

    def init_opt(kappa0):
        return opt.init(kappa0)

    # frame buffer (device)
    num_frames = (time_horizon + save_every - 1) // save_every + 1
    frame_shape = (num_frames, edge_lengths.shape[0] + 1, 3)

    @jit
    def step(carry, it):
        kappa, opt_state, key, frames, fidx = carry

        # base grad/update
        ((L, (lk, reg, q)), g) = loss_grad(kappa)
        updates, opt_state = opt.update(g, opt_state, params=kappa)

        # annealed jitter in curvature space
        key, key_noise = jax.random.split(key)
        sigma = jitter_schedule(it, time_horizon, sigma0=5e-5, sigma1=5e-6, warmup_frac=0.25)
        noise = sigma * jax.random.normal(key_noise, shape=updates.shape)

        # project update/noise against post constraint when active
        d_post_soft = softmin_post_distance_from_kappa(kappa, first_point, first_edge, edge_lengths, post, beta_softmin)
        post_active = d_post_soft <= (d_post_min + 1e-4)
        def do_proj(u): return project_update_against_constraint(u, grad_c_post(kappa))
        updates = lax.cond(post_active, do_proj, lambda u: u, updates)
        noise   = lax.cond(post_active, do_proj, lambda u: u, noise)

        updates = updates + noise

        # feasible scaling
        def body_fn(state):
            scale, accepted, best_kappa = state
            trial = optax.apply_updates(kappa, scale * updates)
            q_trial, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, trial, edge_lengths)
            dself = min_self_distance_q(q_trial)
            dpost = min_post_distance_q(q_trial)
            (Ltrial, _), _ = loss_grad(trial)  # loss only
            feas = (dself >= d_self_min) & (dpost >= d_post_min) & jnp.isfinite(Ltrial)
            return (jnp.where(feas, scale, scale * 0.5),
                    accepted | feas,
                    jnp.where(feas, trial, best_kappa))

        def cond_fn(state):
            scale, accepted, _ = state
            return (~accepted) & (scale > (2.0 ** (-max_halvings)))

        init_state = (jnp.array(1.0), jnp.array(False), kappa)
        scale, accepted, kappa_next = lax.while_loop(cond_fn, body_fn, init_state)
        kappa_next = lax.select(accepted, kappa_next, kappa)

        # save frame
        save_flag = (it % save_every == 0)
        q_next, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa_next, edge_lengths)
        frames = lax.cond(save_flag, lambda arr: arr.at[fidx].set(q_next), lambda arr: arr, frames)
        fidx = lax.cond(save_flag, lambda i: i+1, lambda i: i, fidx)

        # logs
        dself_next = min_self_distance_q(q_next)
        dpost_next = min_post_distance_q(q_next)
        (L_next, (lk_next, reg_next, _)), _ = loss_grad(kappa_next)
        log = (L_next, lk_next, reg_next, dself_next, dpost_next, scale, accepted, sigma)

        return (kappa_next, opt_state, key, frames, fidx), log

    return step, init_opt, num_frames, frame_shape

# =========================
# Main
# =========================
if __name__ == "__main__":
    N = 300

    # post: single segment along z
    post = jnp.array([[0.0, 0.0, -10.0],
                      [0.0, 0.0, 10.0]])

    # tentacle: vertical-ish line offset in x,y
    tentacle = jnp.array(np.linspace([0.5, -0.5, 0.5], [0.5, 9.5, 0.5], N))
    natural_curvature = get_der_curvature(tentacle)
    edge_lengths = get_edge_lengths(tentacle)
    first_point = tentacle[0]
    first_edge  = tentacle[1] - tentacle[0]

    # init curvature (tiny noise)
    rng = jax.random.PRNGKey(0)
    initial_curvature = natural_curvature + 0.01 * jax.random.normal(rng, natural_curvature.shape)

    # hard gaps
    total_length = jnp.sum(edge_lengths)
    d_self_min = total_length / 100.0   # skip-neighbor self gap
    d_post_min = 0.2                    # tentacle–post gap

    # pairs for self-collision (skip near neighbors: k=30)
    i_indices, j_indices = jnp.triu_indices(N - 1, k=30)

    # build trainer
    step_fn, opt_init, num_frames, frame_shape = make_trainer(
        first_point, first_edge, edge_lengths, post,
        d_self_min, d_post_min,
        i_indices, j_indices,
        lr=TIME_STEP, momentum=0.9, nesterov=True,
        w_kappa=1e-20, beta_softmin=SOFTMIN_BETA,
        max_halvings=MAX_HALVINGS,
        time_horizon=TIME_HORIZON,
        save_every=SAVE_EVERY
    )

    # initial buffers
    opt_state0 = opt_init(initial_curvature)
    key0 = jax.random.PRNGKey(42)
    frames0 = jnp.zeros(frame_shape, dtype=jnp.float64)
    q_init, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, initial_curvature, edge_lengths)
    frames0 = frames0.at[0].set(q_init)
    fidx0 = jnp.array(1, dtype=jnp.int32)

    @jit
    def run_optim(kappa0, opt_state0, key0, frames0, fidx0):
        def body(carry, it):
            return step_fn(carry, it)
        carry0 = (kappa0, opt_state0, key0, frames0, fidx0)
        (kappa_f, opt_state_f, key_f, frames_f, fidx_f), logs = lax.scan(body, carry0, jnp.arange(TIME_HORIZON))
        return kappa_f, frames_f, fidx_f, logs

    final_kappa, frames_all, frames_count, logs = run_optim(initial_curvature, opt_state0, key0, frames0, fidx0)

    # host logs
    Ls, LKs, Regs, Dselfs, Dposts, Scales, Accs, Sigmas = [np.array(x) for x in logs]
    for it in range(0, TIME_HORIZON, SAVE_EVERY):
        print(f"it {it:5d} | acc {bool(Accs[it])} | loss {Ls[it]: .6e} | "
              f"LK {LKs[it]: .6e} | d_self {Dselfs[it]:.4f} | d_post {Dposts[it]:.4f} | "
              f"scale {Scales[it]:.3f} | jitter {Sigmas[it]:.2e}")

    # reconstruct final and report distances
    rec_final, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, final_kappa, edge_lengths)
    r1, r2 = rec_final[:-1], rec_final[1:]
    pdist_self = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
    A0, A1 = rec_final[:-1], rec_final[1:]
    B0, B1 = post[:-1], post[1:]
    min_d_post = min_distance_between_polylines(A0, A1, B0, B1)
    print(f"Final min self distance (skip-neighbor pairs): {float(jnp.min(pdist_self)):.4e}")
    print(f"Final min tentacle–post distance: {float(min_d_post):.4e}")

    # ========= Polyscope export of frames =========
    frames_np = np.array(frames_all[:int(frames_count)])  # (F,N,3)

    ps.init()
    ps.set_up_dir("x_up")
    ps.set_ground_plane_mode("none")
    edges = jnp.stack([jnp.arange(N - 1), jnp.arange(1, N)], axis=-1)

    ps_curve = ps.register_curve_network("tentacle", frames_np[0], edges)
    ps_curve.set_radius((float(d_self_min) * 0.5), relative=False)
    ps_curve.set_color((0.8, 0.2, 0.2)); ps_curve.set_material("clay")

    ps_rod = ps.register_curve_network("post", post, np.array([[0,1]]))
    ps_rod.set_radius(float(d_post_min)*0.5, relative=False)
    ps_rod.set_color((0.1,0.1,0.1))

    # write PNG sequence for movie
    for k in range(frames_np.shape[0]):
        ps_curve.update_node_positions(frames_np[k])
        ps.screenshot(f"{movie_dir}/frame_{k:04d}.png", transparent_bg=True)

    print(f"wrote {frames_np.shape[0]} frames to {movie_dir}/frame_####.png")
    print("tip: ffmpeg -y -framerate 30 -i {}/frame_%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p tentacle_wrap.mp4".format(movie_dir))
