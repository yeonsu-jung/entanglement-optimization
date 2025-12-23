from jax import grad, jit
import jax.numpy as jnp
from potentials import all_pairwise_distances, create_pairs
from optimization import optimize_fire_nonjax, optimize_fire_nonjax_individual, optimize_fire_jax_individual

def collision_relaxation(q,f_in,params,N_outer,Nmax,atol,dt,atol_min=1,visualize=False,callback=None):    

    num_rods = q.shape[0]//5
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    from potentials import dist_lin_seg_over_ij
    from transforms import q_to_x
    
    for k in range(N_outer):
        col_rad_0 = params["col_rad"]
        params["col_rad"] = params["col_rad"]*(1)
        f = lambda q: f_in(q,params)
        df = jit(grad(jit(f)))
        # q, f_val, num_iterations, error = optimize_fire_nonjax(q, f, df, Nmax, atol, dt, False)
        q, f_val, _, error = optimize_fire_nonjax_individual(q, f, df, Nmax, atol, dt, callback=callback)
        # q, f_val, _, error = optimize_fire_jax_individual(q, f, df, Nmax, atol, dt, callback=callback)
        
        # if (error < atol_min):
        #     print(f"Error is smaller than atol_min: {error}")
        #     break
        
        q_pairs = create_pairs(q.reshape(-1,5))
        distances = all_pairwise_distances(q_pairs)
        # dt = dt/1.1     # TO DO: factor out this numbers
        # if jnp.abs(jnp.min(distances) - col_rad_0) < col_rad_0*1e-6:

        # if k % 100 == 0:
        #     x = q_to_x(q)
        #     r1 = x[:,0:3]
        #     r2 = x[:,3:6]
        #     dist_mat = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
        #     print(f"Min distance between rods: {jnp.min(dist_mat)}")

        if (jnp.min(distances) - col_rad_0*2) > 0:
            print(f"Enough pushoff: {jnp.min(distances)}")           
            break
    
    return q

# %%
if __name__ == "__main__":
    from potentials import total_harmonic_line
    from transforms import q_to_x
    from visualizations import plot_many_rods
    import matplotlib.pyplot as plt

    # create initial configuration
    num_rods = 10
    Lbox = 10
    col_rad = 0.1
    rod_length = 1.0
    params = {"col_rad": col_rad, "rod_length": rod_length, "Lbox": Lbox}

    key = jnp.array([0,0])
    # from initial_states import create_initial_configuration_random
    from protocols import create_intersecting_rods,create_entangled_rods

    from potentials import total_effective_potential
    random_keys = [6,7,8]
    q = create_entangled_rods(num_rods, total_effective_potential, random_keys, rod_diameter = col_rad*2, Nmax = 3000)

    # relax collisions
    N_outer = 10
    Nmax = 5000
    atol = 1e-6
    dt = 0.01

    def callback(qk, fk, k, f_val, error):
        if k % 100 == 0:
            print(f"Iteration {k}, f_val: {f_val}, error: {error}")
            xk = q_to_x(qk)
            plot_many_rods(qk)
            plt.show()

    qq = []
    def callback_collect(qk, fk, k, f_val, error):
        if k % 10 == 0:
            qq.append(qk)

    params["col_rad"] = col_rad
    params["amp"] = 1.0
    q_relaxed = collision_relaxation(q, total_harmonic_line, params, N_outer, Nmax, atol, dt, atol_min=atol*10, visualize=False)
    
    # check final distances
    from potentials import dist_lin_seg_over_ij
    x_relaxed = q_to_x(q_relaxed)
    r1 = x_relaxed[:,0:3]
    r2 = x_relaxed[:,3:6]
    num_rods = q_relaxed.shape[0]//5
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    dist_mat = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
    print(f"Min distance between rods after relaxation: {jnp.min(dist_mat)}")

    # plot final configuration
    plot_many_rods(q_relaxed)
    plt.show()
# %%
