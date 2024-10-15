# %%
import jax
import jax.numpy as jnp
import optax

from matplotlib import pyplot as plt
from protocols import create_random_rods
from visualizations import plot_many_rods
from potentials import total_effective_potential, total_harmonic_line, total_harmonic_line_relax
from potentials import create_pairs, all_pairwise_distances
# %%
q0 = create_random_rods(500)
objective = total_effective_potential

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(q0)

# Gradient of the objective function
@jax.jit  # Just-in-time compilation for speed
def step(q0, opt_state):
    grads = jax.grad(objective)(q0)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_q0 = optax.apply_updates(q0, updates)

    return new_q0, opt_state

# Optimization loop
for i in range(1000):
    q0, opt_state = step(q0, opt_state)
    if i % 100 == 0:
        print(f"Iteration {i}: Objective = {objective(q0)}")

q_pairs = create_pairs(jnp.reshape(q0,(-1,5)))
d = all_pairwise_distances(q_pairs)
print(jnp.min(d))

# %%
plot_many_rods(q0.reshape(-1,5))
plt.show()
# %%
q_entangled = q0
# %%
# q0 = q_entangled.copy()
# %%
params = {"col_rad": 1/100, "amp": 100}
relaxation_objective = lambda q: total_harmonic_line(q,params)
# relaxation_objective = lambda q: total_harmonic_line(q,params) + 0.0000001*total_effective_potential(q)

def relaxation_step(q0,opt_state):
    grads = jax.grad(relaxation_objective)(q0)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_q0 = optax.apply_updates(q0, updates)
    return new_q0, opt_state

# Optimization loop
qq = []
for i in range(10000):
    q0, opt_state = relaxation_step(q0, opt_state)
    if i % 100 == 0:
        q_pairs = create_pairs(jnp.reshape(q0,(-1,5)))
        d = all_pairwise_distances(q_pairs)
        print(f"Iteration {i}: Objective = {objective(q0)}, Min distance = {jnp.min(d)}")
        qq.append(q0)

print("Optimized parameters:", q0)


plot_many_rods(q0.reshape(-1,5))
plt.show()

q_pairs = create_pairs(jnp.reshape(q0,(-1,5)))
d = all_pairwise_distances(q_pairs)

print(jnp.min(d))
print(1/100*2)
# %%
total_effective_potential(q0)
# %%
import numpy as np
qq = np.array(qq)
np.save('qq.npy',qq)