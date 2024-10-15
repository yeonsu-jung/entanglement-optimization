import jax
import jax.numpy as jnp
from jax import grad
from jax.experimental.optimizers import adam

# Objective function (example with a flat fixed point)
def objective(x):
    return jnp.sum(jnp.tanh(x)**2)  # Tanh gives very flat regions around 0

# Gradient of the objective
grad_objective = grad(objective)

# Initialize optimizer
init_fun, update_fun, get_params = adam(step_size=0.01)
opt_state = init_fun(jnp.zeros(10))  # Start with initial guess

# Optimization loop
for i in range(1000):
    params = get_params(opt_state)
    grads = grad_objective(params)
    opt_state = update_fun(i, grads, opt_state)

    if i % 100 == 0:
        print(f"Iteration {i}: Objective = {objective(params)}")

print("Optimized parameters:", get_params(opt_state))