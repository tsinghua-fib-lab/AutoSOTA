import jax
import jax.numpy as jnp

@jax.jit
def my_function(matrix, vector):
    # Example computation
    return jax.lax.exp(matrix @ vector + 1).sum()

# 1. Define the shapes and dtypes of your inputs (no memory allocated)
matrix_shape = jax.ShapeDtypeStruct((1000, 1000), 'float32')
vector_shape = jax.ShapeDtypeStruct((1000,), 'float32')

# 2. Lower the jitted function to XLA HLO (High-Level Optimizer)
# 1. Lower the function
lowered = my_function.lower(matrix_shape, vector_shape)

# 2. Add .compile() here
compiled = lowered.compile()

# 3. Get the cost analysis
cost = compiled.cost_analysis()

# Handle the fact that compiled cost_analysis might return a list of dicts
if isinstance(cost, list):
    cost = cost[0]

print(f"Total FLOPs: {cost.get('flops', 'N/A')}")