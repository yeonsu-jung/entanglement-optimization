
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Set JAX to 64-bit just in case, though transforms.py also does it
jax.config.update("jax_enable_x64", True)

# Add current directory to sys.path to ensure we can import 'core'
sys.path.append(os.getcwd())

try:
    from core.transforms import q_to_x
except ImportError as e:
    print(f"Error importing core.transforms: {e}")
    sys.exit(1)

def main():
    input_path = "results/q_entangled_N200_36,298,312.npy"
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading {input_path}...")
    try:
        q_data = np.load(input_path)
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        sys.exit(1)
        
    print(f"Loaded data shape: {q_data.shape}")

    # Ensure data is in the expected flat format or (N, 5) for q_to_x
    # q_to_x handles reshape internally: q.reshape((-1, 5))
    
    print("Transforming q to x using core.transforms.q_to_x...")
    print(q_data.shape)
    try:
        # Convert to JAX array for the transform
        q_jax = jnp.array(q_data)
        x_jax = q_to_x(q_jax)
        
        # Convert back to numpy for saving
        x_data = np.array(x_jax)
    except Exception as e:
        print(f"Error during transformation: {e}")
        sys.exit(1)

    print(f"Transformed data shape: {x_data.shape}")

    output_path = "results/x_entangled_N200_36,298,312.txt"
    print(f"Saving to {output_path}...")
    try:
        np.savetxt(output_path, x_data)
        print("Done.")
    except Exception as e:
        print(f"Error saving text file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
