
import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd())

from core.nudge_by_optimization_rods import create_random_rods

def main():
    num_rods = 10
    random_keys = [0, 1, 2] # It expects a list of keys or seeds?
    # In nudge_by_optimization_rods.py it does:
    # key = random.key(random_keys[0])
    # So it seems to use jax.random.key which might mean it expects integers if using old jax or newer jax conventions.
    # Let's check core/nudge_by_optimization_rods.py imports to be sure about 'random'.
    
    # Actually I should check what 'random' is in that file.
    # But usually integers work for seeds.
    
    try:
        q = create_random_rods(num_rods, random_keys)
        print(f"create_random_rods shape: {q.shape}")
        if len(q.shape) == 1 and q.shape[0] == num_rods * 5:
             print("It is flattened (N*5,)")
        elif len(q.shape) == 2 and q.shape == (num_rods, 5):
             print("It is (N, 5)")
        else:
             print("Unknown shape")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
