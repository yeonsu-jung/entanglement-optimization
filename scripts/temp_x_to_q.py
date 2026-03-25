import os
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.append(os.getcwd())
from transforms import x_to_q

def main():
    if len(sys.argv) != 3:
        print("Usage: python {} <input_x.txt> <output_q.npy>".format(sys.argv[0]))
        sys.exit(1)
        
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    print(f"Loading {in_path}...")
    x = np.loadtxt(in_path)
    
    print(f"Transforming x to q...")
    q = x_to_q(jnp.array(x))
    
    q_np = np.array(q)
    print(f"Saving to {out_path}...")
    np.save(out_path, q_np)
    np.savetxt(out_path.replace(".npy", ".txt"), q_np)
    print("Done.")

if __name__ == "__main__":
    main()
