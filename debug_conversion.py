
import numpy as np
import os
import sys

# Add path to import q_to_x manually to see what happens
sys.path.append(os.getcwd())
try:
    from core.transforms import q_to_x, sph2cart
    import jax.numpy as jnp
except ImportError:
    print("Could not import core.transforms")

f_npy = "results/q_entangled_N200_36,298,312.npy"
f_txt = "results/x_entangled_N200_36,298,312.txt"

print(f"--- {f_npy} ---")
if os.path.exists(f_npy):
    q = np.load(f_npy)
    print(f"Shape: {q.shape}")
    print(f"First rod (q): {q[:5]}")
    
    # Manual check
    x0 = q[0:3]
    polar = q[3]
    azim = q[4]
    print(f"Polar (idx 3): {polar}, Azim (idx 4): {azim}")
    
    # transform check
    x_jax = q_to_x(jnp.array(q[:5]))
    print(f"First rod (computed x): {x_jax}")
    
else:
    print("File not found.")

print(f"\n--- {f_txt} ---")
if os.path.exists(f_txt):
    try:
        x = np.loadtxt(f_txt)
        print(f"Shape: {x.shape}")
        print(f"First rod (txt): {x[0]}")
    except Exception as e:
        print(f"Error reading txt: {e}")
        # print raw
        with open(f_txt, 'r') as f:
            print("Raw first line:", f.readline())
else:
    print("File not found.")
