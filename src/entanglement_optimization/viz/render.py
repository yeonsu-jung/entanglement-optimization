import argparse
import numpy as np
from jax import numpy as jnp
import polyscope as ps

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="simple distance check."
    )
    p.add_argument("filepath", type=str, help="trajectory data path")
    p.add_argument("--out-dir", type=str, help="output png parent dir")
    return p.parse_args()


args = parse_args()

# get AR from file name
from pathlib import Path
# AR = float(str(Path(args.filepath).parent).split('AR')[0])
# AR = float(str(Path(args.filepath).parent).split('AR')[0])


try:
    AR = float(Path(args.filepath).parent.stem.split('AR')[1])
except:
    AR = 100
    
rod_radius = 1./AR/2.

# rod_radius = np.max([0.01,rod_radius]) # too small diameter can't be seen


x = np.load(args.filepath)

ps.init()

num_rods = x.shape[0]
nodes = x.reshape(-1,3)
edges = np.array([[2*i, 2*i+1] for i in range(num_rods)])

ps_curve = ps.register_curve_network("packing",nodes,edges)
ps_curve.set_radius(rod_radius,relative=False)

# ps.show()
ps.screenshot(f'{args.out_dir}/rendered.png')