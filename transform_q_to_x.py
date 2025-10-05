import numpy as np
from transforms import q_to_x
from pathlib import Path

pth = Path('/Users/yeonsu/GitHub/entanglement-optimization/results/91,22,12/2024-12-07_23_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt')
parent_folder = pth.parent

q_relaxed = np.loadtxt(pth)
x_relaxed = q_to_x(q_relaxed)
np.savetxt(f'{parent_folder}/x_relaxed.txt',x_relaxed)