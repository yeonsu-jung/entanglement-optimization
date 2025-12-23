import sys
from pathlib import Path
import numpy as np

sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')
    
output_dir = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/scripts/outputs'

from protocols import create_nonintersecting_random_rods, create_nonintersecting_random_rods_contained
from matplotlib import pyplot as plt
from transforms import q_to_x

num_rods = 200
rod_length = 1
ARs = [10,25,50,100,150,200,300,500,1000]

container_size = 1.5

for AR in ARs:
    rod_diameter = rod_length/AR
    # q = create_nonintersecting_random_rods(num_rods,rod_diameter,max_attempts=10000)
    # q = create_nonintersecting_random_rods(num_rods,rod_diameter,max_attempts=10000)
    q = create_nonintersecting_random_rods_contained(num_rods,rod_diameter,container_size,max_attempts=10000)
    x = q_to_x(q)
    np.savetxt(f'/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/python/results/x_nonintersecting_packing_{AR}.txt', x)

    # plt.figure()
    # plt.subplot(111, projection='3d')
    # for i in range(num_rods):
    #     start = x[i,:3]
    #     end = x[i,3:]
    #     plt.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-')
    # plt.show()

    # export x to txt



