import numpy as np
import sys

sys.path.append('../core')
qq = np.load('/Users/yeonsu/Downloads/q_history_AR0500_N200.npy')

from visualizations import create_movie_from_q_history

import os
os.makedirs('temp2', exist_ok=False)

create_movie_from_q_history(qq, 'temp2', 200, 1/500)