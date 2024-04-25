from numpy import loadtxt, savetxt
import numpy as np
from jax import numpy as jnp

def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Function {func.__name__} took {time.time()-start:.2f} seconds')
        return result
    return wrapper

def parse_id_string(filename):
    filepart = filename.split('/')[-1]
    # remove .txt extension, and join the rest
    filepart = '.'.join(filepart.split('.')[:-1])    
    return filepart.split('_')
    
if __name__ == '__main__':
    
    tmp = np.array([1,2,3,4,5,6])
    print(tmp[:3])
    
    print(tmp[-3:])
    
    
    pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_21-04-2024_15-35-59.txt'
    
    # xyzform(pth)
    
    
    
    