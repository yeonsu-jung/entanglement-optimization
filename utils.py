from numpy import loadtxt, savetxt
import numpy as np
from jax import numpy as jnp
from datetime import datetime
import os
import glob
import shutil

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

def archiving(folder_name=None):
    # dt string in YYYY-MM-DD_HH-MM-SS
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    if folder_name is None:
        folder_name = f"/Users/yeonsu/Data/cache/{dt_string}"
    else:
        folder_name = f"{folder_name}/{dt_string}"
    # hash = hashlib.md5(dt_string.encode()).hexdigest()
    
    os.makedirs(folder_name, exist_ok=False)
    
    source_dir = './'    
    # copy every py file to the folder
    files = glob.iglob(os.path.join(source_dir, "*.py"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, folder_name)
    
    return dt_string, folder_name
    
if __name__ == '__main__':
    
    tmp = np.array([1,2,3,4,5,6])
    print(tmp[:3])
    
    print(tmp[-3:])
    
    
    pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_21-04-2024_15-35-59.txt'
    
    # xyzform(pth)
    
    
    
    