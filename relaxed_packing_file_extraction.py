import sys
from pathlib import Path
import re
import numpy as np
import shutil


if __name__ == "__main__":
    
    # must get an argument

    if len(sys.argv) < 2:
        print("put a folder path AND output path")
        raise ExceptionType()

    

    # folder path
    folder_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    print("base path: ", folder_path)
    print("output path: ", output_path)
    
    if not (folder_path.is_dir()):
        raise ValueError("put a legit folder")

    # find all x_relaxed.txt

    m = re.search("_N(\d+)_",str(folder_path))
    # print(m.group(0),m.group(1))

    N = int(m.group(1))

    # makedir in the base with N
    N_folder_path = output_path / f"N{N}"
    N_folder_path.mkdir(exist_ok=True)

    
    print("The size of packings:", N)

    for x in folder_path.rglob("x_entangled.txt"):
        # find random key pattern
        

        # m = re.search(r"/results/([\d,]+)",str(x))
        # print(m)
        
        random_key_string = x.parent.parent.stem
        # print(random_key_string)
        
        dta = np.loadtxt(x)
        if (N != dta.shape[0]):
            raise ValueError("N does not match with packing size.")
        # print(x.parent.stem, x.name)

        key_folder = N_folder_path / random_key_string
        key_folder.mkdir(exist_ok=True)

        # copy 
        shutil.copy(x,key_folder / x.name)        






