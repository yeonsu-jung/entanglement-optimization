
import os
import shutil
import glob

source_dirs = [
    "/n/home01/yjung/Github/entanglement-optimization/runs/20251227-0005_ENTANGLEONLY_AR100_N50_randomkeys38,546,617",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20251227-0005_ENTANGLEONLY_AR100_N50_randomkeys312,174,828",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20251227-0005_ENTANGLEONLY_AR100_N50_randomkeys396,760,797",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20251227-0005_ENTANGLEONLY_AR100_N50_randomkeys842,65,125",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20251227-0005_ENTANGLEONLY_AR100_N50_randomkeys886,661,536"
]

dest_base = "/n/home01/yjung/Github/entanglement-optimization/results/entangled_packings_2nd/N50"

if not os.path.exists(dest_base):
    os.makedirs(dest_base)
    print(f"Created base destination: {dest_base}")

for src_dir in source_dirs:
    # parsing keys from src_dir name
    # format: ..._randomkeys<KEY>
    dir_name = os.path.basename(src_dir)
    try:
        keys = dir_name.split("randomkeys")[1]
    except IndexError:
        print(f"Skipping {src_dir}: Could not parse keys")
        continue

    # Find x_entangled.txt
    # Pattern: src_dir/results/<keys>/*/x_entangled.txt or src_dir/results/*/x_entangled.txt
    # Based on exploration: runs/.../results/<keys>/<subfolder>/x_entangled.txt
    # Let's search recursively in src_dir/results for x_entangled.txt
    
    search_path = os.path.join(src_dir, "results", "**", "x_entangled.txt")
    found_files = glob.glob(search_path, recursive=True)
    
    if not found_files:
        print(f"Warning: x_entangled.txt not found in {src_dir}")
        continue
    
    # Use the first one found (simpler, assuming only one relevant one per run dir structure)
    src_file = found_files[0]
    
    dest_dir = os.path.join(dest_base, keys)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    dest_file = os.path.join(dest_dir, "x_entangled.txt")
    
    print(f"Copying {src_file} -> {dest_file}")
    shutil.copy2(src_file, dest_file)

print("Done.")
