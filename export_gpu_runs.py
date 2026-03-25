import os
import shutil
from pathlib import Path

folders = [
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-2340_ENTANGLEONLY_GPU_AR1000_N1000_randomkeys928,634,38",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-2340_ENTANGLEONLY_GPU_AR1000_N1250_randomkeys478,490,159",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-2340_ENTANGLEONLY_GPU_AR1000_N1500_randomkeys32,40,130",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-2340_ENTANGLEONLY_GPU_AR1000_N2000_randomkeys83,359,744"
]

runs = []
# runs = [
#     "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-0354_SEQUENTIAL_GPU_N2000_randomkeys297,604,390",
#     "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-0354_SEQUENTIAL_GPU_N2000_randomkeys387,362,934",
#     "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-0354_SEQUENTIAL_GPU_N2000_randomkeys712,334,668",
#     "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-0354_SEQUENTIAL_GPU_N2000_randomkeys180,168,908",
#     "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-0354_SEQUENTIAL_GPU_N2000_randomkeys297,325,836",
#     "/n/home01/yjung/Github/entanglement-optimization/runs/20260227-0354_SEQUENTIAL_GPU_N1000_randomkeys267,339,352",
# ]

base_out_dir = Path("/n/home01/yjung/Github/entanglement-optimization/relaxation_4th_gpu")

all_run_dirs = folders + runs

for run in all_run_dirs:
    run_dir = Path(run)
    if not run_dir.exists():
        print(f"Directory {run_dir} does not exist.")
        continue

    # Extract N and random keys
    parts = run_dir.name.split('_')
    try:
        N_str = next(p for p in parts if p.startswith('N') and p[1:].isdigit())
        keys_str = next(p for p in parts if p.startswith('randomkeys'))[len('randomkeys'):]
    except StopIteration:
        print(f"Could not parse N or random keys from {run_dir.name}")
        continue

    out_folder = base_out_dir / N_str / keys_str
    out_folder.mkdir(parents=True, exist_ok=True)

    results_dir = run_dir / "results" / keys_str
    if not results_dir.exists():
        print(f"Results dir {results_dir} does not exist.")
        continue

    # Find the packing directories
    seq_subdirs = [d for d in results_dir.iterdir() if d.is_dir() and "SequentialRelaxedPacking" in d.name]
    entangle_subdirs = [d for d in results_dir.iterdir() if d.is_dir() and "EntangledPacking" in d.name]

    if seq_subdirs:
        packing_dir = seq_subdirs[0]
        # Find AR folders
        ar_folders = [d for d in packing_dir.iterdir() if d.is_dir() and d.name.startswith("AR")]
        
        for ar_folder in ar_folders:
            try:
                ar_val = int(ar_folder.name[2:])
            except ValueError:
                continue
                
            in_file = ar_folder / "x_relaxed.txt"
            if not in_file.exists():
                print(f"File {in_file} does not exist. Skipping.")
                continue
            
            out_file = out_folder / f"x_relaxed_AR{ar_val}.txt"
            rod_radius = 0.5 / ar_val
            
            with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
                f_out.write(f"# rod_radius = {rod_radius:g}\n")
                f_out.write(f_in.read())
                
            print(f"Wrote {out_file} (from sequential)")

    elif entangle_subdirs:
        packing_dir = entangle_subdirs[0]
        # parse AR from the folder name e.g. 2026-02-27_23_EntangledPacking-N1000-AR1000-Scale1
        parts = packing_dir.name.split('-')
        ar_val = None
        for p in parts:
            if p.startswith("AR"):
                try:
                    ar_val = int(p[2:])
                except ValueError:
                    pass
        
        if ar_val is None:
            print(f"Could not infer AR from {packing_dir.name}")
            continue
            
        in_file = packing_dir / "x_entangled.txt"
        if not in_file.exists():
            print(f"File {in_file} does not exist. Skipping.")
            continue
        
        # Write output as if it were relaxed for consistency
        out_file = out_folder / f"x_relaxed_AR{ar_val}.txt"
        rod_radius = 0.5 / ar_val
        
        with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
            f_out.write(f"# rod_radius = {rod_radius:g}\n")
            f_out.write(f_in.read())
            
        print(f"Wrote {out_file} (from entangled only)")

    else:
        print(f"No SequentialRelaxedPacking or EntangledPacking folder found in {results_dir}")
        continue

print("Done.")
