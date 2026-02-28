import os
import shutil
from pathlib import Path

runs = [
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260228-0046_ENTANGLEONLY_GPU_AR1000_N1500_randomkeys65,903,5",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260228-0046_ENTANGLEONLY_GPU_AR1000_N2000_randomkeys49,785,283",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260228-0046_ENTANGLEONLY_GPU_AR1000_N1250_randomkeys304,674,793",
    "/n/home01/yjung/Github/entanglement-optimization/runs/20260228-0046_ENTANGLEONLY_GPU_AR1000_N1000_randomkeys815,484,934"
]

base_out_dir = Path("/n/home01/yjung/Github/entanglement-optimization/relaxation_gpu_large_N_2nd")

for run in runs:
    run_dir = Path(run)
    if not run_dir.exists():
        print(f"Directory {run_dir} does not exist.")
        continue

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

    entangle_subdirs = [d for d in results_dir.iterdir() if d.is_dir() and "EntangledPacking" in d.name]

    if entangle_subdirs:
        packing_dir = entangle_subdirs[0]
        
        # Files to copy
        files_to_copy = ["x_entangled.txt", "q_entangled.txt", "log_entangle_only.txt", "q_entangled.npy", "x_entangled.npy"]
        
        for fname in files_to_copy:
            in_file = packing_dir / fname
            if in_file.exists():
                shutil.copy2(in_file, out_folder / fname)
                print(f"Copied {fname} to {out_folder}")
            else:
                print(f"File {in_file} not found")
        
        # Also export x_entangled.txt as x_relaxed_AR1000.txt with header for consistency with earlier extraction logic
        parts = packing_dir.name.split('-')
        ar_val = None
        for p in parts:
            if p.startswith("AR"):
                try:
                    ar_val = int(p[2:])
                except ValueError:
                    pass
        if ar_val is not None:
             in_file = packing_dir / "x_entangled.txt"
             out_file = out_folder / f"x_relaxed_AR{ar_val}.txt"
             rod_radius = 0.5 / ar_val
             with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
                 f_out.write(f"# rod_radius = {rod_radius:g}\n")
                 f_out.write(f_in.read())
             print(f"Created {out_file.name} for compatibility in {out_folder}")
    else:
        print(f"No EntangledPacking folder found in {results_dir}")

print("Done.")
