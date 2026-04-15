# GPU Pipeline Overview: Entanglement & Relaxation

The optimization pipeline for finding dense, entangled packings of flexible rods is broken down into two main physical phases:
1. **Entanglement:** Driving a set of very thin, ghost-like rods (AR = 1000) into a tightly knotted configuration by minimizing an artificial "linking" potential.
2. **Sequential Relaxation:** Inflating the rods from thin (AR=1000) to thick (AR=10) while violently repelling collisions to find valid, non-overlapping geometric packings. 

Because entanglement is computationally expensive for large $N$, you have two different pathways to run this pipeline: **The One-Step Pathway** (all automated in a single job) and **The Two-Step Pathway** (entangle many, filter the best, and resume relaxation).

---

## 1. The One-Step Pathway (End-to-End)
This pathway handles the entire process—from random initialization to the thickest rod packing—inside a single, continuous SLURM job.

**When to use:** For standard runs ($N \le 1000$) where you want an immediate fully-relaxed result without intermediate manual steps.

**Scripts Involved:**
- `submit_sequential_protocol_gpu.py` (SLURM Manager)
- `sequential_protocol_gpu.py` (Worker Script)

**How it works:**
1. **Submit:** You run the submit script (e.g., `python submit_sequential_protocol_gpu.py --num-rods 500 --AR-list 1000,500,200`). This sweeps over random seeds and submits jobs.
2. **Entangle:** The worker uses JAX GPU placement and the native `optimize_fire2` loop to entangle the $N$ rods at the highest AR in the list (e.g., AR=1000).
3. **Relax sequentially:** Once entanglement converges, it immediately passes the state `q` into the `gpu_relax_collision` FIRE optimizer for AR=1000, saves the results, inflates the radius for AR=500, relaxes again, saves, and repeats down the list.

**Outputs:** A single folder `results/{k1},{k2},{k3}/SequentialRelaxedPacking.../` containing all AR subdirectories.

---

## 2. The Two-Step Pathway (Separated)
This pathway decouples entanglement from relaxation. Because entanglement is stochastic and some topologies pack tighter than others, this allows you to generate massive batches of entangled states, extract only the most promising ones, and then spend the computational resources to relax them later.

**When to use:** For massive systems ($N = 1500, 2000+$) where entanglement alone takes significant time, or when you want curated control over which topologies get relaxed.

### Step 2a: Mass Entanglement
Instead of relaxing, we just generate entangled states.
- **Scripts:** `run_entangle_only_gpu.sh` / `submit_entangle_only_gpu.py` $\rightarrow$ `entangle_only_gpu.py`
- **Action:** Submits hundreds of parallel SLURM jobs that do nothing but initialize random placements, run the optimized JAX FIRE entanglement loop at AR 1000, and save the raw `q_entangled.npy` state.

### Step 2b: Extraction
Consolidating the successful runs.
- **Script:** `export_large_N_gpu_runs.py`
- **Action:** You specify a list of successful Entangle-Only run directories. The script extracts the geometries (`q_entangled.npy`, `x_entangled.txt`) and organizes them neatly into a master storage directory (e.g., `relaxation_gpu_large_N/N1500/{randomkeys}/`).

### Step 2c: Resume Sequential Relaxation
Taking the curated entangled states and inflating them.
- **Scripts:** `submit_resume_sequential_gpu.py` $\rightarrow$ `resume_sequential_gpu.py`
- **Action:** You point the wrapper script at your master storage directory (e.g., `relaxation_gpu_large_N`). It recursively finds all `q_entangled.npy` files and dispatches independent SLURM jobs to `seas_gpu`. Each job loads the cached entangled state and applies the fast JAX FIRE `gpu_relax_collision` loop sequentially down your AR list (e.g., 1000 $\rightarrow$ 500 $\rightarrow$ 300).

---

### Core GPU Technology Under the Hood
Regardless of which pathway you choose, the heavy lifting is completely bounded to the GPU hardware:
* **`optimize_fire2` (Entanglement Phase):** Fully unrolled via `jax.lax.while_loop`. Calculates precise gradients of structural linking natively on the device.
* **`gpu_relax_collision` (Relaxation Phase):** Re-implemented to use the same fast-converging FIRE algorithm internally. It aggressively zeroes momentum on uphill climbs ($P \le 0$), allowing it to gracefully obliterate overlaps and reach perfect 0-contact states in thousands of iterations instead of millions.
