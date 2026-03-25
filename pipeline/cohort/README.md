# Cohort Runner

Produces 68 independent (N, AR) entangled + relaxed packings.

```
pipeline/cohort/
  cohort_def.sh          ← edit this to change the grid
  run_local.sh           ← local execution (one GPU, sequential)
  run_cluster.sh         ← Harvard RC: one SLURM job per packing
  cluster.env.template   ← copy → cluster.env and fill in your values
  cluster.env            ← gitignored, never merges
  jobs/                  ← gitignored: generated sbatch scripts + logs
  results/               ← gitignored: output data
```

---

## First-time cluster setup

```bash
# 1. Clone (or pull) the repo on the login node
ssh yjung@login.rc.fas.harvard.edu
cd ~/Github/entanglement-optimization
git pull

# 2. Create your cluster.env from the template
cp pipeline/cohort/cluster.env.template pipeline/cohort/cluster.env
nano pipeline/cohort/cluster.env          # fill in the values below
```

**`cluster.env` fields:**

| Key | What it controls | Typical RC value |
|---|---|---|
| `MAMBA_ENV` | `mamba activate <name>` | `simdata-analysis` |
| `CUDA_MODULE` | `module load <name>` | `cuda/12.9` |
| `PARTITION` | SLURM partition | `gpu_requeue` (patient) / `gpu_test` (fast, 1 h limit) |
| `TIME_LIMIT` | `#SBATCH -t` | `0-08:00` |
| `MEM_MB` | memory per job | `32000` |
| `GRES` | GPU request | `gpu:1` |
| `MAIL_USER` | failure e-mail | `jung@seas.harvard.edu` |
| `REPO_ROOT` | absolute path to this repo | `/n/home01/yjung/Github/entanglement-optimization` |

---

## Running the cohort

```bash
cd ~/Github/entanglement-optimization/pipeline/cohort

# Dry run first — prints sbatch commands, submits nothing
bash run_cluster.sh --dry-run

# Submit all 68 jobs
bash run_cluster.sh 2>&1 | tee run_cluster.log

# Force recompute (skip-existing is the default)
bash run_cluster.sh --force 2>&1 | tee run_cluster.log
```

Each job runs in `jobs/ent_N<N>_AR<AR>_%j.out` / `.err`.
Results land in `results/N<N>/AR<AR>/`.

---

## Monitoring

```bash
# Live queue
squeue -u $USER

# After completion: exit codes + timing
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -X | grep ent_

# Quick pass/fail summary
grep -l "Done: N=" jobs/*.out | wc -l          # completed
grep -rl "FAILED\|Error\|Traceback" jobs/*.err | head  # failed
```

---

## Changing the grid

Edit only `cohort_def.sh` — both runners source it:

```bash
# cohort_def.sh
N_MAIN=(10 20 50 100 200 300 500 1000)
AR_MAIN=(10 20 50 100 200 300 500 1000)
N_LARGE=(1500 2000)
AR_LARGE=(500 1000)

NMAX=10000        # entangle FIRE budget
MAX_ITERS=1000000 # relax FIRE budget
```

Override budgets without editing the file:

```bash
NMAX=20000 MAX_ITERS=2000000 bash run_cluster.sh
```

---

## Adjusting resources per job

All SLURM parameters sit in `cluster.env`.
To use `gpu_test` for quick debugging:

```bash
# Override in-place (only affects this shell)
PARTITION=gpu_test TIME_LIMIT=0-01:00 bash run_cluster.sh --dry-run
```

To submit only the large-N jobs with more memory:

```bash
# Edit cluster.env temporarily OR pass inline
MEM_MB=64000 PARTITION=gpu bash run_cluster.sh
```

---

## Resuming after partial failure

The runners skip packings that already have `q_relaxed.npy`.
Simply re-run — completed packings are a no-op:

```bash
bash run_cluster.sh 2>&1 | tee run_cluster_resume.log
```

Use `--force` only if you want to recompute everything from scratch.
