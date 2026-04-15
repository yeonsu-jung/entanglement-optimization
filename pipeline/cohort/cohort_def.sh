# cohort_def.sh — sourced by run_local.sh and run_cluster.sh
#
# Edit this file to change the cohort grid.  Never edit it per-machine.

# Main grid: 8 × 8 = 64 packings
# N_MAIN=(10 20 50 100 200 300 500 1000)
# AR_MAIN=(10 20 50 100 200 300 500 1000)
N_MAIN=(500)
AR_MAIN=(1000)

# Large-N supplement: 2 × 2 = 4 packings
# N_LARGE=(1500 2000)
# AR_LARGE=(500 1000)
N_LARGE=(2000)
AR_LARGE=(1000)

# FIRE iteration budgets (can be overridden by env vars)
NMAX="${NMAX:-10000}"
MAX_ITERS="${MAX_ITERS:-1000000}"
