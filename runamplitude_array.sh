#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 00:45:00               # Slightly increased buffer time
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4         # Request 4 CPUs per task for parallel numpy/xarray operations
#SBATCH --mem=16G                 # Explicitly allocate 16 GB RAM per array task (adjust to 32G if needed)
#SBATCH --array=1-102%15          # Limit to 15 concurrent tasks to prevent memory/IO thrashing
#SBATCH -o sfs_break_%A_%a.log      # %A = main Job ID, %a = Task Array ID

# Limit OpenMP/MKL thread inflation to match allocated CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations

# Build list of all parameter strings
PARAMS=()

for script in "amplitude_diagnostic.py"; do
  # Atmospheric variables
  for var in Z500 U200 U850 T200 T850 MSLP TMP2m U10m V10m; do
    PARAMS+=("$script -c atm -v $var -i 05 -L 1 2 3 ")
    PARAMS+=("$script -c atm -v $var -i 05 -L 7 8 9 ")
  done

  # Ocean variables
  for var in taux tauy dt20c SST SSS SSH ocnheat MLD_003; do
    PARAMS+=("$script -c ocn -v $var -i 05 -L 1 2 3 ")
    PARAMS+=("$script -c ocn -v $var -i 05 -L 7 8 9 ")
  done
done

# Map 1-based SLURM_ARRAY_TASK_ID to 0-based array index
PARAM_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CMD_ARGS=${PARAMS[$PARAM_INDEX]}

echo "Running task ${SLURM_ARRAY_TASK_ID}: python ${CMD_ARGS}"
python ${CMD_ARGS}
