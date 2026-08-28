#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 00:45:00               # Slightly increased buffer time
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4         # Request 4 CPUs per task for parallel numpy/xarray operations
#SBATCH --mem=16G                 # Explicitly allocate 16 GB RAM per array task (adjust to 32G if needed)
#SBATCH --array=1-16          # Limit to 15 concurrent tasks to prevent memory/IO thrashing
#SBATCH -o sfs_break_%A_%a.log      # %A = main Job ID, %a = Task Array ID

# Limit OpenMP/MKL thread inflation to match allocated CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations

# Build list of all parameter strings
PARAMS=()

for script in "breakdown.py" "acc_snr_examples.py"; do
  # Append --nvr_method acc_varobs ONLY when running breakdown.py
  EXTRA_ARGS=""
  if [ "$script" == "breakdown.py" ]; then
    EXTRA_ARGS="--nvr-method acc_varobs"
  fi

  # Atmospheric variables
  for var in Z500 TMP2m ; do
    PARAMS+=("$script -c atm -v $var -i 05 -L 1 2 3 $EXTRA_ARGS")
    PARAMS+=("$script -c atm -v $var -i 05 -L 7 8 9 $EXTRA_ARGS")
  done

  # Ocean variables
  for var in SST SSH ; do
    PARAMS+=("$script -c ocn -v $var -i 05 -L 1 2 3 $EXTRA_ARGS")
    PARAMS+=("$script -c ocn -v $var -i 05 -L 7 8 9 $EXTRA_ARGS")
  done
done

# Map 1-based SLURM_ARRAY_TASK_ID to 0-based array index
PARAM_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CMD_ARGS=${PARAMS[$PARAM_INDEX]}

echo "Running task ${SLURM_ARRAY_TASK_ID}: python ${CMD_ARGS}"
python ${CMD_ARGS}
