#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 00:30:00               # Reduced time limit since tasks now run in parallel
#SBATCH --nodes=1
#SBATCH --array=1-136             # 136 total parameter combinations
#SBATCH -o atm_acc_%A_%a.log      # %A = main Job ID, %a = Task Array ID

source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations

# Build list of all parameter strings
PARAMS=()

for script in "test_unified_snr_acc.py" "breakdown.py"; do
  # Atmospheric variables
  for var in Z500 U200 U850 T200 T850 MSLP TMP2m U10 V10m; do
    PARAMS+=("$script -c atm -v $var -i 05 -t 6 7 8")
    PARAMS+=("$script -c atm -v $var -i 05 -t 12 1 2")
    PARAMS+=("$script -c atm -v $var -i 11 -t 6 7 8")
    PARAMS+=("$script -c atm -v $var -i 11 -t 12 1 2")
  done

  # Ocean variables
  for var in taux tauy dt20c SST SSS SSH ocnheat MLD_003; do
    PARAMS+=("$script -c ocn -v $var -i 05 -t 6 7 8")
    PARAMS+=("$script -c ocn -v $var -i 05 -t 12 1 2")
    PARAMS+=("$script -c ocn -v $var -i 11 -t 6 7 8")
    PARAMS+=("$script -c ocn -v $var -i 11 -t 12 1 2")
  done
done

# Map 1-based SLURM_ARRAY_TASK_ID to 0-based array index
PARAM_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CMD_ARGS=${PARAMS[$PARAM_INDEX]}

echo "Running task ${SLURM_ARRAY_TASK_ID}: python ${CMD_ARGS}"
python ${CMD_ARGS}
