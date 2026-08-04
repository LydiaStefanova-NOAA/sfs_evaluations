#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 00:50:00
#SBATCH --nodes=1
#SBATCH -o ocn_acc_11%j.log


source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations


python test_unified_snr_acc.py -c ocn -v taux -i 11
python test_unified_snr_acc.py -c ocn -v tauy -i 11
python test_unified_snr_acc.py -c ocn -v dt20c -i 11
python test_unified_snr_acc.py -c ocn -v SST -i 11
python test_unified_snr_acc.py -c ocn -v SSS -i 11
python test_unified_snr_acc.py -c ocn -v SSH -i 11
