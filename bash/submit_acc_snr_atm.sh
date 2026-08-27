#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH -o atm_acc_%j.log


source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations


for script in "test_unified_snr_acc.py" "breakdown.py" ; do
for var in Z500 U200 U850 MSLP TMP2m U10 V10m ; do
 python $script -c atm -v $var -i 05 -t 6 7 8 
 python $script -c atm -v $var -i 05 -t 12 1 2 
 python $script -c atm -v $var -i 11 -t 6 7 8 
 python $script -c atm -v $var -i 11 -t 12 1 2 
done


for var in taux tauy dt20c SST SSS SSH ocnheat ; do
 python $script -c ocn -v $var -i 05 -t 6 7 8 
 python $script -c ocn -v $var -i 05 -t 12 1 2 
 python $script -c ocn -v $var -i 11 -t 6 7 8 
 python $script -c ocn -v $var -i 11 -t 12 1 2 
done
done



