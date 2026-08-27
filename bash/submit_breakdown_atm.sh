#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH -o bkdn_atm_%j.log


source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations

for var in Z500 U200 U850 MSLP TMP2m U10 V10m ; do
 python breakdown.py -c atm -v $var -i 05 -t 6 7 8 
 python breakdown.py -c atm -v $var -i 05 -t 12 1 2 
 python breakdown.py -c atm -v $var -i 11 -t 6 7 8 
 python breakdown.py -c atm -v $var -i 11 -t 12 1 2 
done

