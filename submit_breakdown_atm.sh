#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 00:50:00
#SBATCH --nodes=1
#SBATCH -o atm_break_11%j.log


source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations


python breakdown.py -c atm -v Z500 -i 11
python breakdown.py -c atm -v U200 -i 11
python breakdown.py -c atm -v U850 -i 11
python breakdown.py -c atm -v MSLP -i 11
python breakdown.py -c atm -v TMP2m -i 11
python breakdown.py -c atm -v U10m -i 11
python breakdown.py -c atm -v V10m -i 11
