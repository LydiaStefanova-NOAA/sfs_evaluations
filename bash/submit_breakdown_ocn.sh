#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH -o bkdn_ocn_%j.log


source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations

for var in taux tauy dt20c SST SSS SSH ; do
 python breakdown.py -c ocn -v $var -i 05 -t 6 7 8 
 python breakdown.py -c ocn -v $var -i 05 -t 12 1 2 
 python breakdown.py -c ocn -v $var -i 11 -t 6 7 8 
 python breakdown.py -c ocn -v $var -i 11 -t 12 1 2 
done

