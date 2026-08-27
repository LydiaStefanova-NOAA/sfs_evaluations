#!/bin/bash
#SBATCH -A fv3-cpu
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH -o ice_break_%j.log


source /scratch4/NCEPDEV/ovp/Lydia.B.Stefanova/miniconda/bin/activate sfs_evaluations


python breakdown.py -c ice -v aice_h
