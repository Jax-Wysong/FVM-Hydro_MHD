#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=2D-MHD         # job name  
#SBATCH --partition=compute       # queue/partition
#SBATCH --nodes=1                 # total nodes          
#SBATCH --ntasks-per-node=1      # MPI ranks per node   
#SBATCH --time=00-02:00:00         # wall-clock limit
#SBATCH --mem=10G
#SBATCH --output=advection-N32-HLLD.out       # STDOUT
#SBATCH --error=advection-N32-HLLD.err        # STDERR


# --------- eval command line options ---------- #
python -u main.py

