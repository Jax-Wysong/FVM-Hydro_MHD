#!/bin/bash --login
#  ---------- Slurm directives ----------
#SBATCH --job-name=256None      # job name (also used for log filenames below)
#SBATCH --partition=scavenger        # queue/partition
#SBATCH --nodes=1                    # total nodes
#SBATCH --ntasks-per-node=1          # MPI ranks per node
#SBATCH --time=01-00:00:00           # wall-clock limit
#SBATCH --mem=20G
#SBATCH --output=results_RK2_convergence_test/smooth_vortex_N128_MC_limit/output.out              # STDOUT  (%x = job name)
#SBATCH --error=results_RK2_convergence_test/smooth_vortex_N128_MC_limit/error.err               # STDERR  (%x = job name)

# ------ getting python ------
module purge
module load Miniforge3/25.11.0-1
conda activate mhd


# -------- parameters --------
N=128
TF=20.0
CFL=0.3
OUTDIR=results_RK2_convergence_test/smooth_vortex_N128_MC_limit

# -----------------------------------------
#mkdir -p ${OUTDIR}
python -u main.py --N ${N} --tf ${TF} --CFL ${CFL} --outdir ${OUTDIR}

### write job information to SLURM output file.
scontrol show job $SLURM_JOB_ID

### write resource usage to SLURM output file (uses a powertools command).
module load powertools
js -j $SLURM_JOB_ID