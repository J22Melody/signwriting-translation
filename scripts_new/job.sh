#!/bin/bash
#
#
### comment lines start with ## or #+space
### slurm option lines start with #SBATCH


#SBATCH --job-name=baseline  	## job name
#SBATCH --time=3-00:00:00       ## days-hours:minutes:seconds
###SBATCH --mem=20g             ##   3GB ram (hardware ratio is < 4GB/core)

### SBATCH --output=job.out	## standard out file
#SBATCH --ntasks=1            ## Ntasks.  default is 1.
#SBATCH --cpus-per-task=1	## Ncores per task.  Use greater than 1 for multi-threaded jobs.  default is 1.
#SBATCH --account=iict-sp2.volk.cl.uzh    ## only need to specify if you belong to multiple tenants on ScienceCluster
#SBATCH --gres gpu:1
###SBATCH --constraint=GPUMEM32GB|GPUMEM80GB

module load anaconda3
source activate signbank+
srun $@
