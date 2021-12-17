#!/bin/bash
#
#
### comment lines start with ## or #+space
### slurm option lines start with #SBATCH


#SBATCH --job-name=baseline  	## job name
#SBATCH --time=3-00:00:00       ## days-hours:minutes:seconds
#SBATCH --mem=4000M             ##   3GB ram (hardware ratio is < 4GB/core)

### SBATCH --output=job.out	## standard out file
#SBATCH --ntasks=1            ## Ntasks.  default is 1.
#SBATCH --cpus-per-task=1	## Ncores per task.  Use greater than 1 for multi-threaded jobs.  default is 1.
###SBATCH --partition=generic  ##  can specify partition here, but it is pre-empted by what module is loaded
###SBATCH --account=your_tenant_name    ## only need to specify if you belong to multiple tenants on ScienceCluster
#SBATCH --gres gpu:1
#SBATCH --gres gpu:Tesla-V100-32GB:1

module load nvidia/cuda11.2-cudnn8.1.0
module load anaconda3
source activate sockeye
pip install sockeye
pip install --pre -f https://dist.mxnet.io/python 'mxnet-cu112>=2.0.0b2021'
pip install mxboard

stdbuf -o0 -e0 srun --unbuffered $1 $2