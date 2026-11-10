#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --mail-user=aaf92@pitt.edu    
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# load modules
module load python/ondemand-jupyter-python3.11

#### CARNIVAL infer receptor activity ####
echo "==== Analysis Start: $(date) ===="
source activate carnival_env
python -u /ix/djishnu/Aaron_F/proteomics_proj/Code/Figures/Fig2/Fig2_KL_MEA/01_KL_MEA.py
source deactivate
echo "==== Analysis End: $(date) ===="