#!/bin/bash
#SBATCH --job-name=stat3_cooccurrence_bg
#SBATCH --nodes=1
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --cpus-per-task=2
#SBATCH --mem=24g
#SBATCH --mail-user=aaf92@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=8:00:00
#SBATCH --output=/ix/djishnu/Aaron_F/proteomics_proj/Code/Figures/Fig3/atacseq_visualization/logs/cooccurrence_bg_%j.out
#SBATCH --error=/ix/djishnu/Aaron_F/proteomics_proj/Code/Figures/Fig3/atacseq_visualization/logs/cooccurrence_bg_%j.err


mkdir -p /ix/djishnu/Aaron_F/proteomics_proj/Code/Figures/Fig3/atacseq_visualization/logs
module load python/ondemand-jupyter-python3.11
source activate carnival_env

echo "Started at: $(date)"

python -u /ix/djishnu/Aaron_F/proteomics_proj/Code/Figures/Fig3/atacseq_visualization/10_STAT3_cooccurrence_background_distribution.py

echo "Finished at: $(date)"
