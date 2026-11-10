#!/bin/bash
#SBATCH --job-name=bamCoverage
#SBATCH --nodes=1
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --mail-user=aaf92@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=06:00:00

# load modules -- check `module avail deeptools` on your cluster if this
# exact name/version doesn't exist; deepTools is also installable via
# conda/pip if no module is available
module load deeptools

# same BAM used for the MACS3 peak calling
bam_file="/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/bam/UT_merged.bam"
outdir="/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/bigwig"

mkdir -p "$outdir"

# BAM needs to be indexed for bamCoverage -- skip if a .bai already exists
if [ ! -f "${bam_file}.bai" ]; then
    module load samtools
    samtools index "$bam_file"
fi

out_name=$(basename "$bam_file" .bam)

# RPGC-normalized signal track (1x average genome coverage), 10bp
# bins -- deepTools' recommended normalization specifically for
# ATAC-seq/ChIP-seq genome browser tracks (vs CPM, which is more a
# "quick comparison" convention). RPGC values read as direct fold-
# enrichment over expected genome-wide average coverage, which is
# closer to what's typically expected in a published OCR track.
# effectiveGenomeSize = 2652783500 (GRCm38/mm10, confirmed against
# deepTools' own documented table).
bamCoverage -b "$bam_file" \
    -o "${outdir}/${out_name}.bw" \
    --binSize 10 \
    --normalizeUsing RPGC \
    --effectiveGenomeSize 2652783500 \
    --numberOfProcessors 8
