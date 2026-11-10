"""
Fig3 ATAC-seq visualization, step 4: for EACH of the top 10 IL10 2hr
DEGs (from 01_top_degs_and_confirmed_tfs.py's saved
top10_IL10_2hr_DEGs.csv), plot the ATAC-seq signal (bigwig, from
run_bamCoverage.sh) across that gene's own 8000_TSS_8000 promoter
window. No STAT motif highlighting here -- pure signal tracks, one
grid panel per gene.

STRAND HANDLING: unlike the single-gene Il4ra script (which is + strand
and could skip this), the top-10 DEG list will generally include BOTH
+ and - strand genes, so this is handled properly here:
  - + strand: genomic position increases in the same direction as
    "position relative to TSS" (TSS = bed_start + upstream).
  - - strand: genomic position DECREASES in the gene-sense upstream->
    downstream direction (TSS = bed_start + downstream = bed_end -
    upstream), matching how 8000_TSS_8000_promoters_mm10.bed was
    constructed (get_promoter_fasta.py's write_bed). The extracted
    signal and genomic-position arrays are REVERSED for - strand genes
    so relative-to-TSS position is ascending left-to-right for every
    gene, consistent with the + strand convention.

Requires pyBigWig: pip install pyBigWig --break-system-packages
"""

import os
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib.pyplot as plt

# ---- data sources ----
bigwig_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/bigwig/UT_merged.bw"
promoters_bed_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/8000_TSS_8000_promoters_mm10.bed"
top10_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/top10_IL10_2hr_DEGs.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

UPSTREAM = 8000
DOWNSTREAM = 8000

for f, label in [(bigwig_f, "bigwig"), (promoters_bed_f, "promoters BED"), (top10_f, "top10 DEG CSV")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label} file: {f}")

# ---- load the top 10 DEGs ----
top10_df = pd.read_csv(top10_f)
top10_genes = top10_df["gene"].tolist()
print(f"Top 10 genes: {top10_genes}")

# ---- load the promoter BED, index by gene name ----
promoters_bed = pd.read_csv(
    promoters_bed_f, sep="\t", header=None,
    names=["chrom", "start", "end", "gene", "score", "strand"]
)
promoters_bed = promoters_bed.set_index("gene")

# ---- open the bigwig once, reuse for all genes ----
bw = pyBigWig.open(bigwig_f)
bw_chroms = set(bw.chroms().keys())


def get_gene_window_signal(gene):
    """Returns (relative_positions, values) for this gene's promoter
    window, ascending in relative-to-TSS order regardless of strand."""
    if gene not in promoters_bed.index:
        print(f"  WARNING: {gene} not found in promoters BED, skipping")
        return None, None

    row = promoters_bed.loc[gene]
    chrom, bed_start, bed_end, strand = row["chrom"], int(row["start"]), int(row["end"]), row["strand"]

    if chrom not in bw_chroms:
        print(f"  WARNING: chrom '{chrom}' for {gene} not found in bigwig, skipping")
        return None, None

    if bed_end - bed_start != UPSTREAM + DOWNSTREAM:
        print(f"  WARNING: {gene}'s window size ({bed_end - bed_start}) doesn't match "
              f"UPSTREAM+DOWNSTREAM ({UPSTREAM + DOWNSTREAM}) -- using it anyway, but flagging")

    values = bw.values(chrom, bed_start, bed_end, numpy=True)
    values = np.nan_to_num(values, nan=0.0)
    genomic_positions = np.arange(bed_start, bed_end)

    if strand == "+":
        tss_genomic = bed_start + UPSTREAM
        relative_positions = genomic_positions - tss_genomic
    else:  # "-" strand
        tss_genomic = bed_start + DOWNSTREAM  # == bed_end - UPSTREAM
        relative_positions = tss_genomic - genomic_positions
        # reverse so relative_positions ends up ascending left-to-right,
        # same convention as + strand genes
        relative_positions = relative_positions[::-1]
        values = values[::-1]

    return relative_positions, values


# =============================================================================
# Grid of signal tracks, one per gene
# =============================================================================
ncols = 2
nrows = int(np.ceil(len(top10_genes) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 2.5 * nrows), squeeze=False)
axes = axes.flatten()

for i, gene in enumerate(top10_genes):
    ax = axes[i]
    relative_positions, values = get_gene_window_signal(gene)

    if relative_positions is None:
        ax.set_title(f"{gene} (no data)", fontsize=9)
        ax.axis("off")
        continue

    ax.fill_between(relative_positions, values, color="#4477AA", alpha=0.7, linewidth=0)
    ax.axvline(0, color="#B2182B", linewidth=1.0, linestyle="--", zorder=3)
    ax.set_xlim(-UPSTREAM, DOWNSTREAM)
    ax.set_title(gene, fontsize=10, fontweight="bold")
    ax.set_ylabel("ATAC (CPM)", fontsize=8)
    ax.tick_params(labelsize=7)

for j in range(len(top10_genes), len(axes)):
    axes[j].axis("off")

for ax in axes[max(0, len(top10_genes) - ncols):len(top10_genes)]:
    ax.set_xlabel("Position relative to TSS (bp)", fontsize=8)

bw.close()

fig.suptitle(f"ATAC-seq signal (resting BMDM, CPM) -- {UPSTREAM}_TSS_{DOWNSTREAM} promoter windows, "
             f"top 10 IL10 2hr DEGs", fontsize=12, y=1.01)
plt.tight_layout()

grid_png = f"{out_dir}/top10_DEG_ATAC_signal_tracks.png"
grid_svg = f"{out_dir}/top10_DEG_ATAC_signal_tracks.svg"
plt.savefig(grid_png, dpi=300, bbox_inches="tight")
plt.savefig(grid_svg, bbox_inches="tight")
plt.show()

print(f"\nSaved: {grid_png}")
print("\nDone.")
