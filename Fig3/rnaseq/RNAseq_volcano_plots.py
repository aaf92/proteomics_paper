"""
Fig3A: volcano plots for IL6 and IL10 (1 ng/mL) at 1, 2, and 4 hours vs
PBS -- one plot per cytokine/timepoint (6 total). DEGs defined as
adjusted p < 0.05 AND 2**abs(log2FoldChange) > 1.5 (i.e. >=1.5x linear
fold change), same convention used throughout this project.

NEW: up/down genes with a STAT3 motif hit (p<1e-4, 8000_TSS_8000
window) that ALSO falls inside a MACS3-called ATAC peak are flagged
by COLOR ONLY (same marker size as unflagged points) -- same
coordinate-conversion/overlap logic as
06_cooccurrence_ATAC_signal_tracks.py (STAT3 FIMO positions need NO
strand-flipping, already corrected at extraction time; MACS3 peaks DO
need the strand-aware genomic->TSS-relative conversion).

NEW: genes in the "TFEB only" region of
08_macs3_peak_cooccurrence_venn3.py's saved venn region CSV get a text
label on any volcano they appear on. Labels are positioned via
adjustText (pip install adjustText --break-system-packages), which
auto-repositions them to avoid overlapping each other or the point
cloud, and draws a thin leader line back to each gene's actual point.

Column naming in these DESeq2 result tables isn't yet confirmed (some
tables in this project use "padj", others use "adjusted_padj") -- the
script checks for both and raises a clear error if neither is found,
rather than silently guessing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text

# ---- data sources ----
deseq2_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Data/RNAseq/DESeq2_results"
stat3_fimo_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results/8000_TSS_8000_stats_all/fimo.tsv"
promoters_bed_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/8000_TSS_8000_promoters_mm10.bed"
narrowpeak_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/peakcall_res/UT/UT_merged_peaks.narrowPeak"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/volcano_plots"
os.makedirs(out_dir, exist_ok=True)

CYTOKINES = ["il6", "il10"]
DOSE = "1"  # 1 ng/mL
TIMEPOINTS = ["1", "2", "4"]  # hours

padj_thresh = 0.05
fc_thresh = 1.5  # linear fold change, applied as 2**abs(log2FoldChange) > fc_thresh
stat3_pval_thresh = 1e-4  # FIMO default, same convention used throughout this project

UPSTREAM = 8000
DOWNSTREAM = 8000
TSS_FASTA_POS = UPSTREAM + 1

UP_COLOR = "#E41A1C"
DOWN_COLOR = "#377EB8"
NS_COLOR = "#B0B0B0"
UP_FLAGGED_COLOR = "#800000"    # dark maroon -- STAT3-in-peak up genes
DOWN_FLAGGED_COLOR = "#08306B"  # dark navy -- STAT3-in-peak down genes

CYTOKINE_LABEL = {"il6": "IL-6", "il10": "IL-10"}

X_LIMIT = 2.75


def find_padj_col(df):
    """Checks for either naming convention used across this project's
    result tables -- raises rather than silently guessing."""
    for candidate in ["padj", "adjusted_padj"]:
        if candidate in df.columns:
            return candidate
    raise SystemExit(f"Could not find a recognized padj column (checked 'padj', 'adjusted_padj'). "
                      f"Actual columns: {list(df.columns)}")


# =============================================================================
# Build the per-gene "has a STAT3 motif inside a MACS3 peak" flag --
# same logic/convention as 06_cooccurrence_ATAC_signal_tracks.py
# =============================================================================
def to_relative(pos):
    """FIMO in-sequence position -> TSS-relative position. NO strand
    handling needed -- bedtools getfasta -s already reverse-
    complemented minus-strand genes at extraction time."""
    return pos - TSS_FASTA_POS


def genomic_to_relative(genomic_pos, tss_genomic, strand):
    """Strand-aware genomic -> TSS-relative conversion, for anything
    pulled from RAW genomic coordinates (MACS3 peaks here) -- NOT for
    FIMO hit positions."""
    if strand == "+":
        return genomic_pos - tss_genomic
    else:
        return tss_genomic - genomic_pos


def spans_overlap(a_start, a_stop, spans):
    for (s, e) in spans:
        if a_start < e and a_stop > s:
            return True
    return False


print("Building per-gene STAT3-motif-in-MACS3-peak flags...")
for f, label in [(stat3_fimo_f, "STAT3 FIMO"), (promoters_bed_f, "promoters BED"), (narrowpeak_f, "narrowPeak")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label} file: {f}")

stat3_hits = pd.read_csv(stat3_fimo_f, sep="\t", comment="#")
stat3_hits = stat3_hits[stat3_hits["motif_id"].str.upper().str.startswith("STAT3.")]
stat3_hits = stat3_hits.dropna(subset=["p-value"])
stat3_hits = stat3_hits[stat3_hits["p-value"] < stat3_pval_thresh]
print(f"  STAT3 hits (p<{stat3_pval_thresh}): {len(stat3_hits)} across {stat3_hits['sequence_name'].nunique()} genes")

promoters_bed = pd.read_csv(
    promoters_bed_f, sep="\t", header=None,
    names=["chrom", "start", "end", "gene", "score", "strand"]
).set_index("gene")

narrowpeak_cols = ["chrom", "start", "end", "name", "score", "strand",
                    "signalValue", "pValue", "qValue", "summit"]
peaks_df = pd.read_csv(narrowpeak_f, sep="\t", header=None, names=narrowpeak_cols)

gene_stat3_in_peak = {}
for gene, gdf in stat3_hits.groupby("sequence_name"):
    if gene not in promoters_bed.index:
        continue
    row = promoters_bed.loc[gene]
    chrom, bed_start, bed_end, strand = row["chrom"], int(row["start"]), int(row["end"]), row["strand"]
    tss_genomic = bed_start + UPSTREAM if strand == "+" else bed_start + DOWNSTREAM

    peak_overlap = peaks_df[
        (peaks_df["chrom"] == chrom) & (peaks_df["start"] < bed_end) & (peaks_df["end"] > bed_start)
    ]
    peak_spans = []
    for _, prow in peak_overlap.iterrows():
        r1 = genomic_to_relative(prow["start"], tss_genomic, strand)
        r2 = genomic_to_relative(prow["end"], tss_genomic, strand)
        peak_spans.append((min(r1, r2), max(r1, r2)))

    in_peak = False
    for _, hit in gdf.iterrows():
        hit_r1, hit_r2 = to_relative(hit["start"]), to_relative(hit["stop"])
        if spans_overlap(hit_r1, hit_r2, peak_spans):
            in_peak = True
            break
    gene_stat3_in_peak[gene] = in_peak

n_flagged = sum(gene_stat3_in_peak.values())
print(f"  {n_flagged} genes have a STAT3 motif (p<{stat3_pval_thresh}) inside a MACS3 peak")

# ---- TFEB-only genes (from 08_macs3_peak_cooccurrence_venn3.py's
# saved region CSV) -- these get gene-name text labels on the volcanoes ----
venn_region_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/macs3_peak_cooccurrence_venn3_region_genes.csv"
if not os.path.exists(venn_region_f):
    raise SystemExit(f"Missing venn region CSV: {venn_region_f} -- run 08_macs3_peak_cooccurrence_venn3.py first.")

venn_region_df = pd.read_csv(venn_region_f)
tfeb_only_genes = set(venn_region_df.loc[venn_region_df["region"] == "TFEB only", "gene"])
print(f"  TFEB-only genes to label on volcanoes: {len(tfeb_only_genes)}")


def plot_volcano(cytokine, timepoint):
    f = f"{deseq2_dir}/res_{cytokine}_{DOSE}_{timepoint}.csv"
    if not os.path.exists(f):
        print(f"  WARNING: missing {f}, skipping")
        return

    df = pd.read_csv(f, index_col=0)
    padj_col = find_padj_col(df)

    df = df.dropna(subset=[padj_col, "log2FoldChange"]).copy()
    df["fold_change"] = 2 ** df["log2FoldChange"].abs()
    df["is_deg"] = (df[padj_col] < padj_thresh) & (df["fold_change"] > fc_thresh)
    df["direction"] = np.where(
        ~df["is_deg"], "ns",
        np.where(df["log2FoldChange"] > 0, "up", "down")
    )
    df["stat3_in_peak"] = df.index.map(lambda g: gene_stat3_in_peak.get(g, False))

    n_up = (df["direction"] == "up").sum()
    n_down = (df["direction"] == "down").sum()
    n_ns = (df["direction"] == "ns").sum()
    n_up_flagged = ((df["direction"] == "up") & df["stat3_in_peak"]).sum()
    n_down_flagged = ((df["direction"] == "down") & df["stat3_in_peak"]).sum()
    print(f"  {cytokine.upper()} {timepoint}hr: {n_up} up ({n_up_flagged} w/ STAT3-in-peak), "
          f"{n_down} down ({n_down_flagged} w/ STAT3-in-peak), {n_ns} not significant (of {len(df)} total)")

    # clip -log10(padj) at a reasonable ceiling so a handful of extreme
    # p-values don't compress the rest of the plot
    neg_log10_padj = -np.log10(df[padj_col].clip(lower=1e-300))
    y_ceiling = neg_log10_padj.replace([np.inf], np.nan).quantile(0.999) * 1.2
    neg_log10_padj = neg_log10_padj.clip(upper=y_ceiling)

    # x-axis clipping: genes beyond +/- X_LIMIT get their PLOTTED x-value
    # clamped to the boundary, but keep a flag so they render as a
    # distinct marker (triangle) rather than an ordinary circle sitting
    # exactly at the cap
    x_vals = df["log2FoldChange"].clip(lower=-X_LIMIT, upper=X_LIMIT)
    is_clipped = df["log2FoldChange"].abs() > X_LIMIT
    n_clipped = is_clipped.sum()
    if n_clipped:
        print(f"  {n_clipped} gene(s) exceed +/-{X_LIMIT} log2FC and are shown clipped at the boundary")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_box_aspect(1)  # square AXES BOX -- independent of data-unit scale, since x
                          # (capped at +/-2.5) and y (-log10 padj, often 0-100+) are on
                          # very different scales; forcing equal data units here would
                          # make the plot unusably tall

    for direction, color, label in [
        ("ns", NS_COLOR, f"NS (n={n_ns})"),
        ("down", DOWN_COLOR, f"Down (n={n_down})"),
        ("up", UP_COLOR, f"Up (n={n_up})"),
    ]:
        # plain points (no STAT3-in-peak flag), not clipped
        mask = (df["direction"] == direction) & ~is_clipped & ~df["stat3_in_peak"]
        ax.scatter(x_vals[mask], neg_log10_padj[mask],
                   s=10, alpha=0.5, color=color, label=label,
                   zorder=3 if direction != "ns" else 1)

        clip_mask = (df["direction"] == direction) & is_clipped & ~df["stat3_in_peak"]
        if clip_mask.any():
            ax.scatter(x_vals[clip_mask], neg_log10_padj[clip_mask],
                       s=30, alpha=0.8, color=color, marker="^",
                       edgecolors="black", linewidths=0.5, zorder=4)

        # STAT3-in-peak flagged points -- only meaningful for up/down,
        # SAME size as the unflagged points, distinguished by color only
        if direction != "ns":
            flagged_color = UP_FLAGGED_COLOR if direction == "up" else DOWN_FLAGGED_COLOR

            flagged_mask = (df["direction"] == direction) & ~is_clipped & df["stat3_in_peak"]
            if flagged_mask.any():
                ax.scatter(x_vals[flagged_mask], neg_log10_padj[flagged_mask],
                           s=10, alpha=0.5, color=flagged_color, zorder=5)

            flagged_clip_mask = (df["direction"] == direction) & is_clipped & df["stat3_in_peak"]
            if flagged_clip_mask.any():
                ax.scatter(x_vals[flagged_clip_mask], neg_log10_padj[flagged_clip_mask],
                           s=30, alpha=0.8, color=flagged_color, marker="^",
                           edgecolors="black", linewidths=0.5, zorder=5)

    # ---- text labels for TFEB-only genes present on THIS volcano --
    # adjustText repositions labels to avoid overlapping each other
    # (and the point cloud), then draws leader lines back to each
    # gene's actual point ----
    genes_to_label = [g for g in tfeb_only_genes if g in df.index]
    texts = []
    anchors = {}  # text object -> (anchor_x, anchor_y)
    for g in genes_to_label:
        gx, gy = x_vals.loc[g], neg_log10_padj.loc[g]
        t = ax.text(gx, gy, g, fontsize=7, color="black", zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        texts.append(t)
        anchors[t] = (gx, gy)

    if texts:
        # let adjustText ONLY reposition the labels (no arrowprops --
        # its own arrow-drawing is what was unreliable). We draw every
        # connector line ourselves afterward, guaranteed for every label
        # regardless of how far adjustText moved it.
        adjust_text(texts, x=list(anchors[t][0] for t in texts), y=list(anchors[t][1] for t in texts), ax=ax)

        for t in texts:
            anchor_x, anchor_y = anchors[t]
            final_x, final_y = t.get_position()
            ax.plot([anchor_x, final_x], [anchor_y, final_y],
                    color="grey", linewidth=0.6, zorder=7)

        print(f"    Labeled {len(genes_to_label)}/{len(tfeb_only_genes)} TFEB-only genes on this volcano")

    ax.axhline(-np.log10(padj_thresh), color="grey", linestyle="--", linewidth=0.8, zorder=0)
    fc_line = np.log2(fc_thresh)
    ax.axvline(fc_line, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(-fc_line, color="grey", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xlim(-X_LIMIT, X_LIMIT)

    ax.set_xlabel("log2 fold change")
    ax.set_ylabel(f"-log10({padj_col})")
    ax.set_title(f"{CYTOKINE_LABEL[cytokine]} ({DOSE} ng/mL), {timepoint}hr vs PBS\n"
                 f"DEG cutoff: padj<{padj_thresh}, FC>{fc_thresh}x "
                 f"(^ = clipped at +/-{X_LIMIT} log2FC, n={n_clipped})\n"
                 f"bordered = STAT3 motif (p<{stat3_pval_thresh}) in a MACS3 peak "
                 f"({n_up_flagged} up, {n_down_flagged} down)")

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    flagged_handles = [
        plt.Line2D([0], [0], marker="o", color=UP_FLAGGED_COLOR, linestyle="none",
                   markersize=7, label="Up, STAT3 motif in MACS3 peak"),
        plt.Line2D([0], [0], marker="o", color=DOWN_FLAGGED_COLOR, linestyle="none",
                   markersize=7, label="Down, STAT3 motif in MACS3 peak"),
    ]
    ax.legend(handles=legend_handles + flagged_handles, loc="upper left", fontsize=8, frameon=False)

    plt.tight_layout()

    out_name = f"volcano_{cytokine}_{DOSE}ngml_{timepoint}hr"
    png_f = f"{out_dir}/{out_name}.png"
    svg_f = f"{out_dir}/{out_name}.svg"
    plt.savefig(png_f, dpi=300)
    plt.savefig(svg_f)
    plt.show()
    plt.close(fig)

    print(f"  Saved: {png_f}")


for cytokine in CYTOKINES:
    for timepoint in TIMEPOINTS:
        print(f"\n{cytokine.upper()} {timepoint}hr:")
        plot_volcano(cytokine, timepoint)

print("\nDone.")