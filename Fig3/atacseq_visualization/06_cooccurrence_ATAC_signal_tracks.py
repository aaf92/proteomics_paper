"""
Fig3 ATAC-seq visualization, step 6: for EVERY gene with a STAT3/TF
co-occurrence (from 05_STAT_family_motif_proximity_check.py's saved
STAT_family_motif_proximity_50bp.csv), plot a 4-panel stacked locus
view, all sharing one TSS-relative x-axis:
  1. Gene model(s) -- ALL genes whose transcript body overlaps this
     window (not just the flagged gene), so neighboring genes clipped
     into the window are visible. The flagged gene's label is RED;
     all others are grey, so it's clear which gene the co-occurrence
     was originally found for.
  2. ATAC-seq signal (bigwig)
  3. MACS3-called peaks (narrowPeak intervals overlapping the window)
  4. STAT3/TF co-occurrence motif hits for that gene, each flagged
     (bordered) if it falls inside a MACS3 peak

Also writes a summary CSV flagging, for every co-occurrence pair,
whether the STAT3 hit and the other-TF hit each fall inside a MACS3
peak.

COORDINATE NOTE: FIMO hit positions (start/stop) do NOT need
strand-flipping -- get_promoter_fasta.py already reverse-complemented
minus-strand genes' sequences at EXTRACTION time (bedtools getfasta
-s), so every gene's extracted FASTA reads 5'->3' in gene-sense with
the TSS always at position UPSTREAM+1, regardless of genomic strand.
The ATAC bigwig signal, MACS3 peaks, and gene-model exons are all
pulled from RAW GENOMIC coordinates (untouched by that extraction
step), so all three need the same strand-aware flip.

Requires pyBigWig: pip install pyBigWig --break-system-packages
"""

import os
import re
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- data sources ----
bigwig_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/bigwig/UT_merged.bw"
promoters_bed_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/8000_TSS_8000_promoters_mm10.bed"
proximity_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/STAT_family_motif_proximity_50bp.csv"
gtf_f = "/ix/djishnu/Aaron_F/morgan_satarupa_RNAseq/Code/scripts/Data_processing/STAR_aligner/mm10_build/mm10.refGene.gtf"
narrowpeak_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/peakcall_res/UT/UT_merged_peaks.narrowPeak"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"

UPSTREAM = 8000
DOWNSTREAM = 8000
TSS_FASTA_POS = UPSTREAM + 1

# ---- refined, publication-style color palette, used consistently
# across all four panels ----
ATAC_FILL_COLOR = "#7B9EC9"           # muted steel blue -- signal track
PEAK_COLOR = "#B8860B"                # dark goldenrod -- clearly distinct from the signal's blue
EXON_COLOR = "#3D3D3D"                # dark charcoal -- exon boxes
INTRON_COLOR = "#999999"              # medium grey -- intron/baseline line
FLAGGED_GENE_LABEL_COLOR = "#C0392B"  # refined red -- the gene the co-occurrence was found for
OTHER_GENE_LABEL_COLOR = "#555555"    # neutral grey -- every other overlapping gene
STAT_COLOR = "#C0392B"                # refined red -- STAT3 motif
TF_COLOR = {
    "ERF": "#2E6F95",     # deep teal-blue
    "ETV3": "#D08B3A",    # warm amber (distinct from PEAK_COLOR's goldenrod -- different panel, but kept visually separable)
    "TFEB": "#4C9F70",    # muted green
    "ARID3A": "#8E6BA8",  # muted purple
}
MAX_GENE_ROWS = 5  # cap on how many overlapping genes get their own row in the gene-model panel

for f, label in [(bigwig_f, "bigwig"), (promoters_bed_f, "promoters BED"),
                  (proximity_f, "proximity CSV"), (gtf_f, "GTF"), (narrowpeak_f, "narrowPeak")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label} file: {f}")

proximity_df = pd.read_csv(proximity_f)
print(f"Loaded {len(proximity_df)} co-occurring pairs across "
      f"{proximity_df['other_family'].nunique()} families, {proximity_df['gene'].nunique()} unique genes")

promoters_bed = pd.read_csv(
    promoters_bed_f, sep="\t", header=None,
    names=["chrom", "start", "end", "gene", "score", "strand"]
).set_index("gene")


def parse_attribute(attr_str, key):
    m = re.search(rf'{key} "([^"]+)"', attr_str)
    return m.group(1) if m else None


# =============================================================================
# GENOME-WIDE GTF parse (not restricted to target genes) -- needed so
# we can find ANY gene overlapping a given window, not just the one
# the co-occurrence was flagged for.
# =============================================================================
def get_all_gene_models(gtf_path):
    transcript_info = {}   # (gene, transcript_id) -> (chrom, tss, strand, t_start, t_end)
    transcript_exons = {}  # (gene, transcript_id) -> list of (start, end)

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue
            feature = fields[2]
            chrom, start, end, strand, attrs = fields[0], int(fields[3]), int(fields[4]), fields[6], fields[8]
            gene_name = parse_attribute(attrs, 'gene_name') or parse_attribute(attrs, 'gene_id')
            if gene_name is None:
                continue
            transcript_id = parse_attribute(attrs, 'transcript_id')
            if transcript_id is None:
                continue

            if feature == 'transcript':
                tss = start if strand == '+' else end
                transcript_info[(gene_name, transcript_id)] = (chrom, tss, strand, start, end)
            elif feature == 'exon':
                transcript_exons.setdefault((gene_name, transcript_id), []).append((start, end))

    gene_best = {}
    for key, (chrom, tss, strand, t_start, t_end) in transcript_info.items():
        gene_name, transcript_id = key
        if gene_name not in gene_best:
            gene_best[gene_name] = (transcript_id, chrom, tss, strand, t_start, t_end)
        else:
            _, _, prev_tss, prev_strand, _, _ = gene_best[gene_name]
            if strand == '+' and tss < prev_tss:
                gene_best[gene_name] = (transcript_id, chrom, tss, strand, t_start, t_end)
            elif strand == '-' and tss > prev_tss:
                gene_best[gene_name] = (transcript_id, chrom, tss, strand, t_start, t_end)

    gene_models = {}
    for gene_name, (transcript_id, chrom, tss, strand, t_start, t_end) in gene_best.items():
        exons = sorted(transcript_exons.get((gene_name, transcript_id), []))
        gene_models[gene_name] = {
            "chrom": chrom, "tss_genomic": tss, "strand": strand,
            "t_start": t_start, "t_end": t_end, "exons": exons,
        }
    return gene_models


print("\nParsing GTF genome-wide for gene models (single pass)...")
all_gene_models = get_all_gene_models(gtf_f)
print(f"  Parsed {len(all_gene_models)} total gene models")

# ---- build a simple per-chrom list for overlap lookups ----
genes_by_chrom = {}
for gname, model in all_gene_models.items():
    genes_by_chrom.setdefault(model["chrom"], []).append((gname, model["t_start"], model["t_end"]))


def find_overlapping_genes(chrom, window_start, window_end):
    """Every gene (by name) whose transcript span overlaps this window."""
    candidates = genes_by_chrom.get(chrom, [])
    return [g for g, t_start, t_end in candidates if t_start < window_end and t_end > window_start]


# ---- load MACS3 narrowPeak calls once ----
narrowpeak_cols = ["chrom", "start", "end", "name", "score", "strand",
                   "signalValue", "pValue", "qValue", "summit"]
peaks_df = pd.read_csv(narrowpeak_f, sep="\t", header=None, names=narrowpeak_cols)
print(f"Loaded {len(peaks_df)} MACS3 peak calls")

bw = pyBigWig.open(bigwig_f)
bw_chroms = set(bw.chroms().keys())


def genomic_to_relative(genomic_pos, tss_genomic, strand):
    """Strand-aware genomic -> TSS-relative conversion, for anything
    pulled from RAW genomic coordinates (bigwig signal, MACS3 peaks,
    GTF exons) -- NOT for FIMO hit positions (already strand-corrected
    at extraction time, see module docstring)."""
    if strand == "+":
        return genomic_pos - tss_genomic
    else:
        return tss_genomic - genomic_pos


def get_window_info(gene):
    """chrom, bed_start, bed_end, strand, tss_genomic for this gene's
    own promoter window (defines the plotted x-range)."""
    if gene not in promoters_bed.index:
        return None
    row = promoters_bed.loc[gene]
    chrom, bed_start, bed_end, strand = row["chrom"], int(row["start"]), int(row["end"]), row["strand"]
    tss_genomic = bed_start + UPSTREAM if strand == "+" else bed_start + DOWNSTREAM
    return {"chrom": chrom, "bed_start": bed_start, "bed_end": bed_end, "strand": strand, "tss_genomic": tss_genomic}


def get_gene_atac_signal(window):
    chrom, bed_start, bed_end, strand = window["chrom"], window["bed_start"], window["bed_end"], window["strand"]
    if chrom not in bw_chroms:
        return None, None
    values = bw.values(chrom, bed_start, bed_end, numpy=True)
    values = np.nan_to_num(values, nan=0.0)
    genomic_positions = np.arange(bed_start, bed_end)

    if strand == "+":
        relative_positions = genomic_positions - window["tss_genomic"]
    else:
        relative_positions = window["tss_genomic"] - genomic_positions
        relative_positions = relative_positions[::-1]
        values = values[::-1]
    return relative_positions, values


def get_peaks_in_window(window):
    """MACS3 peaks overlapping this window, as (rel_start, rel_stop)."""
    chrom, bed_start, bed_end = window["chrom"], window["bed_start"], window["bed_end"]
    overlapping = peaks_df[
        (peaks_df["chrom"] == chrom) & (peaks_df["start"] < bed_end) & (peaks_df["end"] > bed_start)
    ]
    spans = []
    for _, prow in overlapping.iterrows():
        r1 = genomic_to_relative(prow["start"], window["tss_genomic"], window["strand"])
        r2 = genomic_to_relative(prow["end"], window["tss_genomic"], window["strand"])
        spans.append((min(r1, r2), max(r1, r2)))
    return spans


def get_exon_spans_for_gene(gname, window):
    """Exon spans for gname's own model, converted to THIS window's
    TSS-relative coordinate system (uses the FLAGGED gene's TSS as the
    x=0 reference point for every gene drawn, since they all share one
    x-axis)."""
    model = all_gene_models.get(gname)
    if model is None or len(model["exons"]) == 0:
        return None
    spans = []
    for (ex_start, ex_end) in model["exons"]:
        r1 = genomic_to_relative(ex_start, window["tss_genomic"], window["strand"])
        r2 = genomic_to_relative(ex_end, window["tss_genomic"], window["strand"])
        spans.append((min(r1, r2), max(r1, r2)))
    return sorted(spans)


def spans_overlap(a_start, a_stop, spans):
    """True if [a_start, a_stop] overlaps ANY interval in spans."""
    for (s, e) in spans:
        if a_start < e and a_stop > s:
            return True
    return False


def to_relative(pos):
    """FIMO in-sequence position -> TSS-relative position. NO strand
    handling needed -- see module docstring."""
    return pos - TSS_FASTA_POS


# =============================================================================
# Summary CSV: for every co-occurrence pair, flag whether the STAT3
# hit and the other-TF hit each fall inside a MACS3 peak
# =============================================================================
peak_flag_rows = []
for gene, gdf in proximity_df.groupby("gene"):
    window = get_window_info(gene)
    if window is None:
        continue
    peak_spans_genomic_rel = get_peaks_in_window(window)
    for _, row in gdf.iterrows():
        stat_r1, stat_r2 = to_relative(row["stat_start"]), to_relative(row["stat_stop"])
        other_r1, other_r2 = to_relative(row["other_start"]), to_relative(row["other_stop"])
        peak_flag_rows.append({
            **row.to_dict(),
            "stat_in_macs3_peak": spans_overlap(stat_r1, stat_r2, peak_spans_genomic_rel),
            "other_in_macs3_peak": spans_overlap(other_r1, other_r2, peak_spans_genomic_rel),
        })

peak_flag_df = pd.DataFrame(peak_flag_rows)
peak_flag_df["both_in_macs3_peak"] = peak_flag_df["stat_in_macs3_peak"] & peak_flag_df["other_in_macs3_peak"]
peak_flag_csv = f"{out_dir}/cooccurrence_macs3_peak_overlap.csv"
peak_flag_df.to_csv(peak_flag_csv, index=False)
print(f"\nSaved: {peak_flag_csv}")
print(f"  {peak_flag_df['both_in_macs3_peak'].sum()}/{len(peak_flag_df)} pairs have BOTH motifs inside a MACS3 peak")

# =============================================================================
# One output folder per TF, one 4-panel plot per co-occurring gene
# =============================================================================
for family_label, family_df in proximity_df.groupby("other_family"):
    family_out_dir = f"{out_dir}/{family_label}_cooccurrence_tracks"
    os.makedirs(family_out_dir, exist_ok=True)

    genes_this_family = sorted(family_df["gene"].unique())
    print(f"\n{family_label}: {len(genes_this_family)} genes with a co-occurrence")

    for gene in genes_this_family:
        window = get_window_info(gene)
        if window is None:
            print(f"  WARNING: {gene} not in promoters BED, skipping")
            continue

        relative_positions, values = get_gene_atac_signal(window)
        if relative_positions is None:
            print(f"  WARNING: no ATAC signal for {gene}, skipping")
            continue

        gene_pairs = family_df[family_df["gene"] == gene]
        other_tf = gene_pairs["other_motif_id"].iloc[0].split(".")[0]
        tf_color = TF_COLOR.get(other_tf.upper(), "#33A02C")

        peak_spans = get_peaks_in_window(window)
        overlapping_genes = find_overlapping_genes(window["chrom"], window["bed_start"], window["bed_end"])
        # flagged gene always first, so it's never cut off by MAX_GENE_ROWS
        overlapping_genes = [gene] + [g for g in overlapping_genes if g != gene]
        n_shown = min(len(overlapping_genes), MAX_GENE_ROWS)
        n_extra = len(overlapping_genes) - n_shown

        fig, (ax_sig, ax_peak, ax_gene, ax_motif) = plt.subplots(
            4, 1, figsize=(12, 2 + 0.5 * n_shown + 4), sharex=True,
            gridspec_kw={"height_ratios": [3, 0.4, 0.3 * n_shown + 0.3, 1], "hspace": 0.1}
        )

        # -- Panel 1: ATAC signal --
        ax_sig.fill_between(relative_positions, values, color=ATAC_FILL_COLOR, alpha=0.85, linewidth=0, clip_on=False)
        ax_sig.axvline(0, color="#777777", linewidth=1.0, linestyle="--", zorder=3)
        ax_sig.set_ylabel("ATAC (RPGC)")
        ax_sig.set_title(f"{gene} \u2014 STAT3 x {other_tf} co-occurrence", fontsize=11)

        # -- Panel 2: MACS3 peak calls --
        for (s, e) in peak_spans:
            ax_peak.axvspan(s, e, color=PEAK_COLOR, alpha=0.6, zorder=2)
        ax_peak.set_ylim(-1, 1)
        ax_peak.set_yticks([])
        ax_peak.set_ylabel("MACS3\npeaks", fontsize=7)

        # -- Panel 3: gene model(s) -- ALL genes overlapping this
        # window, one row each, flagged gene's label in RED, others
        # grey. y-limits computed EXACTLY from the actual box geometry
        # (box half-height 0.25) plus a small margin, so the bottom
        # row's exon box is never clipped by the axes limits. Labels
        # use the y-axis transform (x in AXES FRACTION, y in DATA
        # coordinates) so they render just left of the plot area
        # regardless of the x-axis's actual data range -- placing them
        # directly in data coordinates outside xlim (the old approach)
        # put them outside the visible area entirely.
        row_height = 1.0
        box_half_height = 0.25
        for row_i, gname in enumerate(overlapping_genes[:n_shown]):
            y0 = -(row_i * row_height) - row_height / 2
            spans = get_exon_spans_for_gene(gname, window)
            label_color = FLAGGED_GENE_LABEL_COLOR if gname == gene else OTHER_GENE_LABEL_COLOR
            if spans is not None:
                g_lo = min(s for s, _ in spans)
                g_hi = max(e for _, e in spans)
                ax_gene.plot([g_lo, g_hi], [y0, y0], color=INTRON_COLOR, linewidth=1.0, zorder=1)
                for (s, e) in spans:
                    ax_gene.add_patch(mpatches.Rectangle((s, y0 - box_half_height), e - s, 2 * box_half_height,
                                                           facecolor=EXON_COLOR, edgecolor="none", zorder=2))
            ax_gene.text(-0.01, y0, gname, transform=ax_gene.get_yaxis_transform(),
                          color=label_color, fontsize=7, ha="right", va="center",
                          fontweight="bold" if gname == gene else "normal", clip_on=False)
        if n_extra > 0:
            ax_gene.text(1.0, 0.02, f"+{n_extra} more gene(s) in window (not shown)",
                         transform=ax_gene.transAxes, ha="right", va="bottom", fontsize=6, color="grey")

        y0_first = -row_height / 2
        y0_last = -(n_shown - 1) * row_height - row_height / 2
        margin = 0.3
        ax_gene.set_ylim(y0_last - box_half_height - margin, y0_first + box_half_height + margin)
        ax_gene.set_yticks([])
        ax_gene.set_xlim(-UPSTREAM, DOWNSTREAM)

        # -- Panel 4: STAT3/TF co-occurrence motif track -- bordered
        # (black edge) if that specific motif falls inside a MACS3 peak
        ax_motif.axhline(0, color="black", linewidth=1.0, zorder=1)
        ax_motif.axvline(0, color="grey", linewidth=1.0, linestyle="--", zorder=2)
        for _, pair_row in gene_pairs.iterrows():
            stat_rel_start = to_relative(pair_row["stat_start"])
            stat_rel_stop = to_relative(pair_row["stat_stop"])
            other_rel_start = to_relative(pair_row["other_start"])
            other_rel_stop = to_relative(pair_row["other_stop"])

            stat_in_peak = spans_overlap(stat_rel_start, stat_rel_stop, peak_spans)
            other_in_peak = spans_overlap(other_rel_start, other_rel_stop, peak_spans)

            ax_motif.axvspan(stat_rel_start, stat_rel_stop, facecolor=STAT_COLOR, alpha=0.6,
                              ymin=0.15, ymax=0.85, zorder=3,
                              edgecolor="black" if stat_in_peak else "none", linewidth=1.5 if stat_in_peak else 0)
            ax_motif.axvspan(other_rel_start, other_rel_stop, facecolor=tf_color, alpha=0.6,
                              ymin=0.15, ymax=0.85, zorder=3,
                              edgecolor="black" if other_in_peak else "none", linewidth=1.5 if other_in_peak else 0)
        ax_motif.set_ylim(-1, 1)
        ax_motif.set_yticks([])
        ax_motif.set_xlabel("Position relative to TSS (bp)")

        legend_handles = [
            mpatches.Patch(color=STAT_COLOR, alpha=0.6, label="STAT3"),
            mpatches.Patch(color=tf_color, alpha=0.6, label=other_tf),
            mpatches.Patch(facecolor="none", edgecolor="black", linewidth=1.5, label="in MACS3 peak"),
        ]
        ax_motif.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                         fontsize=8, frameon=False)

        ax_sig.set_xlim(-UPSTREAM, DOWNSTREAM)
        # plt.tight_layout() removed -- warns on every plot because of the
        # legend placed outside the axes via bbox_to_anchor, which it can't
        # account for; savefig's bbox_inches="tight" below already handles
        # cropping correctly around that legend regardless

        safe_gene = gene.replace("/", "_")
        png_f = f"{family_out_dir}/{safe_gene}_ATAC_signal_STAT3_{other_tf}_track.png"
        pdf_f = f"{family_out_dir}/{safe_gene}_ATAC_signal_STAT3_{other_tf}_track.pdf"
        plt.savefig(png_f, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_f, bbox_inches="tight")  # PDF instead of SVG -- confirmed more reliable
                                                    # for Illustrator (SVG's <use>/<defs> pattern
                                                    # for fill_between wasn't rendering there)
        plt.close(fig)

    print(f"  Saved {len(genes_this_family)} plots to: {family_out_dir}")

bw.close()
print("\nDone.")