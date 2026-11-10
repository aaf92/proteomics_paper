"""
Fig3 ATAC-seq visualization, step 3: extract the ATAC-seq signal
(bigwig, from run_bamCoverage.sh) for Il4ra's 8000_TSS_8000 window,
and combine it with the STAT motif track (from
02_il4ra_promoter_STAT_motif_track.py) into one stacked figure --
signal on top, motif highlights below, SHARED x-axis (position
relative to TSS), so the two tracks line up directly.

Requires pyBigWig: pip install pyBigWig --break-system-packages

Il4ra is + strand, so genomic position maps directly onto
"position relative to TSS" with no flip -- if this script is ever
reused for a - strand gene, the signal array and coordinates would
need to be reversed to keep 5'->3' orientation consistent with the
motif track's own convention (which already handles strand correctly
via the bedtools -s reverse-complementing at the FASTA-extraction
step).
"""

import os
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- data sources ----
bigwig_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/bigwig/IL10_4H_merged.bw"
motif_hits_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/Il4ra_STAT_motif_hits_8000_TSS_8000.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

GENE = "Il4ra"
UPSTREAM = 8000
DOWNSTREAM = 8000

# ---- Il4ra's 8000_TSS_8000 promoter window (same as step 2) ----
CHROM = "chr7"
BED_START = 125544270  # 0-based
BED_END = 125560270
STRAND = "+"
assert STRAND == "+", ("this script assumes + strand (direct coordinate mapping, no flip) -- "
                        "generalize (reverse the signal array and recompute relative_positions) "
                        "if reused for a - strand gene")
assert BED_END - BED_START == UPSTREAM + DOWNSTREAM

TSS_GENOMIC = BED_START + UPSTREAM

if not os.path.exists(bigwig_f):
    raise SystemExit(f"Bigwig not found at {bigwig_f} -- confirm run_bamCoverage.sh finished "
                      f"and check the actual output filename (out_name may differ from 'UT_merged').")

# ---- load the ATAC bigwig signal for this exact window ----
bw = pyBigWig.open(bigwig_f)
chrom_names = list(bw.chroms().keys())
print(f"Bigwig has {len(chrom_names)} chroms, e.g.: {chrom_names[:5]}")
if CHROM not in bw.chroms():
    bw.close()
    raise SystemExit(f"Chrom '{CHROM}' not found in bigwig -- check naming convention "
                      f"(e.g. 'chr7' vs '7'). Available: {chrom_names[:10]}")

values = bw.values(CHROM, BED_START, BED_END, numpy=True)
values = np.nan_to_num(values, nan=0.0)  # bigwig gaps (no coverage) come back as NaN
bw.close()

genomic_positions = np.arange(BED_START, BED_END)
relative_positions = genomic_positions - TSS_GENOMIC  # negative=upstream, positive=downstream (+ strand)

print(f"Signal extracted: {len(values)} bp, window {CHROM}:{BED_START}-{BED_END}")
print(f"Signal range: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

# ---- load the STAT motif hits (from step 2) ----
if not os.path.exists(motif_hits_f):
    raise SystemExit(f"Motif hits CSV not found at {motif_hits_f} -- run "
                      f"02_il4ra_promoter_STAT_motif_track.py first.")

motif_hits = pd.read_csv(motif_hits_f)
unique_motifs = sorted(motif_hits["motif_id"].unique())
palette = plt.cm.tab10.colors
motif_color = {m: palette[i % len(palette)] for i, m in enumerate(unique_motifs)}

# =============================================================================
# Combined stacked figure: ATAC signal track on top, STAT motif track
# below, SHARED x-axis (position relative to TSS)
# =============================================================================
fig, (ax_sig, ax_motif) = plt.subplots(
    2, 1, figsize=(14, 5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
)

# -- ATAC signal track --
ax_sig.fill_between(relative_positions, values, color="#4477AA", alpha=0.7, linewidth=0)
ax_sig.axvline(0, color="#B2182B", linewidth=1.2, linestyle="--", zorder=3)
ax_sig.set_ylabel("ATAC signal (CPM)")
ax_sig.set_title(f"{GENE} {UPSTREAM}_TSS_{DOWNSTREAM} -- ATAC-seq signal (resting BMDM) and STAT motif hits")

# -- STAT motif track (same drawing logic as step 2) --
ax_motif.axhline(0, color="black", linewidth=1.0, zorder=1)
ax_motif.axvline(0, color="#B2182B", linewidth=1.2, linestyle="--", zorder=2)
for _, row in motif_hits.iterrows():
    ax_motif.axvspan(row["relative_to_tss_start"], row["relative_to_tss_stop"],
                      color=motif_color[row["motif_id"]], alpha=0.6, ymin=0.15, ymax=0.85, zorder=3)
ax_motif.set_ylim(-1, 1)
ax_motif.set_yticks([])
ax_motif.set_xlabel(f"Position relative to {GENE} TSS (bp)")

legend_handles = [mpatches.Patch(color=motif_color[m], alpha=0.6, label=m) for m in unique_motifs]
legend_handles.append(plt.Line2D([0], [0], color="#B2182B", linestyle="--", label="TSS"))
ax_motif.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)

ax_sig.set_xlim(-UPSTREAM, DOWNSTREAM)

plt.tight_layout()

combined_png = f"{out_dir}/{GENE}_ATAC_signal_STAT_motif_combined_track.png"
combined_svg = f"{out_dir}/{GENE}_ATAC_signal_STAT_motif_combined_track.svg"
plt.savefig(combined_png, dpi=300, bbox_inches="tight")
plt.savefig(combined_svg, bbox_inches="tight")
plt.show()

print(f"\nSaved: {combined_png}")
print("\nDone.")
