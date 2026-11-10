"""
Fig3 ATAC-seq visualization, step 2: extract ALL STAT motif hits
within Il4ra's 8000_TSS_8000 promoter window from the genome-wide FIMO
results, convert each hit's in-sequence position to a genomic
coordinate and a TSS-relative position, and plot them as a highlighted
motif track along the promoter window.

Window: Il4ra, 8000bp upstream + 8000bp downstream of TSS (from
8000_TSS_8000_promoters_mm10.bed): chr7:125544270-125560270, + strand.
Since Il4ra is + strand, the extracted FASTA was NOT reverse-
complemented, so genomic position = bed_start + fimo_position (1-based
FIMO position maps directly onto the genomic coordinate, same
convention used for the individual hits worked through manually).

This produces the motif track ONLY -- no ATAC-seq signal yet (that
gets layered in as a separate panel once the bigwig/peak file paths
are available). Structured with its own standalone axes so an ATAC
track can be stacked below it later via a shared x-axis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- data sources ----
fimo_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results/8000_TSS_8000_stats_all/fimo.tsv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

GENE = "Il4ra"
UPSTREAM = 8000
DOWNSTREAM = 8000

# ---- Il4ra's 8000_TSS_8000 promoter window (from
# 8000_TSS_8000_promoters_mm10.bed) ----
CHROM = "chr7"
BED_START = 125544270  # 0-based
BED_END = 125560270
STRAND = "+"
assert BED_END - BED_START == UPSTREAM + DOWNSTREAM, "window size doesn't match UPSTREAM+DOWNSTREAM"

TSS_FASTA_POS = UPSTREAM + 1  # 1-based position of the TSS within the extracted sequence (8001)
TSS_GENOMIC = BED_START + UPSTREAM  # genomic coordinate of the TSS

# ---- load FIMO results, filter to this gene ----
fimo_df = pd.read_csv(fimo_f, sep="\t", comment="#")
print(f"Total FIMO rows loaded: {len(fimo_df)}")

gene_hits = fimo_df[fimo_df["sequence_name"] == GENE].copy()
print(f"STAT motif hits in {GENE}'s {UPSTREAM}_TSS_{DOWNSTREAM} window: {len(gene_hits)}")

if len(gene_hits) == 0:
    raise SystemExit(f"No hits found for sequence_name == '{GENE}' -- check the exact name/capitalization "
                      f"used in this FIMO run (e.g. print fimo_df['sequence_name'].unique()[:20]).")

# ---- convert each hit's in-sequence position to genomic + TSS-relative coordinates ----
# Il4ra is + strand -- sequence was NOT reverse-complemented, so genomic
# position maps directly: genomic = BED_START + fimo_position
gene_hits["genomic_start"] = BED_START + gene_hits["start"]
gene_hits["genomic_stop"] = BED_START + gene_hits["stop"]
gene_hits["relative_to_tss_start"] = gene_hits["start"] - TSS_FASTA_POS
gene_hits["relative_to_tss_stop"] = gene_hits["stop"] - TSS_FASTA_POS

gene_hits = gene_hits.sort_values("start").reset_index(drop=True)

print(f"\nAll {len(gene_hits)} STAT hits in {GENE}'s promoter window "
      f"(relative_to_tss: negative = upstream, positive = downstream):")
print(gene_hits[["motif_id", "start", "stop", "strand", "score", "p-value", "q-value",
                  "genomic_start", "genomic_stop", "relative_to_tss_start", "relative_to_tss_stop"]]
      .to_string(index=False))

hits_csv = f"{out_dir}/{GENE}_STAT_motif_hits_{UPSTREAM}_TSS_{DOWNSTREAM}.csv"
gene_hits.to_csv(hits_csv, index=False)
print(f"\nSaved: {hits_csv}")

# =============================================================================
# Motif track plot -- x-axis = position relative to TSS, TSS marked at
# 0, each STAT hit highlighted as a colored span. Motifs are colored by
# motif_id in case this file ever contains more than one STAT paralog's
# PWM (currently all STAT3, per the prior conversation).
# =============================================================================
unique_motifs = sorted(gene_hits["motif_id"].unique())
palette = plt.cm.tab10.colors
motif_color = {m: palette[i % len(palette)] for i, m in enumerate(unique_motifs)}

fig, ax = plt.subplots(figsize=(14, 2.2))

ax.axhline(0, color="black", linewidth=1.2, zorder=1)  # the promoter window itself, drawn as a line
ax.axvline(0, color="#B2182B", linewidth=1.2, linestyle="--", zorder=2, label="TSS")

for _, row in gene_hits.iterrows():
    ax.axvspan(row["relative_to_tss_start"], row["relative_to_tss_stop"],
               color=motif_color[row["motif_id"]], alpha=0.6, ymin=0.15, ymax=0.85, zorder=3)

ax.set_xlim(-UPSTREAM, DOWNSTREAM)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xlabel(f"Position relative to {GENE} TSS (bp)")
ax.set_title(f"{GENE} {UPSTREAM}_TSS_{DOWNSTREAM} promoter window -- STAT motif hits (n={len(gene_hits)})")

legend_handles = [mpatches.Patch(color=motif_color[m], alpha=0.6, label=m) for m in unique_motifs]
legend_handles.append(plt.Line2D([0], [0], color="#B2182B", linestyle="--", label="TSS"))
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)

plt.tight_layout()

track_png = f"{out_dir}/{GENE}_STAT_motif_track.png"
track_svg = f"{out_dir}/{GENE}_STAT_motif_track.svg"
plt.savefig(track_png, dpi=300, bbox_inches="tight")
plt.savefig(track_svg, bbox_inches="tight")
plt.show()

print(f"\nSaved: {track_png}")
print("\nDone.")
