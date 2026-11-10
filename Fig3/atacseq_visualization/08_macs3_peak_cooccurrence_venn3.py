"""
Fig3 ATAC-seq visualization, step 8: 3-way Venn of genes where BOTH
the STAT3 hit and the other-TF hit fall inside a MACS3 peak
(both_in_macs3_peak == True), from
06_cooccurrence_ATAC_signal_tracks.py's saved
cooccurrence_macs3_peak_overlap.csv -- one set each for ETV3, ERF, TFEB.

Requires matplotlib-venn: pip install matplotlib-venn --break-system-packages
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# ---- data sources ----
peak_flag_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/cooccurrence_macs3_peak_overlap.csv"
out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

TFS = ["ETV3", "ERF", "TFEB"]  # ARID3A excluded -- no FIMO hits yet (pending threshold fix)

df = pd.read_csv(peak_flag_f)
df["other_tf"] = df["other_motif_id"].str.split(".").str[0].str.upper()

both_df = df[df["both_in_macs3_peak"]]
print(f"{len(both_df)}/{len(df)} total pairs have both motifs in a MACS3 peak")

tf_gene_sets = {}
for tf in TFS:
    genes = set(both_df.loc[both_df["other_tf"] == tf, "gene"])
    tf_gene_sets[tf] = genes
    print(f"  {tf}: {len(genes)} genes")

set_a, set_b, set_c = tf_gene_sets[TFS[0]], tf_gene_sets[TFS[1]], tf_gene_sets[TFS[2]]

VENN_COLORS = ("#E69F00", "#56B4E9", "#009E73")  # Okabe-Ito, same convention as oy_venn3_stat3_erf_arid3a.py

fig, ax = plt.subplots(figsize=(7, 7))
v = venn3([set_a, set_b, set_c], set_labels=tuple(TFS), set_colors=VENN_COLORS, alpha=0.7, ax=ax)

ax.set_title("Genes with STAT3 x TF co-occurrence, BOTH motifs inside a MACS3 peak")
plt.tight_layout()

png_f = f"{out_dir}/macs3_peak_cooccurrence_venn3.png"
svg_f = f"{out_dir}/macs3_peak_cooccurrence_venn3.svg"
plt.savefig(png_f, dpi=300, bbox_inches="tight")
plt.savefig(svg_f, bbox_inches="tight")
plt.show()
print(f"\nSaved: {png_f}")

region_labels = {
    "100": f"{TFS[0]} only", "010": f"{TFS[1]} only", "001": f"{TFS[2]} only",
    "110": f"{TFS[0]} & {TFS[1]} only", "101": f"{TFS[0]} & {TFS[2]} only",
    "011": f"{TFS[1]} & {TFS[2]} only", "111": "all three",
}

# actual region membership, not just counts -- derived directly from
# the same set logic venn3 uses internally (A-only = A minus B minus C, etc.)
region_genes = {
    "100": set_a - set_b - set_c,
    "010": set_b - set_a - set_c,
    "001": set_c - set_a - set_b,
    "110": (set_a & set_b) - set_c,
    "101": (set_a & set_c) - set_b,
    "011": (set_b & set_c) - set_a,
    "111": set_a & set_b & set_c,
}

print("\nRegion counts and genes:")
rows = []
for code, label in region_labels.items():
    genes = sorted(region_genes[code])
    print(f"  {label}: n={len(genes)}")
    print(f"    {genes}")
    for g in genes:
        rows.append({"region": label, "gene": g})

region_df = pd.DataFrame(rows)
region_csv = f"{out_dir}/macs3_peak_cooccurrence_venn3_region_genes.csv"
region_df.to_csv(region_csv, index=False)
print(f"\nSaved: {region_csv}")

print("\nDone.")