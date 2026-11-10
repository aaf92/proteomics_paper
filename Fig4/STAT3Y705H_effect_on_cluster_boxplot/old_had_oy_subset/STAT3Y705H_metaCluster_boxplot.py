"""
Stat3Y705H orange U yellow genes, by meta-cluster (IL6 + IL10
combined). NTC vs Stat3Y705H-edit ctrl-centered expression, boxplots
side by side per meta-cluster/cytokine (not overlaid) -- paired
Wilcoxon signed-rank test (NTC vs edit, matched by gene) shown above
each pair instead of a raw sample count.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os

# raw ctrl-centered expression matrix -- same source file as the heatmap notebook
data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq/be_effect_analysis/out_files/il6_il10_NTC_ctrl_centered_combatCorrected.csv"

# meta-cluster gene assignments, already saved by the heatmap notebook
meta_cluster_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/Stat3Y705H_only/out_files/common_degs_14clusters_8metaClusters_ordered.csv"

# Stat3Y705H's own orange/yellow classification, per cytokine
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/STAT3Y705H_effect_on_cluster_boxplot"
os.makedirs(out_dir, exist_ok=True)

data = pd.read_csv(data_f, header=0, index_col=0)

data_cols = [
    'B1_PBS_NTC_R4', 'B1_PBS_NTC_R5', 'B2_PBS_NTC_R2', 'B2_PBS_NTC_R3', 'B2_PBS_NTC_R5', 'B2_PBS_NTC_R6',
    'B1_PBS_Stat3Y705H_R1', 'B1_PBS_Stat3Y705H_R2', 'B1_PBS_Stat3Y705H_R3', 'B2_PBS_Stat3Y705H_R1', 'B2_PBS_Stat3Y705H_R3',
    'B1_IL6_NTC_R1', 'B1_IL6_NTC_R2', 'B1_IL6_NTC_R3', 'B1_IL6_NTC_R4', 'B1_IL6_NTC_R5', 'B1_IL6_NTC_R6', 'B2_IL6_NTC_R3', 'B2_IL6_NTC_R4', 'B2_IL6_NTC_R5', 'B2_IL6_NTC_R6',
    'B1_IL10_NTC_R1', 'B1_IL10_NTC_R2', 'B1_IL10_NTC_R3', 'B1_IL10_NTC_R4', 'B1_IL10_NTC_R5', 'B1_IL10_NTC_R6', 'B2_IL10_NTC_R4', 'B2_IL10_NTC_R5', 'B2_IL10_NTC_R6',
    'B1_IL6_Stat3Y705H_R1', 'B1_IL6_Stat3Y705H_R2', 'B1_IL6_Stat3Y705H_R3', 'B2_IL6_Stat3Y705H_R1', 'B2_IL6_Stat3Y705H_R2', 'B2_IL6_Stat3Y705H_R3',
    'B1_IL10_Stat3Y705H_R1', 'B1_IL10_Stat3Y705H_R2', 'B1_IL10_Stat3Y705H_R3', 'B2_IL10_Stat3Y705H_R1', 'B2_IL10_Stat3Y705H_R2', 'B2_IL10_Stat3Y705H_R3'
]

print(data.shape)

meta_clus_df = pd.read_csv(meta_cluster_f)
gene_to_meta_cluster = meta_clus_df.set_index("gene")["meta_cluster"].to_dict()
print(f"{len(gene_to_meta_cluster)} genes have a meta-cluster assignment")

BASE_CATEGORIES = ["orange", "yellow"]
STIMS = ["IL6", "IL10"]
STIM_COLOR = {"IL6": "#4477AA", "IL10": "#EE6677"}

box_data = {}            # (stim, group, cluster) -> np.array of values
all_clusters = set()

for STIM in STIMS:
    class_f = f"{interaction_dir_base}/Stat3Y705H/Stat3Y705H_{STIM}_fourCategory_classification_data.csv"
    class_df = pd.read_csv(class_f, index_col=0)
    stat3_oy_genes = set(class_df.index[class_df["category"].isin(BASE_CATEGORIES)])

    # genes for this stim: Stat3 orange_yellow AND has a meta-cluster assignment AND present in data
    eligible_genes = [g for g in stat3_oy_genes if g in gene_to_meta_cluster and g in data.index]

    genes_by_cluster = {}
    for gene in eligible_genes:
        genes_by_cluster.setdefault(gene_to_meta_cluster[gene], []).append(gene)

    # per-group column averages, computed only for the genes needed this stim
    ntc_cols = [c for c in data_cols if re.match(rf"^B[12]_{STIM}_NTC_R\d+$", c)]
    edit_cols = [c for c in data_cols if re.match(rf"^B[12]_{STIM}_Stat3Y705H_R\d+$", c)]

    for cl, genes in genes_by_cluster.items():
        all_clusters.add(cl)
        box_data[(STIM, "NTC", cl)] = data.loc[genes, ntc_cols].mean(axis=1).values
        box_data[(STIM, "Stat3Y705H", cl)] = data.loc[genes, edit_cols].mean(axis=1).values

    print(f"{STIM}: {sum(len(g) for g in genes_by_cluster.values())} gene-cluster assignments "
          f"across {len(genes_by_cluster)} meta-clusters "
          f"({len(ntc_cols)} NTC reps, {len(edit_cols)} Stat3Y705H-edit reps)")


from scipy.stats import wilcoxon

cluster_order = sorted(all_clusters)
n_clusters = len(cluster_order)

# x positions: within each cluster, each stim gets a slot with NTC and
# Stat3Y705H-edit as SEPARATE, side-by-side boxes (not overlaid at the
# same position). Built by iterating and advancing a running cursor,
# rather than a precomputed formula -- simpler to get right and easier
# to adjust spacing later.
BOX_WIDTH = 0.32
BOX_GAP = 0.06      # gap between NTC and edit box within one stim slot
STIM_GAP = 0.35     # gap between stim slots within a cluster
CLUSTER_GAP = 1.0   # gap between clusters

ntc_pos = {}    # (stim, cluster) -> x position
edit_pos = {}   # (stim, cluster) -> x position
pair_center = {}  # (stim, cluster) -> midpoint, for the p-value annotation
cluster_center = {}  # cluster -> midpoint, for the x tick label

x_cursor = 0.0
for cl in cluster_order:
    cluster_start = x_cursor
    for STIM in STIMS:
        n_pos = x_cursor + BOX_WIDTH / 2
        e_pos = n_pos + BOX_WIDTH + BOX_GAP
        ntc_pos[(STIM, cl)] = n_pos
        edit_pos[(STIM, cl)] = e_pos
        pair_center[(STIM, cl)] = (n_pos + e_pos) / 2
        x_cursor = e_pos + BOX_WIDTH / 2 + STIM_GAP
    cluster_end = x_cursor - STIM_GAP
    cluster_center[cl] = (cluster_start + cluster_end) / 2
    x_cursor = cluster_end + CLUSTER_GAP

fig, ax = plt.subplots(figsize=(2.6 * n_clusters + 2, 6))

for cl in cluster_order:
    for STIM in STIMS:
        color = STIM_COLOR[STIM]
        n_pos = ntc_pos[(STIM, cl)]
        e_pos = edit_pos[(STIM, cl)]

        ntc_vals = box_data.get((STIM, "NTC", cl))
        edit_vals = box_data.get((STIM, "Stat3Y705H", cl))

        # NTC: translucent solid fill
        if ntc_vals is not None and len(ntc_vals) > 0:
            bp_ntc = ax.boxplot([ntc_vals], positions=[n_pos], widths=BOX_WIDTH,
                                 patch_artist=True, showfliers=False, zorder=2)
            for patch in bp_ntc["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp_ntc[element]:
                    line.set_color(color)
                    line.set_alpha(0.6)

        # Stat3Y705H-edit: unfilled + hatched, now at its OWN position
        # right next to NTC's, instead of overlaid on top of it
        if edit_vals is not None and len(edit_vals) > 0:
            bp_edit = ax.boxplot([edit_vals], positions=[e_pos], widths=BOX_WIDTH,
                                  patch_artist=True, showfliers=False, zorder=3)
            for patch in bp_edit["boxes"]:
                patch.set_facecolor("none")
                patch.set_edgecolor(color)
                patch.set_hatch("///")
                patch.set_linewidth(1.4)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp_edit[element]:
                    line.set_color(color)
                    line.set_linewidth(1.4)

        # paired Wilcoxon signed-rank test, NTC vs Stat3Y705H-edit --
        # ntc_vals/edit_vals are per-gene means built from the SAME gene
        # list in the SAME order, so they're genuinely paired by gene
        if ntc_vals is not None and edit_vals is not None and len(ntc_vals) == len(edit_vals) and len(ntc_vals) > 0:
            try:
                _, pval = wilcoxon(ntc_vals, edit_vals)
                p_text = f"p={pval:.2g}" if pval >= 1e-4 else f"p={pval:.1e}"
            except ValueError:
                # e.g. all differences are zero, or n too small for wilcoxon
                p_text = "p=n/a"

            y_top = ax.get_ylim()[1]
            ax.annotate(p_text, (pair_center[(STIM, cl)], y_top), ha="center", va="bottom",
                        fontsize=7, color=color)

ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)

ax.set_xticks([cluster_center[cl] for cl in cluster_order])
ax.set_xticklabels([f"{cl}" for cl in cluster_order])
ax.set_xlabel("meta-cluster")
ax.set_ylabel("ctrl-centered expression")
ax.set_title(
    "IL6 + IL10: NTC vs Stat3Y705H ctrl-centered expression, by meta-cluster\n"
    "Stat3Y705H orange\u222ayellow genes only (per stim); NTC = translucent fill, "
    "Stat3Y705H-edit = hashed, side by side; p = paired Wilcoxon signed-rank (NTC vs edit, by gene)"
)

legend_handles = [
    mpatches.Patch(facecolor=STIM_COLOR["IL6"], alpha=0.35, edgecolor=STIM_COLOR["IL6"], label="IL6 NTC"),
    mpatches.Patch(facecolor="none", edgecolor=STIM_COLOR["IL6"], hatch="///", label="IL6 Stat3Y705H-edit"),
    mpatches.Patch(facecolor=STIM_COLOR["IL10"], alpha=0.35, edgecolor=STIM_COLOR["IL10"], label="IL10 NTC"),
    mpatches.Patch(facecolor="none", edgecolor=STIM_COLOR["IL10"], hatch="///", label="IL10 Stat3Y705H-edit"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)

plt.tight_layout()
plt.savefig(f"{out_dir}/metaCluster_stat3OrangeYellow_IL6_IL10_NTCvsEdit_sideBySide_boxplot.png",
            dpi=300, bbox_inches="tight")
plt.savefig(f"{out_dir}/metaCluster_stat3OrangeYellow_IL6_IL10_NTCvsEdit_sideBySide_boxplot.svg",
            dpi=300, bbox_inches="tight")
plt.show()
