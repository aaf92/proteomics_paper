"""
Stat3Y705H's own cytokine-response DEG set (union of IL6_NTC and
IL10_NTC DEGs vs PBS_NTC), ctrl-centered expression heatmap with both the
original k-cluster row-color bar and the merged meta-cluster row-color
bar shown side by side. Stripped down from the exploratory notebook:
drops the intermediate/alternate heatmap variants (batch-averaged,
group-averaged, the k-selection sweep over k=1..14) and the meta-cluster
boxplot section entirely -- just the final heatmap plus the gene-level
cluster/meta-cluster assignment CSVs.
"""

import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster import hierarchy

# ---- directory and data file settings ----
data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq/be_effect_analysis/out_files/il6_il10_NTC_ctrl_centered_combatCorrected.csv"

il6_deg_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq_DEG_analysis/DEG_results/pairwise/IL6_NTC_vs_PBS_NTC/resultsOrdered_IL6_NTC_vs_PBS_NTC.txt"
il10_deg_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq_DEG_analysis/DEG_results/pairwise/IL10_NTC_vs_PBS_NTC/resultsOrdered_IL10_NTC_vs_PBS_NTC.txt"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/NTCdegs_STAT3Y705H_heatmap"
os.makedirs(out_dir, exist_ok=True)

# ---- load RNAseq data ----
data = pd.read_csv(data_f, header=0, index_col=0)

# cytokine-response DEGs in NTC cells: union of IL6_NTC_vs_PBS_NTC and IL10_NTC_vs_PBS_NTC (padj < 0.05)
padj_thresh = 0.05
il6_res = pd.read_csv(il6_deg_f, index_col=0)
il10_res = pd.read_csv(il10_deg_f, index_col=0)
il6_degs = il6_res[il6_res["adjusted_padj"] < padj_thresh].index.tolist()
il10_degs = il10_res[il10_res["adjusted_padj"] < padj_thresh].index.tolist()
degs = list(set(il6_degs) | set(il10_degs))
degs = [g for g in degs if g in data.index]

print(data.shape)
print(f"IL6 DEGs: {len(il6_degs)}, IL10 DEGs: {len(il10_degs)}, union in data: {len(degs)}")

# ---- NTC and Stat3Y705H samples only, both batches (B1/B2) ----
data_cols = [
    'B1_PBS_NTC_R4', 'B1_PBS_NTC_R5', 'B2_PBS_NTC_R2', 'B2_PBS_NTC_R3', 'B2_PBS_NTC_R5', 'B2_PBS_NTC_R6',
    'B1_PBS_Stat3Y705H_R1', 'B1_PBS_Stat3Y705H_R2', 'B1_PBS_Stat3Y705H_R3', 'B2_PBS_Stat3Y705H_R1', 'B2_PBS_Stat3Y705H_R3',
    'B1_IL6_NTC_R1', 'B1_IL6_NTC_R2', 'B1_IL6_NTC_R3', 'B1_IL6_NTC_R4', 'B1_IL6_NTC_R5', 'B1_IL6_NTC_R6', 'B2_IL6_NTC_R3', 'B2_IL6_NTC_R4', 'B2_IL6_NTC_R5', 'B2_IL6_NTC_R6',
    'B1_IL10_NTC_R1', 'B1_IL10_NTC_R2', 'B1_IL10_NTC_R3', 'B1_IL10_NTC_R4', 'B1_IL10_NTC_R5', 'B1_IL10_NTC_R6', 'B2_IL10_NTC_R4', 'B2_IL10_NTC_R5', 'B2_IL10_NTC_R6',
    'B1_IL6_Stat3Y705H_R1', 'B1_IL6_Stat3Y705H_R2', 'B1_IL6_Stat3Y705H_R3', 'B2_IL6_Stat3Y705H_R1', 'B2_IL6_Stat3Y705H_R2', 'B2_IL6_Stat3Y705H_R3',
    'B1_IL10_Stat3Y705H_R1', 'B1_IL10_Stat3Y705H_R2', 'B1_IL10_Stat3Y705H_R3', 'B2_IL10_Stat3Y705H_R1', 'B2_IL10_Stat3Y705H_R2', 'B2_IL10_Stat3Y705H_R3'
]

deg_data = data.loc[degs, data_cols]
print(deg_data.columns)

# ---- clustering scaffold: row-cluster once to get the linkage/ordering --
# this figure itself is NOT saved (only used to derive row_order and the
# linkage that fcluster runs on below); it's closed immediately after ----
c = sns.color_palette("RdBu_r", as_cmap=True)
rna_heatmap = sns.clustermap(deg_data, metric='correlation', method='complete', cmap=c, z_score=None,
                              row_cluster=True, col_cluster=False, vmin=-1.5, vmax=1.5,
                              yticklabels=False, xticklabels=True)
plt.close(rna_heatmap.fig)

row_order = rna_heatmap.dendrogram_row.reordered_ind
ordered_genes_idx = deg_data.index[row_order]
ordered_genes = pd.DataFrame({"gene": ordered_genes_idx})

# ---- k-cluster assignment (k chosen via prior exploration, fixed here) ----
k = 14
link = rna_heatmap.dendrogram_row.linkage
clus_assign = hierarchy.fcluster(link, k, criterion='maxclust')
network_pal = sns.color_palette("tab10", k)
clus_colors = np.ones((len(row_order), 3))
for i in range(len(row_order)):
    clus_colors[i, :] = network_pal[clus_assign[i] - 1]

# ---- meta-clusters: manually merge some of the k clusters above into
# coarser groups, based on visual similarity of their patterns in the
# heatmap ----
meta_cluster_map = {
    1: [1, 2, 3],
    2: [4],
    3: [5],
    4: [6, 7],
    5: [8, 9],
    6: [10, 11, 12],
    7: [13],
    8: [14],
}
cluster_to_meta = {c_: meta for meta, clusters in meta_cluster_map.items() for c_ in clusters}
n_meta = len(meta_cluster_map)

meta_assign = np.array([cluster_to_meta[c_] for c_ in clus_assign])

meta_pal = sns.color_palette("tab10", n_meta)
meta_colors = np.ones((len(meta_assign), 3))
for i in range(len(meta_assign)):
    meta_colors[i, :] = meta_pal[meta_assign[i] - 1]

# two side-by-side color bars: original k clusters vs. merged meta clusters
row_colors_df = pd.DataFrame(
    {
        f"{k} clusters": [mcolors.to_hex(col) for col in clus_colors],
        f"{n_meta} meta-clusters": [mcolors.to_hex(col) for col in meta_colors],
    },
    index=deg_data.index,
)

# =============================================================================
# FINAL heatmap: k-cluster and meta-cluster row-color bars side by side
# =============================================================================
c = sns.color_palette("RdBu_r", as_cmap=True)
temp_meta = sns.clustermap(
    deg_data, metric='correlation', method='complete', z_score=None, vmin=-1.5, vmax=1.5,
    cmap=c, row_colors=row_colors_df, col_cluster=False, figsize=(5.5, 9), yticklabels=False
)
temp_meta.ax_heatmap.set_xticklabels(
    temp_meta.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7
)

# rasterize ONLY the heatmap cells
temp_meta.ax_heatmap.collections[0].set_rasterized(True)

fig_out = f"{out_dir}/cytokine_ntc_heatmap_completeClustered_{n_meta}_metaClusters"
plt.savefig(f"{fig_out}.svg", format="svg", dpi=600)
plt.savefig(f"{fig_out}.png", format="png", dpi=600)
plt.show()

# =============================================================================
# save gene-level table with BOTH the original k-cluster and the merged
# meta-cluster assignment, in heatmap row order
# =============================================================================
meta_clus_df = pd.DataFrame({
    "gene": deg_data.index,
    "cluster": clus_assign,
    "cluster_color": [mcolors.to_hex(col) for col in clus_colors],
    "meta_cluster": meta_assign,
    "meta_cluster_color": [mcolors.to_hex(col) for col in meta_colors],
})
meta_clus_df = pd.merge(ordered_genes, meta_clus_df, how="left", on="gene")
meta_clus_df.to_csv(f"{out_dir}/common_degs_{k}clusters_{n_meta}metaClusters_ordered.csv", index=False)

print(f"\nSaved: {fig_out}.png / .svg")
print(f"Saved: {out_dir}/common_degs_{k}clusters_{n_meta}metaClusters_ordered.csv")
print("\nDone.")
