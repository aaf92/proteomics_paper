"""
Fig4: same underlying data/calculation as STAT3Y705H_metaCluster5v7_boxplot.py
(meta-cluster 5 vs meta-cluster 7, |stim_delta| = Stat3 edit vs
cytokine-matched NTC, generalized for Stat3 spanning multiple batches),
but IL6 and IL10 shown together on ONE plot -- side-by-side boxes per
meta-cluster, colored by cytokine, same layout convention as the
earlier STAT3Y705H_metaCluster_boxplot.py's cytokine_NTC/cytokine_edit
side-by-side grouping.

Two-sided Mann-Whitney U (meta-cluster 5 vs 7) computed SEPARATELY for
IL6 and IL10, each annotated with its own p-value/Cliff's delta above
that cytokine's box pair.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

# ---- data sources ----
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
meta_cluster_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/NTCdegs_STAT3Y705H_heatmap/common_degs_14clusters_8metaClusters_ordered.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/STAT3Y705H_be_effect_boxplot"
os.makedirs(out_dir, exist_ok=True)

TF = "Stat3Y705H"
NTC_LABEL = "NTC"
STIMS = ["IL6", "IL10"]

CLUSTER_A, CLUSTER_B = 5, 7

# ---- raw counts, size-factor normalized (NO ComBat) ----
def deseq2_size_factors(counts):
    log_counts = np.log(counts.replace(0, np.nan))
    log_geo_means = log_counts.mean(axis=1)
    valid = np.isfinite(log_geo_means)
    ratios = log_counts.loc[valid].sub(log_geo_means.loc[valid], axis=0)
    return np.exp(ratios.median(axis=0))


raw_data = pd.read_csv(raw_data_f, index_col=0)
sf = deseq2_size_factors(raw_data)
log2_norm = np.log2(raw_data.div(sf, axis=1) + 1)

col_tokens = [c.split("_") for c in log2_norm.columns]
col_meta = pd.DataFrame({
    "sample": log2_norm.columns.tolist(),
    "batch": [t[0] for t in col_tokens],
    "stim": [t[1] for t in col_tokens],
    "pert": [t[2] for t in col_tokens],
})


def get_cols(stim_g, pert_g, batch):
    mask = (col_meta["stim"] == stim_g) & (col_meta["pert"] == pert_g) & (col_meta["batch"] == batch)
    return col_meta.loc[mask, "sample"].tolist()


batches_tf = sorted(col_meta.loc[col_meta["pert"] == TF, "batch"].unique())
print(f"{TF} batches: {batches_tf}")

# ---- meta-cluster gene assignments ----
meta_clus_df = pd.read_csv(meta_cluster_f)
cluster_a_genes = set(meta_clus_df.loc[meta_clus_df["meta_cluster"] == CLUSTER_A, "gene"])
cluster_b_genes = set(meta_clus_df.loc[meta_clus_df["meta_cluster"] == CLUSTER_B, "gene"])
print(f"Meta-cluster {CLUSTER_A}: {len(cluster_a_genes)} genes")
print(f"Meta-cluster {CLUSTER_B}: {len(cluster_b_genes)} genes")


# ---- Stat3's own edit-vs-NTC fold change, computed fresh from raw
# counts -- generalized for Stat3 spanning multiple batches ----
def stim_delta_for_stim(stim):
    per_replicate_fc = []
    for batch in batches_tf:
        ntc_cols = get_cols(stim, NTC_LABEL, batch)
        edit_cols = get_cols(stim, TF, batch)
        if len(ntc_cols) == 0 or len(edit_cols) == 0:
            continue
        ntc_mean = log2_norm[ntc_cols].mean(axis=1)
        for c in edit_cols:
            per_replicate_fc.append(log2_norm[c] - ntc_mean)
    return pd.concat(per_replicate_fc, axis=1).mean(axis=1)


def cliffs_delta(x, y):
    x = np.array(x)
    y = np.array(y)
    return (np.sum(x[:, None] > y) - np.sum(x[:, None] < y)) / (len(x) * len(y))


# ---- build the combined gene-level table, BOTH cytokines ----
genes_a = [g for g in cluster_a_genes if g in log2_norm.index]
genes_b = [g for g in cluster_b_genes if g in log2_norm.index]
print(f"Meta-cluster {CLUSTER_A} genes in expression data: {len(genes_a)}")
print(f"Meta-cluster {CLUSTER_B} genes in expression data: {len(genes_b)}")

rows = []
for stim in STIMS:
    sd_series = stim_delta_for_stim(stim)
    for g in genes_a:
        sd = sd_series.loc[g]
        rows.append({"gene": g, "meta_cluster": CLUSTER_A, "stim": stim, "stim_delta": sd, "abs_stim_delta": abs(sd)})
    for g in genes_b:
        sd = sd_series.loc[g]
        rows.append({"gene": g, "meta_cluster": CLUSTER_B, "stim": stim, "stim_delta": sd, "abs_stim_delta": abs(sd)})

gene_df = pd.DataFrame(rows)
gene_csv = f"{out_dir}/{TF}_IL6_IL10_metaCluster{CLUSTER_A}v{CLUSTER_B}_perGene.csv"
gene_df.to_csv(gene_csv, index=False)
print(f"Saved: {gene_csv}")

# ---- two-sided Mann-Whitney U (cluster A vs cluster B), computed
# SEPARATELY per cytokine ----
stats_by_stim = {}
for stim in STIMS:
    vals_a = gene_df.loc[(gene_df["meta_cluster"] == CLUSTER_A) & (gene_df["stim"] == stim), "abs_stim_delta"]
    vals_b = gene_df.loc[(gene_df["meta_cluster"] == CLUSTER_B) & (gene_df["stim"] == stim), "abs_stim_delta"]
    stat, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    delta = cliffs_delta(vals_a, vals_b)
    stats_by_stim[stim] = {"vals_a": vals_a, "vals_b": vals_b, "stat": stat, "pval": pval, "delta": delta}

    print(f"\n{stim}: Meta-cluster {CLUSTER_A} n={len(vals_a)}, Meta-cluster {CLUSTER_B} n={len(vals_b)}")
    print(f"  Median |stim_delta| ({CLUSTER_A}): {vals_a.median():.3f}  |  ({CLUSTER_B}): {vals_b.median():.3f}")
    print(f"  Mann-Whitney U = {stat:.3f}, two-sided p = {pval:.3e}, Cliff's delta = {delta:.3f}")

# =============================================================================
# COMBINED plot -- CYTOKINE on the x-axis (IL6, IL10), meta-cluster 5
# and meta-cluster 7 boxes side by side WITHIN each cytokine group
# =============================================================================
BOX_WIDTH = 0.32
BOX_GAP = 0.06       # gap between cluster A and cluster B boxes within one cytokine
STIM_GAP = 1.0       # gap between cytokine groups

CLUSTER_COLOR = {CLUSTER_A: "#6A3D9A", CLUSTER_B: "#FF7F00"}

cluster_a_pos, cluster_b_pos, stim_center = {}, {}, {}
x_cursor = 0.0
for stim in STIMS:
    stim_start = x_cursor
    p_a = x_cursor + BOX_WIDTH / 2
    p_b = p_a + BOX_WIDTH + BOX_GAP
    cluster_a_pos[stim] = p_a
    cluster_b_pos[stim] = p_b
    x_cursor = p_b + BOX_WIDTH / 2
    stim_center[stim] = (stim_start + x_cursor) / 2
    x_cursor += STIM_GAP

fig, ax = plt.subplots(figsize=(6, 6))

for stim in STIMS:
    for cl, pos_map in [(CLUSTER_A, cluster_a_pos), (CLUSTER_B, cluster_b_pos)]:
        vals = gene_df.loc[(gene_df["meta_cluster"] == cl) & (gene_df["stim"] == stim), "abs_stim_delta"]
        bp = ax.boxplot([vals], positions=[pos_map[stim]], widths=BOX_WIDTH,
                         patch_artist=True, showfliers=False, zorder=2)
        for patch in bp["boxes"]:
            patch.set_facecolor(CLUSTER_COLOR[cl])
            patch.set_alpha(0.5)
            patch.set_edgecolor(CLUSTER_COLOR[cl])
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(CLUSTER_COLOR[cl])
                line.set_linewidth(1.3)

# p-value/Cliff's delta annotation above each cytokine's own box pair --
# ONE annotation per cytokine, placed above the midpoint of that
# cytokine's two boxes (cluster A's box and cluster B's box)
y_top = ax.get_ylim()[1]
for stim in STIMS:
    s = stats_by_stim[stim]
    p_text = f"p={s['pval']:.2g}" if s["pval"] >= 1e-4 else f"p={s['pval']:.1e}"
    x_mid = (cluster_a_pos[stim] + cluster_b_pos[stim]) / 2
    ax.annotate(f"{p_text}\nCliff's d={s['delta']:.2f}",
                (x_mid, y_top), ha="center", va="bottom", fontsize=8, color="black")

ax.set_xticks([stim_center[stim] for stim in STIMS])
ax.set_xticklabels(STIMS)
ax.set_ylabel(f"Absolute {TF} edit log2FC (vs cytokine-matched NTC)")
ax.set_title(f"{TF}: Meta-cluster {CLUSTER_A} vs {CLUSTER_B}, IL6 and IL10")

legend_handles = [mpatches.Patch(facecolor=CLUSTER_COLOR[cl], alpha=0.5, edgecolor=CLUSTER_COLOR[cl],
                                  label=f"Meta-cluster {cl}") for cl in [CLUSTER_A, CLUSTER_B]]
ax.legend(handles=legend_handles, title="Meta-cluster", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

plt.tight_layout()

png_f = f"{out_dir}/{TF}_IL6_IL10_metaCluster{CLUSTER_A}v{CLUSTER_B}_boxplot.png"
svg_f = f"{out_dir}/{TF}_IL6_IL10_metaCluster{CLUSTER_A}v{CLUSTER_B}_boxplot.svg"
plt.savefig(png_f, dpi=300, bbox_inches="tight")
plt.savefig(svg_f, bbox_inches="tight")
plt.show()

print(f"\nSaved: {png_f}")
print("\nDone.")