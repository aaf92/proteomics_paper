"""
Fig6c: ErfT529A vs Arid3aT132A pairwise-different genes, fold change vs
cytokine-matched NTC (same batch, same cytokine -- edit_replicate -
mean(NTC replicates, same batch, same cytokine)), individual replicates,
no averaging.

Three heatmaps, all row-clustered with euclidean distance / ward linkage
only (no fallback, no other metric option):
  1. IL6 only  -- IL6-different genes, IL6 columns only
  2. IL10 only -- IL10-different genes, IL10 columns only
  3. Both      -- union of IL6/IL10-different genes, both cytokines' columns
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# ---- data sources -- pairwise gene lists read from the existing pairwise
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
pairwise_heatmap_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/pairwise_union_diff_heatmaps_TestCategories_v3"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6d"
os.makedirs(out_dir, exist_ok=True)

TF_A, TF_B = "ErfT529A", "Arid3aT132A"
PAIR_NAME = f"{TF_A}_vs_{TF_B}"
NTC_LABEL = "NTC"

METRIC = "euclidean"
METHOD = "ward"

EDIT_COLOR = {TF_A: "#1F78B4", TF_B: "#E31A1C"}
CYTOKINE_COLOR = {"IL6": "#F4C430", "IL10": "#88CCEE"}
NTC_DIRECTION_COLOR = {"up": "#B2182B", "down": "#2166AC"}

# ---- raw counts, size-factor normalized (NO ComBat) -- same as single_pair_term_plots.py ----
def deseq2_size_factors(counts):
    """Median-of-ratios size factors, DESeq2's default method."""
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


def get_raw_group_cols(stim_g, pert_g, batch):
    mask = (col_meta["stim"] == stim_g) & (col_meta["pert"] == pert_g) & (col_meta["batch"] == batch)
    return col_meta.loc[mask, "sample"].tolist()


batch_a = col_meta.loc[col_meta["pert"] == TF_A, "batch"].unique()
batch_b = col_meta.loc[col_meta["pert"] == TF_B, "batch"].unique()
assert len(batch_a) == 1 and len(batch_b) == 1, "expected exactly one batch per edit"
batch_a, batch_b = batch_a[0], batch_b[0]
edit_batch = {TF_A: batch_a, TF_B: batch_b}
print(f"{TF_A} batch: {batch_a}  |  {TF_B} batch: {batch_b}")

# ---- NTC's own up/down direction at each cytokine, vs its own PBS
# baseline -- pooling NTC samples from BOTH batches (a biological property
# of the cytokine response itself, not tied to either edit's batch) ----
def ntc_delta_all_batches(stim):
    ntc_stim_cols = get_raw_group_cols(stim, NTC_LABEL, batch_a) + get_raw_group_cols(stim, NTC_LABEL, batch_b)
    ntc_pbs_cols = get_raw_group_cols("PBS", NTC_LABEL, batch_a) + get_raw_group_cols("PBS", NTC_LABEL, batch_b)
    return log2_norm[ntc_stim_cols].mean(axis=1) - log2_norm[ntc_pbs_cols].mean(axis=1)


ntc_delta_il6 = ntc_delta_all_batches("IL6")
ntc_delta_il10 = ntc_delta_all_batches("IL10")


def ntc_direction_row_colors(genes):
    il6_dir = ["up" if ntc_delta_il6.loc[g] > 0 else "down" for g in genes]
    il10_dir = ["up" if ntc_delta_il10.loc[g] > 0 else "down" for g in genes]
    return pd.DataFrame({
        "IL6 NTC direction": [NTC_DIRECTION_COLOR[d] for d in il6_dir],
        "IL10 NTC direction": [NTC_DIRECTION_COLOR[d] for d in il10_dir],
    }, index=genes)


# ---- load this pair's genes, per cytokine ----
def load_pair_genes(cytokine):
    f = f"{pairwise_heatmap_dir}/{PAIR_NAME}/{PAIR_NAME}_{cytokine}_pairwise_different_heatmap_data.csv"
    if not os.path.exists(f):
        print(f"  WARNING: missing {f}")
        return set()
    df = pd.read_csv(f)
    return set(df["gene"].tolist())


il6_genes = load_pair_genes("IL6")
il10_genes = load_pair_genes("IL10")
print(f"IL6-different genes: {len(il6_genes)}  |  IL10-different genes: {len(il10_genes)}")


def build_foldchange_matrix(genes, cytokines):
    """For each gene, each edit's replicate values minus that cytokine's
    own NTC mean (same batch) -- fold change vs cytokine-matched NTC,
    individual replicates, no averaging."""
    genes = sorted(g for g in genes if g in log2_norm.index)

    col_order, col_edit, col_stim = [], [], []
    for stim in cytokines:
        for pert in [TF_A, TF_B]:
            cols = get_raw_group_cols(stim, pert, edit_batch[pert])
            col_order.extend(cols)
            col_edit.extend([pert] * len(cols))
            col_stim.extend([stim] * len(cols))

    fc_df = pd.DataFrame(index=genes, columns=col_order, dtype=float)
    for g in genes:
        for pert in [TF_A, TF_B]:
            for stim in cytokines:
                ntc_cols = get_raw_group_cols(stim, NTC_LABEL, edit_batch[pert])
                ntc_mean = log2_norm.loc[g, ntc_cols].mean()
                edit_cols = get_raw_group_cols(stim, pert, edit_batch[pert])
                fc_df.loc[g, edit_cols] = log2_norm.loc[g, edit_cols].values - ntc_mean

    return fc_df, col_edit, col_stim, col_order


def plot_heatmap(fc_df, col_edit, col_stim, col_order, title, out_name):
    row_link = linkage(pdist(fc_df.values, metric=METRIC), method=METHOD)
    row_order = leaves_list(row_link)
    ordered_genes = [fc_df.index[i] for i in row_order]
    fc_df = fc_df.loc[ordered_genes]

    row_colors_df = ntc_direction_row_colors(ordered_genes)

    edit_colors = [EDIT_COLOR[e] for e in col_edit]
    stim_colors = [CYTOKINE_COLOR[s] for s in col_stim]
    col_colors_df = pd.DataFrame({"Edit": edit_colors, "Cytokine": stim_colors}, index=col_order)

    vabs = max(fc_df.abs().values.max(), 0.1)
    c = sns.color_palette("RdBu_r", as_cmap=True)

    hm = sns.clustermap(
        fc_df,
        row_cluster=False,  # already clustered above
        col_cluster=False,  # fixed edit/cytokine block order
        row_colors=row_colors_df,
        col_colors=col_colors_df,
        cmap=c,
        z_score=None,
        vmin=-vabs, vmax=vabs,
        yticklabels=(len(ordered_genes) <= 150),
        xticklabels=False,  # individual replicate columns -- see col_colors legend instead
        figsize=(9, max(6, 0.15 * len(ordered_genes))),
    )
    if len(ordered_genes) <= 150:
        hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_yticklabels(), fontsize=5)

    edit_legend = [mpatches.Patch(color=EDIT_COLOR[e], label=e) for e in [TF_A, TF_B]]
    l1 = hm.ax_heatmap.legend(handles=edit_legend, title="Edit", bbox_to_anchor=(1.25, 1.0),
                                loc="upper left", fontsize=8, title_fontsize=9)
    hm.ax_heatmap.add_artist(l1)

    cytokines_present = sorted(set(col_stim))
    stim_legend = [mpatches.Patch(color=CYTOKINE_COLOR[s], label=s) for s in cytokines_present]
    l2 = hm.ax_heatmap.legend(handles=stim_legend, title="Cytokine", bbox_to_anchor=(1.25, 0.7),
                          loc="upper left", fontsize=8, title_fontsize=9)
    hm.ax_heatmap.add_artist(l2)

    ntc_dir_legend = [mpatches.Patch(color=NTC_DIRECTION_COLOR[d], label=d) for d in ["up", "down"]]
    hm.ax_heatmap.legend(handles=ntc_dir_legend, title="NTC direction\n(vs own PBS)", bbox_to_anchor=(1.25, 0.4),
                          loc="upper left", fontsize=8, title_fontsize=9)

    hm.fig.suptitle(f"{title}\nn={len(ordered_genes)} genes, fold change vs cytokine-matched NTC "
                     f"(same batch, individual replicates)\nclustering: {METRIC} distance, {METHOD} linkage", y=1.02)

    png_f = f"{out_dir}/{out_name}.png"
    svg_f = f"{out_dir}/{out_name}.svg"
    plt.savefig(png_f, dpi=300, bbox_inches="tight")
    plt.savefig(svg_f, bbox_inches="tight")
    plt.show()
    plt.close(hm.fig)

    save_df = fc_df.copy()
    save_df.insert(0, "gene", ordered_genes)
    save_df.to_csv(f"{out_dir}/{out_name}_data.csv", index=False)

    print(f"Saved: {png_f}")


# 1. IL6 only
fc_il6, edit_il6, stim_il6, cols_il6 = build_foldchange_matrix(il6_genes, ["IL6"])
plot_heatmap(fc_il6, edit_il6, stim_il6, cols_il6,
             f"{PAIR_NAME} -- IL6-different genes, IL6 only",
             f"{PAIR_NAME}_IL6_foldchange_heatmap")

# 2. IL10 only
fc_il10, edit_il10, stim_il10, cols_il10 = build_foldchange_matrix(il10_genes, ["IL10"])
plot_heatmap(fc_il10, edit_il10, stim_il10, cols_il10,
             f"{PAIR_NAME} -- IL10-different genes, IL10 only",
             f"{PAIR_NAME}_IL10_foldchange_heatmap")

# 3. Both -- union of genes, both cytokines' columns
both_genes = il6_genes | il10_genes
fc_both, edit_both, stim_both, cols_both = build_foldchange_matrix(both_genes, ["IL6", "IL10"])
plot_heatmap(fc_both, edit_both, stim_both, cols_both,
             f"{PAIR_NAME} -- IL6 U IL10-different genes, both cytokines",
             f"{PAIR_NAME}_bothCytokines_foldchange_heatmap")

print("\nDone.")
