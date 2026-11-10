"""
Fig6c-alt3: two SEPARATE heatmaps, each restricted to genes shared
between Stat3Y705H and ONE other edit, EXCLUDING genes shared by all
THREE edits (mutually exclusive from the triple-overlap region) --
Venn-style raw overlap (union of IL6/IL10 orange/yellow per edit, then
raw set intersection minus the third edit's set, no pairwise
significance filter -- same method as oy_venn3_stat3_erf_arid3a.py,
these are that Venn diagram's "101" and "011" regions specifically, not
"101"+"111" or "011"+"111"). Additionally filtered to genes where THAT
OTHER EDIT specifically (NOT Stat3) shows >=1.5x LINEAR fold change
(2**abs(mean log2FC) > 1.5) in at least one cytokine, checked on the
MEAN fold change per gene.

Both edits restricted to their MATCHED/shared batch only -- for heatmap
1, Stat3's samples from any batch other than Erf's are excluded
entirely; for heatmap 2, restricted to Arid3a's batch instead.

  1. Stat3 + Erf    -- genes = (Stat3 oy INTERSECT Erf oy) MINUS Arid3a
     oy, Erf FC>=1.5x in >=1 cytokine; both edits restricted to Erf's batch.
  2. Stat3 + Arid3a -- same idea, (Stat3 oy INTERSECT Arid3a oy) MINUS
     Erf oy, Arid3a FC>=1.5x, restricted to Arid3a's batch.

Fixed figure size (does NOT scale with number of rows).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# ---- data sources ----
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6/FC_heatmap_stat3_shared_matchedBatch"
os.makedirs(out_dir, exist_ok=True)

STAT3, TF_A, TF_B = "Stat3Y705H", "ErfT529A", "Arid3aT132A"
NTC_LABEL = "NTC"
STIMS = ["IL6", "IL10"]
BASE_CATEGORIES = ["orange", "yellow"]

METRIC = "euclidean"
METHOD = "ward"

FC_THRESH = 1.5  # LINEAR fold change -- applied as 2**abs(mean log2FC) > FC_THRESH

EDIT_COLOR = {STAT3: "#33A02C", TF_A: "#1F78B4", TF_B: "#E31A1C"}
CYTOKINE_COLOR = {"IL6": "#F4C430", "IL10": "#88CCEE"}
NTC_DIRECTION_COLOR = {"up": "#B2182B", "down": "#2166AC"}

# ---- raw counts, size-factor normalized (NO ComBat) ----
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


# ---- orange/yellow per TF, unioned across IL6/IL10 -- SAME logic as
# oy_venn3_stat3_erf_arid3a.py ----
def load_orange_yellow_union(tf):
    tf_union = set()
    for stim in STIMS:
        class_f = f"{interaction_dir_base}/{tf}/{tf}_{stim}_fourCategory_classification_data.csv"
        if not os.path.exists(class_f):
            print(f"WARNING: missing {class_f}, {tf}/{stim} contributes nothing to the union")
            continue
        class_df = pd.read_csv(class_f, index_col=0)
        stim_genes = set(class_df.index[class_df["category"].isin(BASE_CATEGORIES)])
        print(f"  {tf} / {stim}: orange_yellow = {len(stim_genes)}")
        tf_union |= stim_genes
    return tf_union


oy_stat3 = load_orange_yellow_union(STAT3)
oy_erf = load_orange_yellow_union(TF_A)
oy_arid3a = load_orange_yellow_union(TF_B)
print(f"{STAT3}: orange_yellow (IL6 U IL10) = {len(oy_stat3)}")
print(f"{TF_A}: orange_yellow (IL6 U IL10) = {len(oy_erf)}")
print(f"{TF_B}: orange_yellow (IL6 U IL10) = {len(oy_arid3a)}")

stat3_erf_genes = (oy_stat3 & oy_erf) - oy_arid3a
stat3_arid3a_genes = (oy_stat3 & oy_arid3a) - oy_erf
print(f"\nStat3 & Erf shared genes (excluding triple overlap with Arid3a): {len(stat3_erf_genes)}")
print(f"Stat3 & Arid3a shared genes (excluding triple overlap with Erf): {len(stat3_arid3a_genes)}")


def gene_passes_fc_threshold(gene, pert, batch):
    """Check ONLY this specific edit's mean fold change (NOT Stat3's),
    across both cytokines, at the given batch."""
    for stim in STIMS:
        ntc_cols = get_raw_group_cols(stim, NTC_LABEL, batch)
        edit_cols = get_raw_group_cols(stim, pert, batch)
        if len(ntc_cols) == 0 or len(edit_cols) == 0:
            continue
        mean_fc = log2_norm.loc[gene, edit_cols].mean() - log2_norm.loc[gene, ntc_cols].mean()
        if 2 ** abs(mean_fc) > FC_THRESH:
            return True
    return False


# ---- NTC direction, computed FRESH per batch (each heatmap is
# restricted to just one matched batch) ----
def ntc_deltas_for_batch(batch):
    il6_ntc_cols = get_raw_group_cols("IL6", NTC_LABEL, batch)
    il10_ntc_cols = get_raw_group_cols("IL10", NTC_LABEL, batch)
    pbs_ntc_cols = get_raw_group_cols("PBS", NTC_LABEL, batch)
    il6_delta = log2_norm[il6_ntc_cols].mean(axis=1) - log2_norm[pbs_ntc_cols].mean(axis=1)
    il10_delta = log2_norm[il10_ntc_cols].mean(axis=1) - log2_norm[pbs_ntc_cols].mean(axis=1)
    return il6_delta, il10_delta


def ntc_direction_row_colors(genes, il6_delta, il10_delta):
    il6_dir = ["up" if il6_delta.loc[g] > 0 else "down" for g in genes]
    il10_dir = ["up" if il10_delta.loc[g] > 0 else "down" for g in genes]
    return pd.DataFrame({
        "IL6 NTC direction": [NTC_DIRECTION_COLOR[d] for d in il6_dir],
        "IL10 NTC direction": [NTC_DIRECTION_COLOR[d] for d in il10_dir],
    }, index=genes)


def overall_ntc_direction(gene, il6_delta, il10_delta):
    d_il6 = il6_delta.loc[gene]
    d_il10 = il10_delta.loc[gene]
    if (d_il6 > 0) == (d_il10 > 0):
        winner = d_il6
    else:
        winner = d_il6 if abs(d_il6) >= abs(d_il10) else d_il10
    return "up" if winner > 0 else "down"


def cluster_subset(fc_df, genes):
    if len(genes) <= 1:
        return list(genes)
    sub = fc_df.loc[genes]
    link = linkage(pdist(sub.values, metric=METRIC), method=METHOD)
    order = leaves_list(link)
    return [genes[i] for i in order]


def build_foldchange_matrix(genes, other_edit, batch, cytokines):
    """Both Stat3 and other_edit restricted to ONLY this one batch."""
    genes = sorted(g for g in genes if g in log2_norm.index)
    perts = [STAT3, other_edit]

    col_order, col_edit, col_stim = [], [], []
    for stim in cytokines:
        for pert in perts:
            cols = get_raw_group_cols(stim, pert, batch)
            col_order.extend(cols)
            col_edit.extend([pert] * len(cols))
            col_stim.extend([stim] * len(cols))

    fc_df = pd.DataFrame(index=genes, columns=col_order, dtype=float)
    for g in genes:
        for pert in perts:
            for stim in cytokines:
                ntc_cols = get_raw_group_cols(stim, NTC_LABEL, batch)
                ntc_mean = log2_norm.loc[g, ntc_cols].mean()
                edit_cols = get_raw_group_cols(stim, pert, batch)
                fc_df.loc[g, edit_cols] = log2_norm.loc[g, edit_cols].values - ntc_mean

    return fc_df, col_edit, col_stim, col_order


def plot_heatmap(fc_df, col_edit, col_stim, col_order, batch, other_edit, title, out_name):
    il6_delta, il10_delta = ntc_deltas_for_batch(batch)

    up_genes = [g for g in fc_df.index if overall_ntc_direction(g, il6_delta, il10_delta) == "up"]
    down_genes = [g for g in fc_df.index if overall_ntc_direction(g, il6_delta, il10_delta) == "down"]

    up_ordered = cluster_subset(fc_df, up_genes)
    down_ordered = cluster_subset(fc_df, down_genes)
    ordered_genes = up_ordered + down_ordered
    print(f"  {title}: {len(up_ordered)} up-regulated NTC DEGs, {len(down_ordered)} down-regulated NTC DEGs")

    fc_df = fc_df.loc[ordered_genes]
    row_colors_df = ntc_direction_row_colors(ordered_genes, il6_delta, il10_delta)

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
        yticklabels=True,
        xticklabels=False,  # individual replicate columns -- see col_colors legend instead
        figsize=(9, 10),  # fixed -- does NOT scale with number of rows
    )
    hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_yticklabels(), fontsize=6)

    edit_legend = [mpatches.Patch(color=EDIT_COLOR[e], label=e) for e in [STAT3, other_edit]]
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
                     f"(batch {batch} only, individual replicates), {other_edit} >={FC_THRESH}x FC in >=1 cytokine\n"
                     f"clustering: {METRIC} distance, {METHOD} linkage -- separately within up- and "
                     f"down-regulated NTC DEG blocks, then stacked", y=1.02)

    plt.tight_layout()

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


# ---- Erf and Arid3a's own batches (each confirmed single-batch) ----
batch_erf = col_meta.loc[col_meta["pert"] == TF_A, "batch"].unique()
batch_arid3a = col_meta.loc[col_meta["pert"] == TF_B, "batch"].unique()
assert len(batch_erf) == 1, "expected exactly one batch for Erf"
assert len(batch_arid3a) == 1, "expected exactly one batch for Arid3a"
batch_erf, batch_arid3a = batch_erf[0], batch_arid3a[0]
print(f"\nErf batch: {batch_erf}  |  Arid3a batch: {batch_arid3a}")

# sanity check -- does Stat3 actually have samples in each of these
# batches? If not, that heatmap's Stat3 columns will come back empty.
stat3_batches = set(col_meta.loc[col_meta["pert"] == STAT3, "batch"].unique())
print(f"Stat3 batches available: {sorted(stat3_batches)}")
if batch_erf not in stat3_batches:
    print(f"WARNING: Stat3 has no samples in batch {batch_erf} (Erf's batch) -- heatmap 1 will have no Stat3 columns")
if batch_arid3a not in stat3_batches:
    print(f"WARNING: Stat3 has no samples in batch {batch_arid3a} (Arid3a's batch) -- heatmap 2 will have no Stat3 columns")

# ---- Heatmap 1: Stat3 + Erf, matched to Erf's batch, FC filter on Erf ----
genes_erf = [g for g in stat3_erf_genes if g in log2_norm.index]
n_before = len(genes_erf)
genes_erf = [g for g in genes_erf if gene_passes_fc_threshold(g, TF_A, batch_erf)]
print(f"\nStat3 & Erf genes: {n_before} -> {len(genes_erf)} after requiring {TF_A} FC >= {FC_THRESH}x in >=1 cytokine")

fc_erf, edit_erf, stim_erf, cols_erf = build_foldchange_matrix(genes_erf, TF_A, batch_erf, ["IL6", "IL10"])
plot_heatmap(fc_erf, edit_erf, stim_erf, cols_erf, batch_erf, TF_A,
             f"{STAT3} & {TF_A} shared orange/yellow genes, excl. triple overlap "
             f"({TF_A} FC>={FC_THRESH}x, batch {batch_erf} only)",
             f"stat3_erf_sharedGenes_exclTriple_fc{FC_THRESH}_batch{batch_erf}_foldchange_heatmap")

# ---- Heatmap 2: Stat3 + Arid3a, matched to Arid3a's batch, FC filter on Arid3a ----
genes_arid3a = [g for g in stat3_arid3a_genes if g in log2_norm.index]
n_before = len(genes_arid3a)
genes_arid3a = [g for g in genes_arid3a if gene_passes_fc_threshold(g, TF_B, batch_arid3a)]
print(f"Stat3 & Arid3a genes: {n_before} -> {len(genes_arid3a)} after requiring {TF_B} FC >= {FC_THRESH}x in >=1 cytokine")

fc_arid3a, edit_arid3a, stim_arid3a, cols_arid3a = build_foldchange_matrix(
    genes_arid3a, TF_B, batch_arid3a, ["IL6", "IL10"]
)
plot_heatmap(fc_arid3a, edit_arid3a, stim_arid3a, cols_arid3a, batch_arid3a, TF_B,
             f"{STAT3} & {TF_B} shared orange/yellow genes, excl. triple overlap "
             f"({TF_B} FC>={FC_THRESH}x, batch {batch_arid3a} only)",
             f"stat3_arid3a_sharedGenes_exclTriple_fc{FC_THRESH}_batch{batch_arid3a}_foldchange_heatmap")

print("\nDone.")