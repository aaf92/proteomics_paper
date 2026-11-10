"""
Extension of erf_arid3a_foldchange_heatmaps.py: computes pairwise-
different gene lists for ALL THREE of Arid3aT132A's comparisons (vs
ErfT529A, vs TfebS392P, vs Etv3S29P), not just Erf. Gene selection is
computed FRESH from the DESeq2 pairwise results, using the exact method
from 05_pairwise_arid3a_vs_etv3.py (padj<0.05, FC>1.5, restricted to
each pair's own orange/yellow union, independently per cytokine, then
unioned across IL6/IL10) -- NOT read from any pre-computed heatmap CSV,
since those may use different thresholds/logic. Tries both naming
orders (A_vs_B / B_vs_A) per pair for the DESeq2 result files.

Fold change vs cytokine-matched NTC (same batch, same cytokine --
edit_replicate - mean(NTC replicates, same batch, same cytokine)),
individual replicates, no averaging. Row-clustered with euclidean
distance / ward linkage only (no fallback, no other metric option) --
same as the original script.

Four heatmaps:
  1. IL6 only, ALL EDITS  -- union of IL6-different genes across the 3
     Arid3a pairwise comparisons; columns = all 4 edits, IL6 only.
  2. IL10 only, ALL EDITS -- same, for IL10.
  3. Arid3a ONLY -- union of (IL6-different U IL10-different) genes
     across all 3 pairs and both cytokines; columns = Arid3a alone,
     both cytokines.
  4. Erf + Arid3a ONLY -- same gene set as #3, columns = Erf & Arid3a.
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

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6d_allEdits"
os.makedirs(out_dir, exist_ok=True)

ARID3A = "Arid3aT132A"
OTHER_EDITS = ["ErfT529A", "TfebS392P", "Etv3S29P"]
ALL_EDITS = [ARID3A] + OTHER_EDITS
NTC_LABEL = "NTC"

METRIC = "euclidean"
METHOD = "ward"

EDIT_COLOR = {
    "ErfT529A": "#1F78B4", "TfebS392P": "#E31A1C",
    "Arid3aT132A": "#6A3D9A", "Etv3S29P": "#FF7F00",
}
CYTOKINE_COLOR = {"IL6": "#F4C430", "IL10": "#88CCEE"}
NTC_DIRECTION_COLOR = {"up": "#B2182B", "down": "#2166AC"}

# ---- raw counts, size-factor normalized (NO ComBat) -- same as original script ----
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


# ---- one batch per edit, generalized to all 4 edits ----
edit_batch = {}
for tf in ALL_EDITS:
    b = col_meta.loc[col_meta["pert"] == tf, "batch"].unique()
    assert len(b) == 1, f"expected exactly one batch for {tf}"
    edit_batch[tf] = b[0]
    print(f"{tf} batch: {edit_batch[tf]}")

batches_used = sorted(set(edit_batch.values()))

# ---- NTC's own up/down direction at each cytokine, vs its own PBS
# baseline -- pooling NTC samples from ALL batches used across the 4 edits
# (a biological property of the cytokine response itself, not tied to
# any one edit's batch) ----
def ntc_delta_all_batches(stim):
    ntc_stim_cols = [c for b in batches_used for c in get_raw_group_cols(stim, NTC_LABEL, b)]
    ntc_pbs_cols = [c for b in batches_used for c in get_raw_group_cols("PBS", NTC_LABEL, b)]
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


# =============================================================================
# GENE SELECTION -- fresh computation, exact method from
# 05_pairwise_arid3a_vs_etv3.py, generalized across all 3 Arid3a pairs:
#   - pair_oy (per cytokine) = orange/yellow union of JUST the two edits
#     in that pair, independently for IL6 and IL10.
#   - pairwise DESeq2 result for that pair, at each cytokine: padj<0.05
#     AND FC>1.2, intersected with that cytokine's pair_oy.
#   - pair_different = union of the IL6-filtered and IL10-filtered sets.
# =============================================================================
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"
pairwise_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_pairwise_DEG_res/pairwise"

padj_thresh = 0.05
fc_thresh = 1.2  # applied as 2**abs(log2FoldChange) > fc_thresh


def load_orange_yellow(tf, stim):
    f = f"{interaction_dir_base}/{tf}/{tf}_{stim}_fourCategory_classification_data.csv"
    df = pd.read_csv(f, index_col=0)
    return set(df.index[df["category"].isin(["orange", "yellow"])])


oy_by_edit_stim = {}
for tf in ALL_EDITS:
    for stim in ["IL6", "IL10"]:
        oy_by_edit_stim[(tf, stim)] = load_orange_yellow(tf, stim)
        print(f"{tf} orange/yellow ({stim}): n={len(oy_by_edit_stim[(tf, stim)])}")


def load_pairwise_result(tf_a, tf_b, stim):
    """Full result table (not pre-filtered) -- tries both naming orders."""
    candidates = [f"{tf_a}_vs_{tf_b}", f"{tf_b}_vs_{tf_a}"]
    for pair_name in candidates:
        f = f"{pairwise_dir_base}/{pair_name}/resultsOrdered_{pair_name}_at_{stim}.txt"
        if os.path.exists(f):
            print(f"  Loaded pairwise result: {pair_name} at {stim}")
            return pd.read_csv(f, index_col=0)
    print(f"  WARNING: no pairwise result found for {tf_a} vs {tf_b} at {stim} (tried: {candidates})")
    return None


def sig_genes(res, oy_universe):
    if res is None:
        return set()
    filt = res.dropna(subset=["padj", "log2FoldChange"])
    sig = filt[(filt["padj"] < padj_thresh) & (2 ** filt["log2FoldChange"].abs() > fc_thresh)]
    return set(sig.index) & oy_universe


# ---- pull genes for all 3 Arid3a pairwise comparisons ----
PAIRS = [(other, ARID3A) for other in OTHER_EDITS]

il6_sets, il10_sets = {}, {}
for tf_a, tf_b in PAIRS:
    pair_name = f"{tf_a}_vs_{tf_b}"
    print(f"\n=== {pair_name} ===")

    pair_oy_il6 = oy_by_edit_stim[(tf_a, "IL6")] | oy_by_edit_stim[(tf_b, "IL6")]
    pair_oy_il10 = oy_by_edit_stim[(tf_a, "IL10")] | oy_by_edit_stim[(tf_b, "IL10")]

    res_il6 = load_pairwise_result(tf_a, tf_b, "IL6")
    res_il10 = load_pairwise_result(tf_a, tf_b, "IL10")

    il6_sets[pair_name] = sig_genes(res_il6, pair_oy_il6)
    il10_sets[pair_name] = sig_genes(res_il10, pair_oy_il10)
    print(f"  IL6-different: n={len(il6_sets[pair_name])}")
    print(f"  IL10-different: n={len(il10_sets[pair_name])}")

il6_union_all = set().union(*il6_sets.values()) & set(log2_norm.index)
il10_union_all = set().union(*il10_sets.values()) & set(log2_norm.index)
both_union_all = il6_union_all | il10_union_all
print(f"\nUnion across all 3 Arid3a pairs -- IL6-different: n={len(il6_union_all)}")
print(f"Union across all 3 Arid3a pairs -- IL10-different: n={len(il10_union_all)}")
print(f"Union across all 3 Arid3a pairs -- IL6 U IL10-different: n={len(both_union_all)}")


def build_foldchange_matrix(genes, cytokines, perts):
    """For each gene, each edit's replicate values minus that cytokine's
    own NTC mean (same batch) -- fold change vs cytokine-matched NTC,
    individual replicates, no averaging. Generalized to an arbitrary
    list of edits (perts), not just two."""
    genes = sorted(g for g in genes if g in log2_norm.index)

    col_order, col_edit, col_stim = [], [], []
    for stim in cytokines:
        for pert in perts:
            cols = get_raw_group_cols(stim, pert, edit_batch[pert])
            col_order.extend(cols)
            col_edit.extend([pert] * len(cols))
            col_stim.extend([stim] * len(cols))

    fc_df = pd.DataFrame(index=genes, columns=col_order, dtype=float)
    for g in genes:
        for pert in perts:
            for stim in cytokines:
                ntc_cols = get_raw_group_cols(stim, NTC_LABEL, edit_batch[pert])
                ntc_mean = log2_norm.loc[g, ntc_cols].mean()
                edit_cols = get_raw_group_cols(stim, pert, edit_batch[pert])
                fc_df.loc[g, edit_cols] = log2_norm.loc[g, edit_cols].values - ntc_mean

    return fc_df, col_edit, col_stim, col_order


def plot_heatmap(fc_df, col_edit, col_stim, col_order, perts, title, out_name):
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
        figsize=(9, 10),
    )
    if len(ordered_genes) <= 150:
        hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_yticklabels(), fontsize=5)

    plt.tight_layout()

    edit_legend = [mpatches.Patch(color=EDIT_COLOR[e], label=e) for e in perts]
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


# 1. IL6 only, ALL EDITS -- union of IL6-different genes across the 3
# Arid3a pairs; columns = all 4 edits, IL6 only
fc_il6, edit_il6, stim_il6, cols_il6 = build_foldchange_matrix(il6_union_all, ["IL6"], ALL_EDITS)
plot_heatmap(fc_il6, edit_il6, stim_il6, cols_il6, ALL_EDITS,
             f"{ARID3A} pairwise-different genes (vs Erf, Tfeb, Etv3) -- IL6 only, all edits",
             "allEdits_IL6_foldchange_heatmap")

# 2. IL10 only, ALL EDITS
fc_il10, edit_il10, stim_il10, cols_il10 = build_foldchange_matrix(il10_union_all, ["IL10"], ALL_EDITS)
plot_heatmap(fc_il10, edit_il10, stim_il10, cols_il10, ALL_EDITS,
             f"{ARID3A} pairwise-different genes (vs Erf, Tfeb, Etv3) -- IL10 only, all edits",
             "allEdits_IL10_foldchange_heatmap")

# 3. Arid3a ONLY -- union of IL6/IL10-different genes across all 3
# pairs and both cytokines; columns = Arid3a alone, both cytokines
fc_arid3a, edit_arid3a, stim_arid3a, cols_arid3a = build_foldchange_matrix(both_union_all, ["IL6", "IL10"], [ARID3A])
plot_heatmap(fc_arid3a, edit_arid3a, stim_arid3a, cols_arid3a, [ARID3A],
             f"{ARID3A} only -- IL6 U IL10-different genes (union across vs Erf, Tfeb, Etv3), both cytokines",
             "arid3aOnly_bothCytokines_foldchange_heatmap")

# 4. Erf + Arid3a ONLY -- same gene set as heatmap 3 (union of IL6/IL10-
# different genes across all 3 Arid3a pairs, both cytokines), but with
# Erf's columns added back in alongside Arid3a -- same two-edit layout
# as the original erf_arid3a_foldchange_heatmaps.py script
ERF = "ErfT529A"
fc_erf_arid3a, edit_erf_arid3a, stim_erf_arid3a, cols_erf_arid3a = build_foldchange_matrix(
    both_union_all, ["IL6", "IL10"], [ERF, ARID3A]
)
plot_heatmap(fc_erf_arid3a, edit_erf_arid3a, stim_erf_arid3a, cols_erf_arid3a, [ERF, ARID3A],
             f"{ERF} & {ARID3A} -- IL6 U IL10-different genes (union across vs Erf, Tfeb, Etv3), both cytokines",
             "arid3aErf_bothCytokines_foldchange_heatmap")

print("\nDone.")