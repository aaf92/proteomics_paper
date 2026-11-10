"""
Across all C(4,2)=6 pairwise combinations of four edits (Erf, Tfeb, Arid3a,
Etv3 -- no Stat3), finds genes that distinguish each edit pair from the
other -- using a SHARED candidate gene universe (union of orange/yellow
genes across all four edits), tested for pairwise significance
independently in the IL6 and IL10 contexts (edit A vs edit B, within the
same cytokine -- from the batch-corrected joint pairwise DESeq2 model,
NOT vs PBS). GENE SELECTION IS UNCHANGED from before.

"Different" for a given pair = union of (IL6: edit A vs edit B, padj<0.05
and |log2FC|>log2(1.5)) and (IL10: same), restricted to the shared
orange/yellow gene universe.

Two visualizations:
  1. INDIVIDUAL per-pair heatmaps, SPLIT BY CYTOKINE -- one IL6 heatmap and
     one IL10 heatmap per pair, rather than one combined heatmap showing
     both. The IL6 heatmap shows genes with context "IL6-only" or "both",
     using only IL6 (+ PBS baseline) columns. The IL10 heatmap shows genes
     with context "IL10-only" or "both", using only IL10 (+ PBS baseline)
     columns. "both"-context genes appear in BOTH heatmaps. Raw size-
     factor-normalized counts (NO ComBat), individual samples (no
     averaging), each edit shown against its OWN batch-matched PBS_NTC
     baseline -- one shared NTC block if the pair's two edits share a
     batch, two separate batch-matched blocks if not. The range/baseline
     used for control-centering is computed ONCE per pair from the full
     (both-cytokine) column set, then reused for both cytokine-specific
     heatmaps, so their color scales stay directly comparable.
  2. ONE combined summary heatmap -- UNCHANGED. Union of different genes
     across ALL SIX pairs, ComBat-corrected averaged data, every edit +
     NTC across IL6/IL10, normal clustermap defaults.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# ---- data sources ----
data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq/be_effect_analysis/out_files/il6_il10_NTC_ctrl_centered_combatCorrected.csv"
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"
pairwise_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_pairwise_DEG_res/pairwise"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/pairwise_union_diff_heatmaps_TestCategories_v3"
os.makedirs(out_dir, exist_ok=True)

NTC_LABEL = "NTC"

padj_thresh = 0.05
fc_thresh = 1.5  # applied as 2**abs(log2FoldChange) > fc_thresh, same convention as the rest of the pipeline

ALL_EDITS = ["ErfT529A", "TfebS392P", "Arid3aT132A", "Etv3S29P"]  # no Stat3
pairs = list(itertools.combinations(ALL_EDITS, 2))  # all C(4,2) = 6 pairwise combinations

AVERAGE_REPLICATES = True  # applies to Part 2 (combined heatmap) only -- Part 1 always shows individual samples

CONTEXT_COLORS = {"IL6-only": "#F4C430", "IL10-only": "#88CCEE", "both": "#882255"}
EDIT_COLORS = {"ErfT529A": "#1F78B4", "TfebS392P": "#E31A1C",
               "Arid3aT132A": "#6A3D9A", "Etv3S29P": "#FF7F00"}

# =============================================================================
# ComBat-corrected data -- used ONLY for Part 2 (combined summary heatmap),
# unchanged from before
# =============================================================================
data = pd.read_csv(data_f, index_col=0)
print("ComBat-corrected data shape:", data.shape)

all_cols = data.columns.tolist()
col_tokens = [c.split("_") for c in all_cols]
col_meta = pd.DataFrame({
    "sample": all_cols,
    "batch": [t[0] for t in col_tokens],
    "stim":  [t[1] for t in col_tokens],
    "pert":  [t[2] for t in col_tokens],
})


def get_group_cols(stim_g, pert_g, batch=None):
    mask = (col_meta["stim"] == stim_g) & (col_meta["pert"] == pert_g)
    if batch is not None:
        mask &= (col_meta["batch"] == batch)
    return col_meta.loc[mask, "sample"].tolist()


# =============================================================================
# Raw counts, size-factor normalized (NO ComBat) -- used ONLY for Part 1
# (individual per-pair heatmaps), so batch identity is visible rather than
# corrected away
# =============================================================================
def deseq2_size_factors(counts):
    """Median-of-ratios size factors, DESeq2's default method."""
    log_counts = np.log(counts.replace(0, np.nan))
    log_geo_means = log_counts.mean(axis=1)
    valid = np.isfinite(log_geo_means)
    ratios = log_counts.loc[valid].sub(log_geo_means.loc[valid], axis=0)
    return np.exp(ratios.median(axis=0))


raw_data = pd.read_csv(raw_data_f, index_col=0)
print("Raw data shape:", raw_data.shape)

sf = deseq2_size_factors(raw_data)
norm_counts = raw_data.div(sf, axis=1)
log2_norm = np.log2(norm_counts + 1)
print("Size-factor normalized (no ComBat). Size factor range:",
      sf.min().round(3), "-", sf.max().round(3))

raw_cols = log2_norm.columns.tolist()
raw_col_tokens = [c.split("_") for c in raw_cols]
raw_col_meta = pd.DataFrame({
    "sample": raw_cols,
    "batch": [t[0] for t in raw_col_tokens],
    "stim":  [t[1] for t in raw_col_tokens],
    "pert":  [t[2] for t in raw_col_tokens],
})


def get_raw_group_cols(stim_g, pert_g, batch):
    mask = (raw_col_meta["stim"] == stim_g) & (raw_col_meta["pert"] == pert_g) & (raw_col_meta["batch"] == batch)
    return raw_col_meta.loc[mask, "sample"].tolist()


def load_orange_yellow(tf, stim):
    """orange/yellow classification exists PER STIM (02_fourCategory_scatter.py
    loops over IL6 and IL10 separately) -- must specify which one."""
    f = f"{interaction_dir_base}/{tf}/{tf}_{stim}_fourCategory_classification_data.csv"
    df = pd.read_csv(f, index_col=0)
    return df[df["category"].isin(["orange", "yellow"])]


def load_pairwise_different_genes(tf_a, tf_b, stim):
    """Genes where the BATCH-CORRECTED pairwise edit_group contrast
    (tf_a vs tf_b, AT the given stim) clears padj AND FC. Output structure
    is ONE folder per pair (not per stim): pairwise/{tf_a}_vs_{tf_b}/,
    containing resultsOrdered_{tf_a}_vs_{tf_b}_at_{stim}.txt among other
    files.
    """
    candidates = [f"{tf_a}_vs_{tf_b}", f"{tf_b}_vs_{tf_a}"]
    for pair_name in candidates:
        f = f"{pairwise_dir_base}/{pair_name}/resultsOrdered_{pair_name}_at_{stim}.txt"
        if os.path.exists(f):
            res = pd.read_csv(f, index_col=0)
            filt = res.dropna(subset=["padj", "log2FoldChange"])
            sig = filt[(filt["padj"] < padj_thresh) & (2 ** filt["log2FoldChange"].abs() > fc_thresh)]
            print(f"  Loaded pairwise result from: {pair_name} (n_sig={len(sig)}, stim={stim})")
            return set(sig.index)
    print(f"  WARNING: no pairwise result found for {tf_a} vs {tf_b} at {stim} "
          f"(tried: {candidates})")
    return set()


def maybe_average(cols, label):
    if AVERAGE_REPLICATES:
        return data[cols].mean(axis=1).rename(label)
    return data[cols]


# ---- build the shared candidate gene universe -- PER STIM, since
# orange/yellow classification exists separately for IL6 and IL10.
# oy_genes_by_edit_and_stim keeps each edit's own per-stim set for driver
# classification below; union_oy_genes_by_stim is the per-stim candidate
# universe used to restrict each stim's pairwise-different genes ----
oy_genes_by_edit_and_stim = {}
union_oy_genes_by_stim = {}
for stim in ["IL6", "IL10"]:
    union_this_stim = set()
    for edit in ALL_EDITS:
        oy = load_orange_yellow(edit, stim)
        oy_genes_by_edit_and_stim[(edit, stim)] = set(oy.index)
        union_this_stim |= oy_genes_by_edit_and_stim[(edit, stim)]
        print(f"{edit} orange/yellow ({stim}): n={len(oy)}")
    union_oy_genes_by_stim[stim] = union_this_stim
    print(f"Union orange/yellow gene universe ({stim}): n={len(union_this_stim)}")

union_oy_genes = union_oy_genes_by_stim["IL6"] | union_oy_genes_by_stim["IL10"]  # for Part 2 combined reporting

# =============================================================================
# PART 1: INDIVIDUAL PER-PAIR HEATMAPS, SPLIT BY CYTOKINE
# gene selection UNCHANGED; display batch-separated, raw, individual
# samples, own-baseline control-centering
# =============================================================================
all_different_genes = set()  # accumulated for the combined heatmap in Part 2
genes_by_edit_stim = {}  # {(edit, stim): set of genes} -- union across every pair involving that edit, kept SEPARATE per cytokine
part1_heatmap_pngs = []  # tracked for the summary PDF (main, batch-separated heatmaps only)

for tf_a, tf_b in pairs:
    pair_name = f"{tf_a}_vs_{tf_b}"
    print(f"\n{'='*70}\n{pair_name}\n{'='*70}")

    # restrict to THIS PAIR's own two edits' orange/yellow union (not the
    # full four-edit union) -- ensures every gene here already has a
    # genuine orange/yellow call from tf_a or tf_b specifically, so the
    # "ambiguous" driver category can no longer occur
    pair_oy_il6 = oy_genes_by_edit_and_stim[(tf_a, "IL6")] | oy_genes_by_edit_and_stim[(tf_b, "IL6")]
    pair_oy_il10 = oy_genes_by_edit_and_stim[(tf_a, "IL10")] | oy_genes_by_edit_and_stim[(tf_b, "IL10")]

    il6_diff = load_pairwise_different_genes(tf_a, tf_b, stim="IL6") & pair_oy_il6
    il10_diff = load_pairwise_different_genes(tf_a, tf_b, stim="IL10") & pair_oy_il10
    print(f"  IL6-different (padj<{padj_thresh}, FC>{fc_thresh}, in union universe): n={len(il6_diff)}")
    print(f"  IL10-different (padj<{padj_thresh}, FC>{fc_thresh}, in union universe): n={len(il10_diff)}")

    pair_different = il6_diff | il10_diff
    pair_different = [g for g in pair_different if g in log2_norm.index]
    print(f"  Union different genes for this pair (IL6 or IL10): n={len(pair_different)}")

    all_different_genes |= set(pair_different)

    for edit in (tf_a, tf_b):
        genes_by_edit_stim.setdefault((edit, "IL6"), set()).update(il6_diff)
        genes_by_edit_stim.setdefault((edit, "IL10"), set()).update(il10_diff)

    if len(pair_different) == 0:
        print(f"  SKIP {pair_name}: no pairwise-different genes found")
        continue

    gene_context = {}
    for g in pair_different:
        in_il6, in_il10 = g in il6_diff, g in il10_diff
        if in_il6 and in_il10:
            gene_context[g] = "both"
        elif in_il6:
            gene_context[g] = "IL6-only"
        else:
            gene_context[g] = "IL10-only"

    # ---- which batch is each edit actually in? ----
    batch_a = raw_col_meta.loc[raw_col_meta["pert"] == tf_a, "batch"].unique()
    batch_b = raw_col_meta.loc[raw_col_meta["pert"] == tf_b, "batch"].unique()
    if len(batch_a) != 1 or len(batch_b) != 1:
        print(f"  WARNING: {tf_a} batches={list(batch_a)}, {tf_b} batches={list(batch_b)} -- "
              f"expected exactly one each, skipping {pair_name}")
        continue
    batch_a, batch_b = batch_a[0], batch_b[0]
    same_batch = (batch_a == batch_b)
    print(f"  {tf_a} batch: {batch_a}  |  {tf_b} batch: {batch_b}  |  same batch: {same_batch}")

    # ---- range/baseline computed ONCE per pair, from the FULL (both-
    # cytokine) column set -- reused for both cytokine-specific heatmaps
    # below so their color scales stay directly comparable ----
    def batch_cols_all(batch):
        cols = []
        for stim_g in ["PBS", "IL6", "IL10"]:
            for pert_g in [NTC_LABEL, tf_a, tf_b]:
                cols.extend(get_raw_group_cols(stim_g, pert_g, batch))
        return cols

    range_cols = list(dict.fromkeys(batch_cols_all(batch_a) + (batch_cols_all(batch_b) if not same_batch else [])))
    full_for_range = log2_norm.loc[pair_different, range_cols]
    data_range = full_for_range.max(axis=1) - full_for_range.min(axis=1)
    data_range = data_range.replace(0, np.nan)

    control_mean_a = log2_norm.loc[pair_different, get_raw_group_cols("PBS", NTC_LABEL, batch_a)].mean(axis=1)
    if same_batch:
        control_mean_b = control_mean_a
    else:
        control_mean_b = log2_norm.loc[pair_different, get_raw_group_cols("PBS", NTC_LABEL, batch_b)].mean(axis=1)

    batch_a_col_set_full = set(batch_cols_all(batch_a))
    batch_b_col_set_full = set(batch_cols_all(batch_b)) if not same_batch else set()

    # ---- one heatmap per cytokine ----
    for cytokine in ["IL6", "IL10"]:
        genes_for_plot = [g for g in pair_different if gene_context[g] in (f"{cytokine}-only", "both")]
        print(f"  {cytokine} heatmap genes ({cytokine}-only + both): n={len(genes_for_plot)}")

        if len(genes_for_plot) == 0:
            print(f"  SKIP {pair_name} {cytokine}: no genes for this cytokine")
            continue

        # ---- which edit is actually driving each gene's difference? ----
        # orange/yellow exists PER STIM (confirmed: 02_fourCategory_scatter.py
        # loops over both IL6 and IL10), so use each edit's own per-stim
        # orange/yellow set directly for both cytokines -- no fallback needed.
        a_changed = oy_genes_by_edit_and_stim[(tf_a, cytokine)]
        b_changed = oy_genes_by_edit_and_stim[(tf_b, cytokine)]

        gene_driver = {}
        for g in genes_for_plot:
            in_a, in_b = g in a_changed, g in b_changed
            if in_a and not in_b:
                gene_driver[g] = f"{tf_a}-specific"
            elif in_b and not in_a:
                gene_driver[g] = f"{tf_b}-specific"
            elif in_a and in_b:
                gene_driver[g] = "both changed"
            else:
                gene_driver[g] = "ambiguous"

        # should be empty now that il6_diff/il10_diff are restricted to
        # THIS pair's own orange/yellow union -- dropped explicitly as a
        # safety net in case of any edge-case mismatch
        n_ambiguous = sum(1 for g in genes_for_plot if gene_driver[g] == "ambiguous")
        if n_ambiguous > 0:
            print(f"    WARNING: {n_ambiguous} unexpected ambiguous gene(s) -- dropping")
        genes_for_plot = [g for g in genes_for_plot if gene_driver[g] != "ambiguous"]

        if len(genes_for_plot) == 0:
            print(f"  SKIP {pair_name} {cytokine}: no genes remain after dropping ambiguous")
            continue

        driver_order = [f"{tf_a}-specific", f"{tf_b}-specific", "both changed"]
        driver_counts = {d: sum(1 for g in genes_for_plot if gene_driver[g] == d) for d in driver_order}
        print(f"    Driver breakdown: {driver_counts}")

        # ---- columns: PBS baseline + this cytokine only ----
        col_order, col_group_colors, col_group_names, col_group_legend_colors = [], [], [], []

        def add_group(stim_g, pert_g, batch, color, label):
            cols = get_raw_group_cols(stim_g, pert_g, batch)
            if len(cols) == 0:
                print(f"    WARNING: no samples for {stim_g}_{pert_g} (batch {batch})")
                return
            col_order.extend(cols)
            col_group_colors.extend([color] * len(cols))
            col_group_names.append(label)
            col_group_legend_colors.append(color)

        if same_batch:
            add_group("PBS", NTC_LABEL, batch_a, "#B0B0B0", f"PBS_{NTC_LABEL}")
            add_group(cytokine, NTC_LABEL, batch_a, "#4477AA", f"{cytokine}_NTC")
            add_group(cytokine, tf_a, batch_a, EDIT_COLORS[tf_a], f"{cytokine}_{tf_a}")
            add_group(cytokine, tf_b, batch_a, EDIT_COLORS[tf_b], f"{cytokine}_{tf_b}")
        else:
            add_group("PBS", NTC_LABEL, batch_a, "#B0B0B0", f"PBS_NTC_{batch_a}")
            add_group(cytokine, NTC_LABEL, batch_a, "#4477AA", f"{cytokine}_NTC_{batch_a}")
            add_group(cytokine, tf_a, batch_a, EDIT_COLORS[tf_a], f"{cytokine}_{tf_a}")
            add_group("PBS", NTC_LABEL, batch_b, "#D9D9D9", f"PBS_NTC_{batch_b}")
            add_group(cytokine, NTC_LABEL, batch_b, "#6BAED6", f"{cytokine}_NTC_{batch_b}")
            add_group(cytokine, tf_b, batch_b, EDIT_COLORS[tf_b], f"{cytokine}_{tf_b}")

        display_df = log2_norm.loc[genes_for_plot, col_order]

        # apply the pair-level (both-cytokine) range/baseline, subset to
        # this cytokine's genes and columns
        cm_a = control_mean_a.loc[genes_for_plot]
        cm_b = control_mean_b.loc[genes_for_plot]
        dr = data_range.loc[genes_for_plot]

        centered_df = display_df.copy()
        for col in centered_df.columns:
            if col in batch_a_col_set_full:
                centered_df[col] = (display_df[col] - cm_a) / dr
            elif col in batch_b_col_set_full:
                centered_df[col] = (display_df[col] - cm_b) / dr
            else:
                raise ValueError(f"Column {col} not assigned to either batch block -- check group builders.")
        display_df = centered_df

        # ---- single clustering across ALL genes (no more driver-block
        # stacking in the plot -- driver classification is still computed
        # above and saved in the CSV, just not used to reorder/group rows
        # or shown as a legend here) ----
        try:
            row_link = linkage(pdist(display_df.values, metric="correlation"), method="complete")
        except Exception:
            row_link = linkage(pdist(display_df.values, metric="euclidean"), method="complete")
        row_order = leaves_list(row_link)
        genes_for_plot = [genes_for_plot[i] for i in row_order]
        display_df = display_df.loc[genes_for_plot]

        row_colors_series = pd.Series(
            [CONTEXT_COLORS[gene_context[g]] for g in genes_for_plot], index=genes_for_plot
        )

        c = sns.color_palette("RdBu_r", as_cmap=True)
        vabs = max(display_df.abs().values.max(), 0.1)

        hm = sns.clustermap(
            display_df,
            row_cluster=False,  # already clustered above
            col_cluster=False,
            row_colors=row_colors_series,
            col_colors=pd.Series(col_group_colors, index=display_df.columns),
            cmap=c,
            z_score=None,
            vmin=-vabs, vmax=vabs,
            yticklabels=(len(genes_for_plot) <= 150),
            xticklabels=True,
            figsize=(9, 10),
        )

        ordered_genes = genes_for_plot  # already in final display order
        hm.ax_heatmap.set_xticklabels(hm.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=7)
        if len(genes_for_plot) <= 150:
            hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_yticklabels(), fontsize=5)

        present_contexts = sorted(set(gene_context[g] for g in genes_for_plot))
        context_legend = [mpatches.Patch(color=CONTEXT_COLORS[c_], label=c_) for c_ in present_contexts]
        l1 = hm.ax_heatmap.legend(handles=context_legend, title="Different in",
                                    bbox_to_anchor=(1.25, 1.0), loc="upper left",
                                    fontsize=8, title_fontsize=9)
        hm.ax_heatmap.add_artist(l1)

        group_legend = [mpatches.Patch(color=col, label=name)
                        for name, col in zip(col_group_names, col_group_legend_colors)]
        hm.ax_heatmap.legend(handles=group_legend, title="Sample group",
                              bbox_to_anchor=(1.25, 0.6), loc="upper left",
                              fontsize=8, title_fontsize=9)

        batch_note = "same batch" if same_batch else f"different batches ({batch_a} vs {batch_b})"
        hm.fig.suptitle(
            f"{pair_name} -- {cytokine} ({batch_note}): pairwise-different genes\n"
            f"(union orange/yellow universe, padj<{padj_thresh}, FC>{fc_thresh}), "
            f"raw size-factor normalized, no ComBat, own-batch-baseline centered, "
            f"n={len(ordered_genes)} genes",
            y=1.03
        )

        pair_out_dir = f"{out_dir}/{pair_name}"
        os.makedirs(pair_out_dir, exist_ok=True)
        png_f = f"{pair_out_dir}/{pair_name}_{cytokine}_pairwise_different_heatmap.png"
        svg_f = f"{pair_out_dir}/{pair_name}_{cytokine}_pairwise_different_heatmap.svg"
        plt.savefig(png_f, format="png", bbox_inches="tight", dpi=300)
        plt.savefig(svg_f, format="svg", bbox_inches="tight")
        plt.show()
        plt.close(hm.fig)
        part1_heatmap_pngs.append(png_f)  # tracked for the summary PDF below

        save_df = display_df.loc[ordered_genes].copy()
        save_df.insert(0, "driver", [gene_driver[g] for g in ordered_genes])
        save_df.insert(1, "context", [gene_context[g] for g in ordered_genes])
        save_df.insert(2, "gene", ordered_genes)
        save_df.to_csv(f"{pair_out_dir}/{pair_name}_{cytokine}_pairwise_different_heatmap_data.csv", index=False)

        print(f"    Saved: {png_f}")

        # =====================================================================
        # ADDITIONAL VARIANT: same genes, ComBat-corrected data, no batch
        # split -- 4 pooled groups (PBS_NTC, cytokine_NTC, cytokine_tfA,
        # cytokine_tfB) instead of the batch-separated raw display above
        # =====================================================================
        combat_group_specs = [
            ("PBS", NTC_LABEL, "#B0B0B0"),
            (cytokine, NTC_LABEL, "#4477AA"),
            (cytokine, tf_a, EDIT_COLORS[tf_a]),
            (cytokine, tf_b, EDIT_COLORS[tf_b]),
        ]
        combat_col_order, combat_col_colors, combat_col_names, combat_col_legend_colors = [], [], [], []
        for stim_g, pert_g, color in combat_group_specs:
            cols = get_group_cols(stim_g, pert_g)
            if len(cols) == 0:
                print(f"    WARNING: no ComBat-corrected samples for {stim_g}_{pert_g}")
                continue
            combat_col_order.extend(cols)
            combat_col_colors.extend([color] * len(cols))
            combat_col_names.append(f"{stim_g}_{pert_g}")
            combat_col_legend_colors.append(color)

        combat_display_df = data.loc[genes_for_plot, combat_col_order]

        try:
            combat_row_link = linkage(pdist(combat_display_df.values, metric="correlation"), method="complete")
        except Exception:
            combat_row_link = linkage(pdist(combat_display_df.values, metric="euclidean"), method="complete")
        combat_row_order = leaves_list(combat_row_link)
        combat_genes_ordered = [genes_for_plot[i] for i in combat_row_order]
        combat_display_df = combat_display_df.loc[combat_genes_ordered]

        combat_row_colors_series = pd.Series(
            [CONTEXT_COLORS[gene_context[g]] for g in combat_genes_ordered], index=combat_genes_ordered
        )
        combat_vabs = max(combat_display_df.abs().values.max(), 0.1)

        hm_combat = sns.clustermap(
            combat_display_df,
            row_cluster=False,
            col_cluster=False,
            row_colors=combat_row_colors_series,
            col_colors=pd.Series(combat_col_colors, index=combat_display_df.columns),
            cmap=c,
            z_score=None,
            vmin=-combat_vabs, vmax=combat_vabs,
            yticklabels=(len(combat_genes_ordered) <= 150),
            xticklabels=True,
            figsize=(9, 10),
        )
        hm_combat.ax_heatmap.set_xticklabels(hm_combat.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=7)
        if len(combat_genes_ordered) <= 150:
            hm_combat.ax_heatmap.set_yticklabels(hm_combat.ax_heatmap.get_yticklabels(), fontsize=5)

        combat_context_legend = [mpatches.Patch(color=CONTEXT_COLORS[c_], label=c_) for c_ in present_contexts]
        lc1 = hm_combat.ax_heatmap.legend(handles=combat_context_legend, title="Different in",
                                           bbox_to_anchor=(1.25, 1.0), loc="upper left",
                                           fontsize=8, title_fontsize=9)
        hm_combat.ax_heatmap.add_artist(lc1)

        combat_group_legend = [mpatches.Patch(color=col, label=name)
                                for name, col in zip(combat_col_names, combat_col_legend_colors)]
        hm_combat.ax_heatmap.legend(handles=combat_group_legend, title="Sample group",
                                     bbox_to_anchor=(1.25, 0.6), loc="upper left",
                                     fontsize=8, title_fontsize=9)

        hm_combat.fig.suptitle(
            f"{pair_name} -- {cytokine} (ComBat-corrected, pooled across batches): "
            f"pairwise-different genes\nn={len(combat_genes_ordered)} genes",
            y=1.03
        )

        combat_png_f = f"{pair_out_dir}/{pair_name}_{cytokine}_pairwise_different_heatmap_combatCorrected.png"
        combat_svg_f = f"{pair_out_dir}/{pair_name}_{cytokine}_pairwise_different_heatmap_combatCorrected.svg"
        plt.savefig(combat_png_f, format="png", bbox_inches="tight", dpi=300)
        plt.savefig(combat_svg_f, format="svg", bbox_inches="tight")
        plt.show()
        plt.close(hm_combat.fig)

        combat_save_df = combat_display_df.copy()
        combat_save_df.insert(0, "driver", [gene_driver[g] for g in combat_genes_ordered])
        combat_save_df.insert(1, "context", [gene_context[g] for g in combat_genes_ordered])
        combat_save_df.insert(2, "gene", combat_genes_ordered)
        combat_save_df.to_csv(
            f"{pair_out_dir}/{pair_name}_{cytokine}_pairwise_different_heatmap_combatCorrected_data.csv",
            index=False
        )

        part1_heatmap_pngs.append(combat_png_f)  # tracked for the summary PDF, right after its main-heatmap counterpart
        print(f"    Saved: {combat_png_f}")

# =============================================================================
# SUMMARY PDF: every per-pair-per-cytokine heatmap, both variants (main
# batch-separated + ComBat-corrected pooled), one page per image, in
# generation order -- main and ComBat pages alternate for each pair/cytokine
# =============================================================================
pdf_path = f"{out_dir}/all_pairwise_heatmaps_summary.pdf"
with PdfPages(pdf_path) as pdf:
    for png_path in part1_heatmap_pngs:
        if not os.path.exists(png_path):
            continue
        fig, ax = plt.subplots(figsize=(10, 11))
        ax.imshow(mpimg.imread(png_path))
        ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
print(f"\nSaved summary PDF: {pdf_path}  ({len(part1_heatmap_pngs)} pages)")

# =============================================================================
# UNION SUMMARY TABLE: every gene identified across the per-edit-per-
# cytokine analysis, with which edit and cytokine (condition) it came
# from. A gene appears on MULTIPLE rows if it showed up for more than one
# edit and/or both cytokines -- edit/condition repeat by design, not
# deduplicated.
# =============================================================================
union_summary_rows = []
for (edit, cytokine), genes in genes_by_edit_stim.items():
    for g in genes:
        union_summary_rows.append({"gene": g, "edit": edit, "condition": cytokine})

union_summary_df = pd.DataFrame(union_summary_rows).sort_values(["edit", "condition", "gene"]).reset_index(drop=True)
print(f"\nUnion summary table: {len(union_summary_df)} rows "
      f"({union_summary_df['gene'].nunique()} unique genes, "
      f"{union_summary_df['edit'].nunique()} edits, {union_summary_df['condition'].nunique()} conditions)")

union_summary_out = f"{out_dir}/all_pairwise_genes_by_edit_condition.csv"
union_summary_df.to_csv(union_summary_out, index=False)
print(f"Saved: {union_summary_out}")

# =============================================================================
# PART 2: PER-EDIT, PER-CYTOKINE HEATMAP -- for each edit and each cytokine
# SEPARATELY, the union of that edit's own pairwise-different genes for
# THAT cytokine only (across every pair involving it) vs everything else.
# IL6 and IL10 are never combined here: genes and columns are both
# restricted to one cytokine at a time. ComBat-corrected, ALL individual
# replicates (no averaging). Columns: NTC first, then each edit, all
# within that one cytokine.
# =============================================================================
print(f"\n{'='*70}\nPer-edit, per-cytokine heatmaps\n{'='*70}")

pert_palette = sns.color_palette("tab10", len(ALL_EDITS) + 1)
pert_lut = dict(zip([NTC_LABEL] + ALL_EDITS, pert_palette))
stim_lut = {"IL6": "#d62728", "IL10": "#1f77b4"}

part2_pngs_by_cytokine = {"IL6": [], "IL10": []}  # tracked for the two summary PDFs below

for edit in ALL_EDITS:
    for cytokine in ["IL6", "IL10"]:
        edit_genes = sorted(g for g in genes_by_edit_stim.get((edit, cytokine), set()) if g in data.index)
        print(f"\n{edit} / {cytokine}: n={len(edit_genes)} pairwise-different genes "
              f"(union across all pairs involving {edit}, {cytokine} only)")

        if len(edit_genes) == 0:
            print(f"  SKIP {edit} / {cytokine}: no genes")
            continue

        # individual replicate columns, ComBat-corrected, THIS CYTOKINE ONLY
        group_specs = [(cytokine, NTC_LABEL)] + [(cytokine, e) for e in ALL_EDITS]
        col_order, col_pert_list = [], []
        for stim_g, pert_g in group_specs:
            cols = get_group_cols(stim_g, pert_g)
            if len(cols) == 0:
                print(f"  WARNING: no samples for {stim_g}_{pert_g}")
                continue
            col_order.extend(cols)
            col_pert_list.extend([pert_g] * len(cols))

        plot_df = data.loc[edit_genes, col_order]

        try:
            row_link = linkage(pdist(plot_df.values, metric="correlation"), method="complete")
        except Exception:
            row_link = linkage(pdist(plot_df.values, metric="euclidean"), method="complete")
        row_order = leaves_list(row_link)
        ordered_genes = [edit_genes[i] for i in row_order]
        plot_df = plot_df.loc[ordered_genes]

        pert_colors = [pert_lut[p] for p in col_pert_list]
        col_colors_series = pd.Series(pert_colors, index=plot_df.columns)

        c = sns.color_palette("RdBu_r", as_cmap=True)
        vabs = max(plot_df.abs().values.max(), 0.1)

        hm = sns.clustermap(
            plot_df,
            row_cluster=False,  # already clustered above
            col_cluster=False,  # fixed NTC-first ordering
            col_colors=col_colors_series,
            cmap=c,
            z_score=None,
            vmin=-vabs, vmax=vabs,
            yticklabels=(len(ordered_genes) <= 150),
            xticklabels=False,  # too many individual replicate columns to label meaningfully -- see col_colors legend instead
            figsize=(9, 10),
        )
        if len(ordered_genes) <= 150:
            hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_yticklabels(), fontsize=5)

        pert_legend = [mpatches.Patch(color=pert_lut[p], label=p) for p in [NTC_LABEL] + ALL_EDITS]
        hm.ax_heatmap.legend(handles=pert_legend, title="Edit", bbox_to_anchor=(1.25, 1.0),
                              loc="upper left", fontsize=8, title_fontsize=9)

        hm.fig.suptitle(
            f"{edit} pairwise-different genes vs everything else -- {cytokine} only\n"
            f"(union across all pairs involving {edit}, padj<{padj_thresh}, FC>{fc_thresh})\n"
            f"ComBat-corrected, individual replicates, n={len(ordered_genes)} genes",
            y=1.02
        )

        png_f = f"{out_dir}/{edit}_{cytokine}_vs_all_pairwise_genes_heatmap.png"
        svg_f = f"{out_dir}/{edit}_{cytokine}_vs_all_pairwise_genes_heatmap.svg"
        plt.savefig(png_f, format="png", bbox_inches="tight", dpi=300)
        plt.savefig(svg_f, format="svg", bbox_inches="tight")
        plt.show()
        plt.close(hm.fig)

        save_df = plot_df.copy()
        save_df.insert(0, "gene", ordered_genes)
        save_df.to_csv(f"{out_dir}/{edit}_{cytokine}_vs_all_pairwise_genes_heatmap_data.csv", index=False)

        part2_pngs_by_cytokine[cytokine].append(png_f)  # tracked for the summary PDF below
        print(f"  Saved: {png_f}")
        print(f"  Saved: {svg_f}")

# =============================================================================
# SUMMARY PDFs: one for all IL6 per-edit heatmaps, one for all IL10
# per-edit heatmaps -- each page is one edit's heatmap for that cytokine
# =============================================================================
for cytokine, png_list in part2_pngs_by_cytokine.items():
    pdf_path = f"{out_dir}/{cytokine}_pairwise_edits_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        for png_path in png_list:
            if not os.path.exists(png_path):
                continue
            fig, ax = plt.subplots(figsize=(10, 11))
            ax.imshow(mpimg.imread(png_path))
            ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    print(f"\nSaved summary PDF: {pdf_path}  ({len(png_list)} pages)")

print("\nDone.")