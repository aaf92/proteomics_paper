"""
Fig6b: subset of enhance_dampen_NTC_genes.py -- just the FINAL combined
(IL6 U IL10) stacked bar PROPORTION plot, dropping the per-cytokine
(IL6-only, IL10-only) plots and the combined raw-count plot.

Classifies what each edit does to NTC's own cytokine response direction,
per gene (enhanced = same direction as NTC's own response, dampened =
pulled back toward baseline), unioned across IL6 and IL10 (a gene
qualifies if orange/yellow in EITHER stim; for a gene present in both,
whichever stim has the larger |stim_delta| is used).

stim_delta is computed fresh from raw counts (size-factor-normalized
log2_norm, batch-matched) as cytokine_edit - cytokine_NTC -- the same
approach used in the fold-change heatmap script -- rather than pulled
from the classification file's own stim_delta column, which comes from
a different upstream normalization pipeline (02_fourCategory_scatter.py's
ranked_f). The classification file is still used for orange/yellow gene
selection, just not for the stim_delta value itself.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- data sources ----
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6b"
os.makedirs(out_dir, exist_ok=True)

ALL_EDITS = ["ErfT529A", "Arid3aT132A"]
STIMS = ["IL6", "IL10"]
NTC_LABEL = "NTC"

# White fill with colored hatch lines for dampened; solid fill for
# enhanced -- lighter red/blue tints as the base color. Matplotlib draws
# hatch lines using edgecolor, not facecolor.
CATEGORY_FACECOLOR = {
    "enhanced_up": "#FB9A99", "dampened_up": "white",
    "enhanced_down": "#A6CEE3", "dampened_down": "white",
}
CATEGORY_EDGECOLOR = {
    "enhanced_up": "#FB9A99", "dampened_up": "#FB9A99",
    "enhanced_down": "#A6CEE3", "dampened_down": "#A6CEE3",
}
CATEGORY_HATCH = {
    "enhanced_up": None, "dampened_up": "////",
    "enhanced_down": None, "dampened_down": "////",
}
CATEGORY_ORDER = ["dampened_down", "dampened_up", "enhanced_down", "enhanced_up"]
CATEGORY_ORDER = list(reversed(CATEGORY_ORDER))


def stacked_bar(ax, x_index, data, categories):
    """One ax.bar call per category, applying face/edge color + hatch consistently."""
    bottom = np.zeros(len(x_index))
    for cat in categories:
        ax.bar(x_index, data[cat], bottom=bottom, color=CATEGORY_FACECOLOR[cat],
               hatch=CATEGORY_HATCH[cat], edgecolor=CATEGORY_EDGECOLOR[cat], linewidth=1.0, label=cat)
        bottom += data[cat].values


# ---- raw counts, size-factor normalized (NO ComBat) -- for NTC's own
# cytokine-vs-PBS direction, batch-matched to whichever edit we're
# comparing against ----
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


def load_orange_yellow(tf, stim):
    f = f"{interaction_dir_base}/{tf}/{tf}_{stim}_fourCategory_classification_data.csv"
    df = pd.read_csv(f, index_col=0)
    return df[df["category"].isin(["orange", "yellow"])]


# ---- per-edit, per-stim gene-level classification (feeds the combined union below) ----
rows = []
for tf in ALL_EDITS:
    batch = col_meta.loc[col_meta["pert"] == tf, "batch"].unique()
    assert len(batch) == 1, f"expected exactly one batch for {tf}"
    batch = batch[0]

    for stim in STIMS:
        oy = load_orange_yellow(tf, stim)
        print(f"{tf} / {stim}: {len(oy)} orange/yellow genes")

        ntc_pbs_cols = get_cols("PBS", NTC_LABEL, batch)
        ntc_stim_cols = get_cols(stim, NTC_LABEL, batch)
        edit_stim_cols = get_cols(stim, tf, batch)

        genes = [g for g in oy.index if g in log2_norm.index]
        n_missing = len(oy) - len(genes)
        if n_missing:
            print(f"  {n_missing} genes not found in expression data, skipped")

        ntc_delta = log2_norm.loc[genes, ntc_stim_cols].mean(axis=1) - \
                    log2_norm.loc[genes, ntc_pbs_cols].mean(axis=1)

        # stim_delta computed fresh from raw counts (same size-factor-
        # normalized log2_norm, same batch-matching) as the fold-change
        # heatmap script -- cytokine_edit - cytokine_NTC, both at this
        # stim -- rather than pulled from the classification file's own
        # stim_delta column, which comes from a different upstream
        # normalization pipeline
        stim_delta = log2_norm.loc[genes, edit_stim_cols].mean(axis=1) - \
                     log2_norm.loc[genes, ntc_stim_cols].mean(axis=1)

        for g in genes:
            nd = ntc_delta.loc[g]
            sd = stim_delta.loc[g]

            ntc_direction = "up" if nd > 0 else "down"
            same_sign = (sd > 0) == (nd > 0)
            effect = "enhanced" if same_sign else "dampened"
            category = f"{effect}_{ntc_direction}"

            rows.append({
                "edit": tf, "stim": stim, "gene": g,
                "ntc_delta": nd, "stim_delta": sd, "category": category,
            })

results_df = pd.DataFrame(rows)

# =============================================================================
# COMBINED: UNION of IL6 and IL10 orange/yellow genes (not intersection --
# a gene qualifies if it's orange/yellow in EITHER stim, no direction-
# agreement requirement). For a gene present in both stims, use whichever
# stim has the LARGER |stim_delta| and take that stim's own paired
# (ntc_delta, stim_delta) together.
# =============================================================================
pivot_ntc = results_df.pivot_table(index=["edit", "gene"], columns="stim", values="ntc_delta")
pivot_stim = results_df.pivot_table(index=["edit", "gene"], columns="stim", values="stim_delta")

union_index = pivot_ntc.dropna(how="all", subset=["IL6", "IL10"]).index
print(f"\nGenes in EITHER IL6 or IL10 orange/yellow set (any edit): {len(union_index)}")

combined_rows = []
for edit, gene in union_index:
    nd_il6, nd_il10 = pivot_ntc.loc[(edit, gene), "IL6"], pivot_ntc.loc[(edit, gene), "IL10"]
    sd_il6, sd_il10 = pivot_stim.loc[(edit, gene), "IL6"], pivot_stim.loc[(edit, gene), "IL10"]

    has_il6, has_il10 = pd.notna(sd_il6), pd.notna(sd_il10)
    if has_il6 and has_il10:
        if abs(sd_il6) >= abs(sd_il10):
            nd, sd = nd_il6, sd_il6
        else:
            nd, sd = nd_il10, sd_il10
    elif has_il6:
        nd, sd = nd_il6, sd_il6
    else:
        nd, sd = nd_il10, sd_il10

    ntc_direction = "up" if nd > 0 else "down"
    same_sign = (sd > 0) == (nd > 0)
    effect = "enhanced" if same_sign else "dampened"
    category = f"{effect}_{ntc_direction}"

    combined_rows.append({"edit": edit, "gene": gene, "ntc_delta": nd, "stim_delta": sd, "category": category})

combined_df = pd.DataFrame(combined_rows)
combined_out_csv = f"{out_dir}/ntc_cytokine_edit_effect_combined_union.csv"
combined_df.to_csv(combined_out_csv, index=False)
print(f"Saved: {combined_out_csv}")

combined_counts = combined_df.groupby(["edit", "category"]).size().unstack(fill_value=0)
combined_counts = combined_counts.reindex(columns=CATEGORY_ORDER, fill_value=0).reindex(index=ALL_EDITS)
print("\nCombined (union) counts per edit/category:")
print(combined_counts)

# =============================================================================
# FINAL PLOT: combined (IL6 U IL10) proportion stacked bar
# =============================================================================
combined_prop = combined_counts.div(combined_counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(7, 5))
stacked_bar(ax, combined_prop.index, combined_prop, CATEGORY_ORDER)
for i, edit in enumerate(combined_prop.index):
    ax.text(i, 1.02, f"n={int(combined_counts.loc[edit].sum())}", ha="center", fontsize=8)
ax.set_xlabel("Edit")
ax.set_ylabel("Proportion of genes")
ax.set_ylim(0, 1.1)
ax.set_title("IL6 ∪ IL10: effect of each edit on NTC's own orange/yellow genes (proportions)")
ax.legend(title="Category", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

fig_out = f"{out_dir}/ntc_cytokine_edit_effect_stackedBarPlot.png"
svg_out = f"{out_dir}/ntc_cytokine_edit_effect_stackedBarPlot.svg"
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.savefig(svg_out, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_out}")

print("\nDone.")
