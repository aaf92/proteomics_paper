"""
Per-TF heatmap of orange/yellow genes only (IL10 classification only).

Simplified version of 03_fourCategory_stacked_heatmap_IL10.py: drops the red
category, and does NOT cluster each category separately and stack the
blocks -- orange and yellow genes are pooled and clustered together in one
pass (row_cluster=True), so the dendrogram is free to interleave the two
categories however the expression correlation actually falls out. A row
color sidebar still marks each gene's category for reference.

Columns are the actual samples from PBS_NTC, PBS_<TF>, IL10_NTC, IL10_<TF>,
in that fixed order, with a column color sidebar marking which group each
sample belongs to (no averaging within a group). NTC groups are
batch-matched to this TF's edit, same convention as the source script.

Also produces a baseline_delta vs stim_delta scatter plot per TF, same
style as 02_fourCategory_scatter.py, restricted to orange/yellow genes
only (reuses baseline_delta/stim_delta directly from class_df, which
already carries them from that script's saved output).

One PNG/SVG scatter + one PNG/SVG heatmap + one data CSV is written per TF.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---- data sources ----
data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq/be_effect_analysis/out_files/il6_il10_NTC_ctrl_centered_combatCorrected.csv"
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig5/oy_plots"
os.makedirs(out_dir, exist_ok=True)

tf_list = [
    "Stat3Y705H", "ErfT3A", "ErfT529A", "TfebS392P",
    "Arid3aT132A", "Etv3S29P", "Hivep3S1949P",
]
STIM = "IL10"
NTC_LABEL = "NTC"

CATEGORIES = ["orange", "yellow"]
# CATEGORY_COLORS = {"yellow": "#F4C430", "orange": "#E67E22"}
CATEGORY_COLORS = {"yellow": "#E67E22", "orange": "#E67E22"}

# scatter-only: grey and red genes shown together as one merged grey
# category for context (matching 02_fourCategory_scatter.py's full view,
# but with red folded into grey rather than kept as its own color)
SCATTER_COLOR_MAP = {"grey": "#BBBBBB", "red": "#BBBBBB", "yellow": CATEGORY_COLORS["yellow"], "orange": CATEGORY_COLORS["orange"]}

# ---- load expression data ----
data = pd.read_csv(data_f, index_col=0)
print("Data shape:", data.shape)

all_cols = data.columns.tolist()
col_tokens = [c.split("_") for c in all_cols]
col_meta = pd.DataFrame({
    "sample": all_cols,
    "batch": [t[0] for t in col_tokens],
    "stim":  [t[1] for t in col_tokens],
    "pert":  [t[2] for t in col_tokens],
})


def get_group_cols(group_name, batches_with_edit):
    """group_name like 'PBS_NTC' or 'IL10_<tf>'. NTC groups are restricted
    to the batch(es) this TF's edit was sequenced in; edit groups need no
    extra batch filter since the edit only exists in its own batch(es)."""
    stim_g, pert_g = group_name.split("_", 1)
    mask = (col_meta["stim"] == stim_g) & (col_meta["pert"] == pert_g)
    if pert_g == NTC_LABEL:
        mask &= col_meta["batch"].isin(batches_with_edit)
    return col_meta.loc[mask, "sample"].tolist()


for tf in tf_list:
    class_f = f"{interaction_dir_base}/{tf}/{tf}_{STIM}_fourCategory_classification_data.csv"
    if not os.path.exists(class_f):
        print(f"WARNING: missing {class_f}, skipping {tf}")
        continue

    full_class_df = pd.read_csv(class_f, index_col=0)  # unfiltered -- all 4 categories, for the scatter
    class_df = full_class_df[full_class_df["category"].isin(CATEGORIES)]  # orange/yellow only, for the heatmap
    if len(class_df) == 0:
        print(f"SKIP {tf}: no orange/yellow genes")
        continue

    counts = class_df["category"].value_counts()
    print(f"{tf} ({STIM}): " +
          ", ".join(f"{c}={int(counts.get(c, 0))}" for c in CATEGORIES))

    # =====================================================================
    # scatter plot: baseline_delta vs stim_delta, orange/yellow genes only
    # -- same style as 02_fourCategory_scatter.py, just restricted to
    # these two categories (class_df already carries baseline_delta/
    # stim_delta directly from that script's saved output, no need to
    # reload ranked_f separately)
    # =====================================================================
    tf_out_dir = f"{out_dir}/{tf}"
    os.makedirs(tf_out_dir, exist_ok=True)
 
    fig_s, ax_s = plt.subplots(figsize=(7, 7))
 
    # grey and red combined into one merged grey entry (same color, single
    # legend label with the combined n) -- draw first/lowest zorder, same
    # as 02's convention for grey
    grey_and_red = full_class_df[full_class_df["category"].isin(["grey", "red"])]
    ax_s.scatter(grey_and_red["stim_delta"], grey_and_red["baseline_delta"],
                 c=SCATTER_COLOR_MAP["grey"], s=12, alpha=0.55,
                 label=f"grey (n={len(grey_and_red)})", zorder=1)
 
    # yellow, then orange last (most-trusted call, drawn on top so it's
    # never occluded) -- same convention as 02's script
    for cat in ["yellow", "orange"]:
        sub = full_class_df[full_class_df["category"] == cat]
        ax_s.scatter(sub["stim_delta"], sub["baseline_delta"],
                     c=SCATTER_COLOR_MAP[cat], s=12, alpha=0.85,
                     label=f"{cat} (n={len(sub)})",
                     zorder=3 if cat == "orange" else 2)
 
    lims = [
        min(full_class_df["baseline_delta"].min(), full_class_df["stim_delta"].min()),
        max(full_class_df["baseline_delta"].max(), full_class_df["stim_delta"].max()),
    ]
    ax_s.plot(lims, lims, linestyle="--", color="steelblue", linewidth=1, zorder=0, label="y=x")
    ax_s.axhline(0, color="lightgrey", zorder=0)
    ax_s.axvline(0, color="lightgrey", zorder=0)
    ax_s.set_xlabel("stim_delta")
    ax_s.set_ylabel("baseline_delta")
    ax_s.set_title(f"{tf} / {STIM}\nall genes shown; red folded into grey")
    ax_s.set_aspect("equal", adjustable="box")  # square axes -- y=x line renders at a true 45 degrees
    ax_s.legend(fontsize=8, loc="best")
    plt.tight_layout()
 
    scatter_png = f"{tf_out_dir}/{tf}_{STIM}_orangeYellow_scatter.png"
    scatter_svg = f"{tf_out_dir}/{tf}_{STIM}_orangeYellow_scatter.svg"
    plt.savefig(scatter_png, dpi=300, bbox_inches="tight")
    plt.savefig(scatter_svg, bbox_inches="tight")
    plt.show()
    plt.close(fig_s)
    print(f"  Saved: {scatter_png}")
 
    batches_with_edit = col_meta.loc[col_meta["pert"] == tf, "batch"].unique().tolist()
 
    # individual-sample columns, no averaging -- PBS_NTC, PBS_edit,
    # stim_NTC, stim_edit, in that fixed order. NTC groups are
    # batch-matched to this TF; edit groups need no extra filter.
    group_specs = [
        ("PBS", NTC_LABEL, "#B0B0B0"),
        ("PBS", tf, "#707070"),
        (STIM, NTC_LABEL, "#4477AA"),
        (STIM, tf, "#EE6677"),
    ]
    col_order = []
    col_group_colors = []
    col_group_names = []
    for stim_g, pert_g, color in group_specs:
        grp_name = f"{stim_g}_{pert_g}"
        cols = get_group_cols(grp_name, batches_with_edit)
        if len(cols) == 0:
            print(f"  WARNING: no samples found for group '{grp_name}' ({tf})")
            continue
        col_order.extend(cols)
        col_group_colors.extend([color] * len(cols))
        col_group_names.append(grp_name)
 
    genes = [g for g in class_df.index if g in data.index]
    if len(genes) == 0:
        print(f"SKIP {tf}: no genes remaining after matching to expression data")
        continue
 
    display_df = data.loc[genes, col_order]
    row_colors_series = pd.Series(
        [CATEGORY_COLORS[class_df.loc[g, "category"]] for g in genes], index=display_df.index
    )
    col_colors_series = pd.Series(col_group_colors, index=display_df.columns)
 
    c = sns.color_palette("RdBu_r", as_cmap=True)
 
    hm = sns.clustermap(
        display_df,
        row_cluster=True,
        col_cluster=False,
        metric='correlation',
        method='complete',
        row_colors=row_colors_series,
        col_colors=col_colors_series,
        cmap=c,
        z_score=None,
        vmin=-1.5, vmax=1.5,
        yticklabels=False,
        xticklabels=True,
        figsize=(8, 10),
    )
    hm.ax_heatmap.set_xticklabels(hm.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=6)
 
    category_legend_patches = [mpatches.Patch(color=CATEGORY_COLORS[c_], label=c_) for c_ in CATEGORIES]
    category_legend = hm.ax_heatmap.legend(
        handles=category_legend_patches, title="Category", bbox_to_anchor=(1.25, 1.0),
        loc="upper left", fontsize=8, title_fontsize=9,
    )
    hm.ax_heatmap.add_artist(category_legend)
 
    group_specs_by_name = {f"{s}_{p}": col for s, p, col in group_specs}
    group_legend_patches = [
        mpatches.Patch(color=group_specs_by_name[name], label=name) for name in col_group_names
    ]
    hm.ax_heatmap.legend(
        handles=group_legend_patches, title="Sample group", bbox_to_anchor=(1.25, 0.6),
        loc="upper left", fontsize=8, title_fontsize=9,
    )
 
    hm.fig.suptitle(f"{tf} ({STIM}): orange/yellow genes, clustered together", y=1.05)
    hm.fig.tight_layout()
 
    png_f = f"{tf_out_dir}/{tf}_{STIM}_orangeYellow_heatmap.png"
    svg_f = f"{tf_out_dir}/{tf}_{STIM}_orangeYellow_heatmap.svg"
    plt.savefig(png_f, format="png", bbox_inches="tight", dpi=300)
    plt.savefig(svg_f, format="svg", bbox_inches="tight")
    plt.show()
    plt.close(hm.fig)
 
    row_order = hm.dendrogram_row.reordered_ind
    ordered_genes = [genes[i] for i in row_order]
    save_df = display_df.loc[ordered_genes].copy()
    save_df.insert(0, "category", [class_df.loc[g, "category"] for g in ordered_genes])
    save_df.insert(1, "gene", ordered_genes)
    save_df.to_csv(f"{tf_out_dir}/{tf}_{STIM}_orangeYellow_heatmap_data.csv", index=False)
 
    print(f"  Saved: {png_f}")
    print(f"  Saved: {svg_f}")
 
print("\nDone.")