"""
REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION pathway genes (from the m2.cp
MSigDB GMT file), restricted to genes that are orange/yellow for
Arid3aT132A (unioned across IL6 and IL10), fold change vs cytokine-
matched NTC (same batch, same cytokine -- edit_replicate - mean(NTC
replicates, same batch, same cytokine)), individual replicates, no
averaging. Arid3aT132A ONLY -- no Erf, no pairwise comparison. Same
fold-change/heatmap approach as erf_arid3a_foldchange_heatmaps.py, just
with the gene set coming from this one Reactome pathway intersected
with Arid3a's own orange/yellow classification, rather than a
pairwise-different gene list, and restricted to a single edit.
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
GMT_FILE = "/ix/djishnu/Aaron_F/proteomics_proj/Data/gene_sets/m2.cp.v2026.1.Mm.symbols.gmt"
PATHWAY_NAME = "REACTOME_CHEMOKINE_RECEPTORS_BIND_CHEMOKINES"
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = f"/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6/arid3a_pathway_foldchange/{PATHWAY_NAME}"
os.makedirs(out_dir, exist_ok=True)

TF = "Arid3aT132A"
NTC_LABEL = "NTC"

METRIC = "euclidean"
METHOD = "ward"

CYTOKINE_COLOR = {"IL6": "#F4C430", "IL10": "#88CCEE"}
NTC_DIRECTION_COLOR = {"up": "#B2182B", "down": "#2166AC"}

# ---- raw counts, size-factor normalized (NO ComBat) -- same as erf_arid3a_foldchange_heatmaps.py ----
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


batch_tf = col_meta.loc[col_meta["pert"] == TF, "batch"].unique()
assert len(batch_tf) == 1, f"expected exactly one batch for {TF}"
batch_tf = batch_tf[0]
print(f"{TF} batch: {batch_tf}")

# ---- NTC's own up/down direction at each cytokine, vs its own PBS
# baseline -- restricted to this edit's own batch (only one batch here,
# unlike the two-edit script which pooled across both edits' batches) ----
def ntc_delta(stim):
    ntc_stim_cols = get_raw_group_cols(stim, NTC_LABEL, batch_tf)
    ntc_pbs_cols = get_raw_group_cols("PBS", NTC_LABEL, batch_tf)
    return log2_norm[ntc_stim_cols].mean(axis=1) - log2_norm[ntc_pbs_cols].mean(axis=1)


ntc_delta_il6 = ntc_delta("IL6")
ntc_delta_il10 = ntc_delta("IL10")


def ntc_direction_row_colors(genes):
    il6_dir = ["up" if ntc_delta_il6.loc[g] > 0 else "down" for g in genes]
    il10_dir = ["up" if ntc_delta_il10.loc[g] > 0 else "down" for g in genes]
    return pd.DataFrame({
        "IL6 NTC direction": [NTC_DIRECTION_COLOR[d] for d in il6_dir],
        "IL10 NTC direction": [NTC_DIRECTION_COLOR[d] for d in il10_dir],
    }, index=genes)


# ---- parse the GMT file, pull out just this one pathway's genes ----
def parse_gmt(path):
    """GMT format: pathway_name \\t description \\t gene1 \\t gene2 \\t ..."""
    pathway_genes = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            pathway_genes[fields[0]] = set(fields[2:])
    return pathway_genes


pathway_genes_raw = parse_gmt(GMT_FILE)
if PATHWAY_NAME not in pathway_genes_raw:
    raise SystemExit(f"'{PATHWAY_NAME}' not found in the GMT file -- check exact name/capitalization "
                      f"(e.g. print a few keys containing 'EXTRACELLULAR_MATRIX' to find the right one).")

pathway_genes = pathway_genes_raw[PATHWAY_NAME]
print(f"{PATHWAY_NAME}: {len(pathway_genes)} genes in the GMT")

# ---- restrict to Arid3a's own orange/yellow genes, unioned across IL6
# and IL10 -- same convention used throughout this project (e.g.
# ntc_cytokine_edit_effect_stackedBarPlot.py, oy_venn3_stat3_erf_arid3a.py)
def load_orange_yellow(tf, stim):
    f = f"{interaction_dir_base}/{tf}/{tf}_{stim}_fourCategory_classification_data.csv"
    df = pd.read_csv(f, index_col=0)
    return set(df.index[df["category"].isin(["orange", "yellow"])])


arid3a_oy_genes = set()
for stim in ["IL6", "IL10"]:
    stim_genes = load_orange_yellow(TF, stim)
    print(f"  {TF} / {stim}: orange/yellow = {len(stim_genes)}")
    arid3a_oy_genes |= stim_genes
print(f"  {TF}: orange/yellow (IL6 U IL10) = {len(arid3a_oy_genes)}")

genes = sorted(pathway_genes & set(log2_norm.index) & arid3a_oy_genes)
print(f"  {len(genes)} pathway genes present in the RNA-seq data AND orange/yellow for {TF} "
      f"(of {len(pathway_genes)} total pathway genes)")


def build_foldchange_matrix(genes, cytokines):
    """For each gene, Arid3a's replicate values minus that cytokine's
    own NTC mean (same batch) -- fold change vs cytokine-matched NTC,
    individual replicates, no averaging."""
    genes = sorted(g for g in genes if g in log2_norm.index)

    col_order, col_stim = [], []
    for stim in cytokines:
        cols = get_raw_group_cols(stim, TF, batch_tf)
        col_order.extend(cols)
        col_stim.extend([stim] * len(cols))

    fc_df = pd.DataFrame(index=genes, columns=col_order, dtype=float)
    for g in genes:
        for stim in cytokines:
            ntc_cols = get_raw_group_cols(stim, NTC_LABEL, batch_tf)
            ntc_mean = log2_norm.loc[g, ntc_cols].mean()
            edit_cols = get_raw_group_cols(stim, TF, batch_tf)
            fc_df.loc[g, edit_cols] = log2_norm.loc[g, edit_cols].values - ntc_mean

    return fc_df, col_stim, col_order


def plot_heatmap(fc_df, col_stim, col_order, title, out_name):
    row_link = linkage(pdist(fc_df.values, metric=METRIC), method=METHOD)
    row_order = leaves_list(row_link)
    ordered_genes = [fc_df.index[i] for i in row_order]
    fc_df = fc_df.loc[ordered_genes]

    row_colors_df = ntc_direction_row_colors(ordered_genes)

    stim_colors = [CYTOKINE_COLOR[s] for s in col_stim]
    col_colors_df = pd.DataFrame({"Cytokine": stim_colors}, index=col_order)

    vabs = max(fc_df.abs().values.max(), 0.1)
    c = sns.color_palette("RdBu_r", as_cmap=True)

    hm = sns.clustermap(
        fc_df,
        row_cluster=False,  # already clustered above
        col_cluster=False,  # fixed cytokine block order
        row_colors=row_colors_df,
        col_colors=col_colors_df,
        cmap=c,
        z_score=None,
        vmin=-vabs, vmax=vabs,
        yticklabels=True,  # small pathway gene set -- always show labels
        xticklabels=False,  # individual replicate columns -- see col_colors legend instead
        figsize=(7, 10),
    )
    hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_yticklabels(), fontsize=7)

    cytokines_present = sorted(set(col_stim))
    stim_legend = [mpatches.Patch(color=CYTOKINE_COLOR[s], label=s) for s in cytokines_present]
    l1 = hm.ax_heatmap.legend(handles=stim_legend, title="Cytokine", bbox_to_anchor=(1.25, 1.0),
                               loc="upper left", fontsize=8, title_fontsize=9)
    hm.ax_heatmap.add_artist(l1)

    ntc_dir_legend = [mpatches.Patch(color=NTC_DIRECTION_COLOR[d], label=d) for d in ["up", "down"]]
    hm.ax_heatmap.legend(handles=ntc_dir_legend, title="NTC direction\n(vs own PBS)", bbox_to_anchor=(1.25, 0.7),
                          loc="upper left", fontsize=8, title_fontsize=9)

    hm.fig.suptitle(f"{title}\nn={len(ordered_genes)} genes, fold change vs cytokine-matched NTC "
                     f"(same batch, individual replicates)\nclustering: {METRIC} distance, {METHOD} linkage", y=1.02)

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


# both cytokines, Arid3a only
fc_both, stim_both, cols_both = build_foldchange_matrix(genes, ["IL6", "IL10"])
plot_heatmap(fc_both, stim_both, cols_both,
             f"{TF} -- {PATHWAY_NAME} genes, both cytokines",
             f"{TF}_{PATHWAY_NAME}")

# =============================================================================
# Top 10 genes by |IL10 fold change| (edit vs NTC) -- dot plot panel,
# raw log2_norm values (NO ComBat, same data as the heatmap above), 5
# groups per gene: PBS_NTC, IL6_NTC, IL10_NTC, IL6_edit, IL10_edit.
# Values are NOT centered/subtracted -- these are the actual raw
# log2-normalized expression levels for each group, plotted directly.
# =============================================================================
il10_ntc_cols = get_raw_group_cols("IL10", NTC_LABEL, batch_tf)
il10_edit_cols = get_raw_group_cols("IL10", TF, batch_tf)

il10_fc = {
    g: log2_norm.loc[g, il10_edit_cols].mean() - log2_norm.loc[g, il10_ntc_cols].mean()
    for g in genes
}

top10_genes = sorted(genes, key=lambda g: abs(il10_fc[g]), reverse=True)[:10]
print(f"\nTop 10 {TF} genes by |IL10 fold change| (edit vs NTC):")
for g in top10_genes:
    print(f"  {g}: log2FC = {il10_fc[g]:.3f}")

GROUP_ORDER = [("PBS", NTC_LABEL), ("IL10", NTC_LABEL), ("IL10", TF)]
GROUP_COLOR = {"PBS": "#B0B0B0", "IL6": CYTOKINE_COLOR["IL6"], "IL10": CYTOKINE_COLOR["IL10"]}
GROUP_MARKER = {NTC_LABEL: "o", TF: "s"}

group_cols_cache = {(stim, pert): get_raw_group_cols(stim, pert, batch_tf) for stim, pert in GROUP_ORDER}

ncols = 5
nrows = int(np.ceil(len(top10_genes) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
axes = axes.flatten()

for i, g in enumerate(top10_genes):
    ax = axes[i]
    for x, (stim, pert) in enumerate(GROUP_ORDER):
        cols = group_cols_cache[(stim, pert)]
        if len(cols) == 0:
            continue
        vals = log2_norm.loc[g, cols].values.astype(float)
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals,
                   color=GROUP_COLOR[stim], marker=GROUP_MARKER[pert],
                   s=40, edgecolors="black", linewidths=0.5, alpha=0.85, zorder=3)
        ax.plot([x - 0.15, x + 0.15], [vals.mean(), vals.mean()],
                color="black", linewidth=1.5, zorder=4)

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([f"{pert}\n{stim}" for stim, pert in GROUP_ORDER], fontsize=7)
    ax.set_title(f"{g}\nIL10 log2FC={il10_fc[g]:.2f}", fontsize=9, fontweight="bold")
    ax.set_ylabel("log2 normalized\nexpression", fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(len(top10_genes), len(axes)):
    axes[j].axis("off")

group_legend = [plt.Line2D([0], [0], marker=GROUP_MARKER[p], color="grey", linestyle="none",
                            markersize=7, label=p) for p in [NTC_LABEL, TF]]
cytokine_legend = [plt.Line2D([0], [0], marker="o", color=GROUP_COLOR[s], linestyle="none",
                               markersize=7, label=s) for s in ["PBS", "IL10"]]
fig.legend(handles=group_legend + cytokine_legend, loc="lower center",
           ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(f"{TF} -- top 10 {PATHWAY_NAME} genes by |IL10 fold change| (edit vs NTC)\n"
             f"raw log2 normalized expression, no ComBat", fontsize=11, y=1.0)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

dot_png = f"{out_dir}/{TF}_top10_IL10foldchange_dotplot_panel.png"
dot_svg = f"{out_dir}/{TF}_top10_IL10foldchange_dotplot_panel.svg"
plt.savefig(dot_png, dpi=300, bbox_inches="tight")
plt.savefig(dot_svg, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved: {dot_png}")

print("\nDone.")