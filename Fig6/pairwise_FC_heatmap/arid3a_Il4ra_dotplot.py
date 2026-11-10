"""
Single-gene dot plot for Il4ra -- Arid3aT132A, both IL6 and IL10, raw
log2-normalized expression (NO ComBat), 5 groups: PBS_NTC, IL6_NTC,
IL10_NTC, IL6_edit, IL10_edit. Same style/data source as the dot plot
panel in arid3a_editGenes_and_pathwayGenes_foldchange_heatmap.py, just
for one specific gene instead of a top-10 selection, and both cytokines
shown instead of IL10 only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- data sources ----
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6/arid3a_pathway_foldchange/singleGene_dotplots"
os.makedirs(out_dir, exist_ok=True)

TF = "TfebS392P" # Arid3aT132A, Etv3S29P, 
NTC_LABEL = "NTC"
GENE = "Tnfrsf1a"

CYTOKINE_COLOR = {"IL6": "#F4C430", "IL10": "#88CCEE"}

# ---- raw counts, size-factor normalized (NO ComBat) -- same as the source script ----
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

if GENE not in log2_norm.index:
    raise SystemExit(f"'{GENE}' not found in the expression data -- check gene symbol/capitalization.")

# ---- fold changes, for reference in the title ----
il6_ntc_cols = get_raw_group_cols("IL6", NTC_LABEL, batch_tf)
il6_edit_cols = get_raw_group_cols("IL6", TF, batch_tf)
il10_ntc_cols = get_raw_group_cols("IL10", NTC_LABEL, batch_tf)
il10_edit_cols = get_raw_group_cols("IL10", TF, batch_tf)

il6_fc = log2_norm.loc[GENE, il6_edit_cols].mean() - log2_norm.loc[GENE, il6_ntc_cols].mean()
il10_fc = log2_norm.loc[GENE, il10_edit_cols].mean() - log2_norm.loc[GENE, il10_ntc_cols].mean()
print(f"{GENE}: IL6 log2FC (edit vs NTC) = {il6_fc:.3f}  |  IL10 log2FC (edit vs NTC) = {il10_fc:.3f}")

# ---- 5 groups: PBS_NTC, IL6_NTC, IL10_NTC, IL6_edit, IL10_edit ----
GROUP_ORDER = [("PBS", NTC_LABEL), ("PBS", TF), ("IL6", NTC_LABEL), ("IL10", NTC_LABEL), ("IL6", TF), ("IL10", TF)]
GROUP_COLOR = {"PBS": "#B0B0B0", "IL6": CYTOKINE_COLOR["IL6"], "IL10": CYTOKINE_COLOR["IL10"]}
GROUP_MARKER = {NTC_LABEL: "o", TF: "s"}

group_cols_cache = {(stim, pert): get_raw_group_cols(stim, pert, batch_tf) for stim, pert in GROUP_ORDER}

# ---- save the underlying per-sample values plotted below ----
csv_rows = []
for stim, pert in GROUP_ORDER:
    cols = group_cols_cache[(stim, pert)]
    for c in cols:
        csv_rows.append({
            "gene": GENE, "stim": stim, "pert": pert, "sample": c,
            "log2_norm_expression": log2_norm.loc[GENE, c],
        })
values_df = pd.DataFrame(csv_rows)
values_csv = f"{out_dir}/{TF}_{GENE}_dotplot_values.csv"
values_df.to_csv(values_csv, index=False)
print(f"Saved: {values_csv}")

fig, ax = plt.subplots(figsize=(6, 5))

for x, (stim, pert) in enumerate(GROUP_ORDER):
    cols = group_cols_cache[(stim, pert)]
    if len(cols) == 0:
        continue
    vals = log2_norm.loc[GENE, cols].values.astype(float)
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), x) + jitter, vals,
               color=GROUP_COLOR[stim], marker=GROUP_MARKER[pert],
               s=60, edgecolors="black", linewidths=0.5, alpha=0.85, zorder=3)
    ax.plot([x - 0.15, x + 0.15], [vals.mean(), vals.mean()],
            color="black", linewidth=1.5, zorder=4)

ax.set_xticks(range(len(GROUP_ORDER)))
ax.set_xticklabels([f"{pert}\n{stim}" for stim, pert in GROUP_ORDER], fontsize=9)
ax.set_ylabel("log2 normalized expression", fontsize=10)
ax.set_title(f"{GENE} -- {TF}\nIL6 log2FC={il6_fc:.2f}, IL10 log2FC={il10_fc:.2f} (edit vs NTC)",
             fontsize=11, fontweight="bold")
ax.tick_params(labelsize=9)

group_legend = [plt.Line2D([0], [0], marker=GROUP_MARKER[p], color="grey", linestyle="none",
                            markersize=8, label=p) for p in [NTC_LABEL, TF]]
cytokine_legend = [plt.Line2D([0], [0], marker="o", color=GROUP_COLOR[s], linestyle="none",
                               markersize=8, label=s) for s in ["PBS", "IL6", "IL10"]]
ax.legend(handles=group_legend + cytokine_legend, loc="upper left", bbox_to_anchor=(1.02, 1),
          fontsize=9, frameon=False)

plt.tight_layout()

dot_png = f"{out_dir}/{TF}_{GENE}_dotplot.png"
dot_svg = f"{out_dir}/{TF}_{GENE}_dotplot.svg"
plt.savefig(dot_png, dpi=300, bbox_inches="tight")
plt.savefig(dot_svg, bbox_inches="tight")
plt.show()

print(f"Saved: {dot_png}")
print("\nDone.")