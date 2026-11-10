"""
Fig4: adapted from an older notebook (be_effect_box_plot.ipynb) that
compared the BE-edit's own |log2FC| between NTC-upregulated and
NTC-downregulated genes, for a single cytokine (IL10) using a separate
edit-vs-NTC DESeq2 table.

Genes = ALL NTC DEGs (adjusted_padj < 0.05 AND 2**abs(log2FoldChange) >
1.5) for that cytokine, from its own IL6_NTC_vs_PBS_NTC or
IL10_NTC_vs_PBS_NTC result table -- NOT further restricted to Stat3's
own orange/yellow genes, and NOT unioned across IL6/IL10. TWO SEPARATE
analyses/boxplots, one per cytokine, each using only that cytokine's
own DEG set and its own log2FoldChange sign for the up/down split.

Plotted magnitude: |stim_delta| (Stat3 edit vs cytokine-matched NTC,
same cytokine as the DEG call, same batch), computed FRESH from raw
counts (size-factor normalized log2_norm) rather than pulled from a
separate DESeq2 result table -- same approach as
ntc_cytokine_edit_effect_stackedBarPlot.py. Generalized to handle
Stat3Y705H spanning multiple batches (B1 and B2): each individual
Stat3 replicate is centered against ITS OWN batch's NTC mean, then
those per-replicate fold-change values are averaged across all
replicates (both batches) into one gene-level scalar.

For each cytokine: one boxplot, |stim_delta| for "NTC-upregulated" vs
"NTC-downregulated" genes (direction from that cytokine's own DESeq2
log2FoldChange sign), with a one-sided Mann-Whitney U test
(upregulated > downregulated) and Cliff's delta effect size.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ---- data sources ----
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"

il6_deg_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq_DEG_analysis/DEG_results/pairwise/IL6_NTC_vs_PBS_NTC/resultsOrdered_IL6_NTC_vs_PBS_NTC.txt"
il10_deg_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq_DEG_analysis/DEG_results/pairwise/IL10_NTC_vs_PBS_NTC/resultsOrdered_IL10_NTC_vs_PBS_NTC.txt"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/STAT3Y705H_be_effect_boxplot"
os.makedirs(out_dir, exist_ok=True)

TF = "Stat3Y705H"
NTC_LABEL = "NTC"

padj_thresh = 0.05
fc_thresh = 1.5  # applied as 2**abs(log2FoldChange) > fc_thresh

# ---- raw counts, size-factor normalized (NO ComBat) -- same as
# ntc_cytokine_edit_effect_stackedBarPlot.py ----
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


# ---- Stat3's own edit-vs-NTC fold change, computed fresh from raw
# counts -- generalized for Stat3 spanning multiple batches: each
# individual Stat3 replicate is centered against ITS OWN batch's NTC
# mean, then those per-replicate fold-change values are averaged
# across ALL replicates (both batches) into one gene-level scalar. ----
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


def run_be_effect_analysis(stim, deg_f):
    print(f"\n{'='*70}\n{stim}\n{'='*70}")

    # ---- NTC DEGs for THIS cytokine only: adjusted_padj < 0.05 AND
    # 2**abs(log2FoldChange) > 1.5 -- no orange/yellow restriction, no
    # union with the other cytokine ----
    deg_res = pd.read_csv(deg_f, index_col=0)
    sig = deg_res[
        (deg_res["adjusted_padj"] < padj_thresh) & (2 ** deg_res["log2FoldChange"].abs() > fc_thresh)
    ]
    genes = [g for g in sig.index if g in log2_norm.index]
    print(f"NTC DEGs (adjusted_padj<{padj_thresh}, FC>{fc_thresh}x): {len(genes)}")

    # ---- direction from THIS cytokine's own DESeq2 log2FoldChange sign ----
    ntc_direction = {g: ("up" if sig.loc[g, "log2FoldChange"] > 0 else "down") for g in genes}

    # ---- Stat3's stim_delta AT THIS cytokine ----
    sd_series = stim_delta_for_stim(stim)

    rows = []
    for g in genes:
        sd = sd_series.loc[g]
        rows.append({"gene": g, "ntc_direction": ntc_direction[g], "stim_delta": sd, "abs_stim_delta": abs(sd)})
    gene_df = pd.DataFrame(rows)

    gene_csv = f"{out_dir}/{TF}_{stim}_be_effect_perGene.csv"
    gene_df.to_csv(gene_csv, index=False)
    print(f"Saved: {gene_csv}")

    pos_vals = gene_df.loc[gene_df["ntc_direction"] == "up", "abs_stim_delta"]
    neg_vals = gene_df.loc[gene_df["ntc_direction"] == "down", "abs_stim_delta"]

    plot_df = pd.DataFrame({
        "group": ["Positive DEGs"] * len(pos_vals) + ["Negative DEGs"] * len(neg_vals),
        "abs_edit_log2fc": list(pos_vals) + list(neg_vals),
    })

    # one-sided Mann-Whitney U: positive DEGs have larger |edit effect|
    # than negative DEGs
    stat, pval = mannwhitneyu(pos_vals, neg_vals, alternative="greater")

    print(f"Number of positive (NTC-up) DEGs: {len(pos_vals)}")
    print(f"Number of negative (NTC-down) DEGs: {len(neg_vals)}")
    print(f"Median |stim_delta| (positive DEGs): {pos_vals.median():.3f}")
    print(f"Median |stim_delta| (negative DEGs): {neg_vals.median():.3f}")
    print(f"Mann-Whitney U statistic: {stat:.3f}")
    print(f"One-sided p-value (positive > negative): {pval:.3e}")

    delta = cliffs_delta(pos_vals, neg_vals)
    print(f"Cliff's delta: {delta:.3f}")

    # =========================================================================
    # FINAL boxplot -- same clean style as the old notebook's last cell
    # =========================================================================
    sns.set(style="white", context="talk")

    plt.figure(figsize=(5, 6))

    sns.boxplot(
        data=plot_df,
        x="group",
        y="abs_edit_log2fc",
        width=0.5,
        showfliers=False,
        boxprops=dict(facecolor='none', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.5),
    )

    y_max = plot_df["abs_edit_log2fc"].quantile(0.99)
    plt.text(
        0.5, y_max * 0.95,
        f"p = {pval:.2e}\nCliff's delta = {delta:.3f}",
        ha="center", va="top", fontsize=11,
    )

    plt.ylabel(f"Absolute {TF} edit log2FC ({stim} vs cytokine-matched NTC)")
    plt.xlabel("")
    plt.title(stim)
    plt.grid(False)
    sns.despine()
    plt.tight_layout()

    png_f = f"{out_dir}/{TF}_{stim}_be_effect_boxplot.png"
    svg_f = f"{out_dir}/{TF}_{stim}_be_effect_boxplot.svg"
    plt.savefig(png_f, dpi=300, bbox_inches="tight")
    plt.savefig(svg_f, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {png_f}")


run_be_effect_analysis("IL6", il6_deg_f)
run_be_effect_analysis("IL10", il10_deg_f)

print("\nDone.")
