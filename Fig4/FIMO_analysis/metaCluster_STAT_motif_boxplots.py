"""
STAT motif (FIMO) analysis by NTC-DEG meta-cluster.

Uses the meta-cluster gene assignments saved by
NTC_degs_STAT3Y705H_heatmap.py, and the STAT-family FIMO results
path/parsing logic from fimo_edit_correlation.py's tf_fimo_map entry
for Stat3Y705H.

Two per-gene metrics, BOTH including genes with zero hits as real
datapoints (not excluded/dropped):
  - occurrence count: the raw NUMBER of FIMO hits for the STAT motif per
    gene (0 for genes with no hits at all).
  - score: that gene's BEST (max) FIMO score among its own hits, or 0
    if the gene has no hit at all (0 is used as a "no motif" placeholder
    sitting below the real reported-score range; worth double-checking
    this is a sensible floor for your particular FIMO output before
    trusting the score boxplot's exact shape at the low end).

On cluster-size "domination": both boxplots plot ONE DATAPOINT PER GENE
(not an aggregated per-cluster total), so each box's median/quartiles
already reflect the WITHIN-cluster proportion/distribution -- this is
inherently size-independent. A 500-gene cluster and a 20-gene cluster
with the same true underlying hit rate land at the same box position;
cluster size doesn't inflate a bigger cluster's result. What size DOES
affect is how reliable/noisy that estimate is (fewer genes = a less
stable read on the true rate) -- rather than rescaling the values to
"correct" for this, n= per cluster is annotated on each plot so a
reader can judge that directly.
"""

import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ---- data sources ----
meta_cluster_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/NTCdegs_STAT3Y705H_heatmap/common_degs_14clusters_8metaClusters_ordered.csv"
fimo_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results/8000_TSS_8000_stats_all/fimo.tsv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/STAT3Y705H_metaCluster_FIMO"
os.makedirs(out_dir, exist_ok=True)

# ---- load meta-cluster assignments ----
meta_clus_df = pd.read_csv(meta_cluster_f)
print(f"Genes with a meta-cluster assignment: {len(meta_clus_df)}")
print(meta_clus_df["meta_cluster"].value_counts().sort_index())

# ---- load FIMO results, same parsing convention as fimo_edit_correlation.py ----
fimo_df = pd.read_csv(fimo_f, sep="\t", comment="#")
print(f"\nFIMO hit rows loaded: {len(fimo_df)}")

hit_counts = fimo_df.groupby("sequence_name").size().rename("hit_count")
best_scores = fimo_df.groupby("sequence_name")["score"].max().rename("best_score")

# ---- merge onto the meta-cluster gene list -- genes with NO fimo entry
# at all get hit_count=0 (no motif found anywhere); best_score stays
# NaN for those (there's no score to report if there's no hit) ----
meta_clus_df = meta_clus_df.set_index("gene")
meta_clus_df["hit_count"] = hit_counts.reindex(meta_clus_df.index).fillna(0)
meta_clus_df["best_score"] = best_scores.reindex(meta_clus_df.index).fillna(0)  # 0 if no hit -- contributes, not dropped
meta_clus_df["has_motif"] = (meta_clus_df["hit_count"] >= 1).astype(int)

n_with_hit = meta_clus_df["has_motif"].sum()
print(f"Genes with >=1 STAT motif hit: {n_with_hit} / {len(meta_clus_df)}")

meta_clus_df = meta_clus_df.reset_index()
meta_clus_df.to_csv(f"{out_dir}/metaCluster_STAT_fimo_perGene.csv", index=False)
print(f"Saved: {out_dir}/metaCluster_STAT_fimo_perGene.csv")

cluster_order = sorted(meta_clus_df["meta_cluster"].unique())
n_per_cluster = meta_clus_df.groupby("meta_cluster").size()

# =============================================================================
# Boxplot 1: motif OCCURRENCE COUNT -- number of FIMO hits per gene
# (continuous, includes 0 for genes with no hits at all)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=meta_clus_df, x="meta_cluster", y="hit_count", order=cluster_order,
            color="#3A7EBF", width=0.5, ax=ax)

y_top = meta_clus_df["hit_count"].max()
for i, cl in enumerate(cluster_order):
    ax.text(i, y_top * 1.05, f"n={n_per_cluster[cl]}", ha="center", fontsize=8)

ax.set_xlabel("Meta-cluster")
ax.set_ylabel("Number of STAT motif hits")
ax.set_title("STAT motif occurrence count by NTC-DEG meta-cluster\n"
             "one datapoint per gene (0 for genes with no hits) -- box position "
             "reflects the WITHIN-cluster distribution, already size-independent; "
             "n= shown for reliability")
plt.tight_layout()

occ_png = f"{out_dir}/metaCluster_STAT_motif_occurrenceCount_boxplot.png"
occ_svg = f"{out_dir}/metaCluster_STAT_motif_occurrenceCount_boxplot.svg"
plt.savefig(occ_png, dpi=300, bbox_inches="tight")
plt.savefig(occ_svg, bbox_inches="tight")
plt.show()
print(f"Saved: {occ_png}")

# =============================================================================
# Boxplot 2: motif SCORE -- best FIMO score per gene (0 for genes with
# no hit -- they contribute a real datapoint, not excluded)
# =============================================================================
n_per_cluster_scored = n_per_cluster

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=meta_clus_df, x="meta_cluster", y="best_score", order=cluster_order,
            color="#E84545", width=0.5, ax=ax)

y_top = meta_clus_df["best_score"].max()
for i, cl in enumerate(cluster_order):
    ax.text(i, y_top * 1.05, f"n={n_per_cluster[cl]}", ha="center", fontsize=8)

ax.set_xlabel("Meta-cluster")
ax.set_ylabel("Best FIMO score (0 for genes with no hit)")
ax.set_title("STAT motif score by NTC-DEG meta-cluster\n"
             "one datapoint per gene -- genes with no hit contribute a score of 0, "
             "not excluded")
plt.tight_layout()

score_png = f"{out_dir}/metaCluster_STAT_motif_score_boxplot.png"
score_svg = f"{out_dir}/metaCluster_STAT_motif_score_boxplot.svg"
plt.savefig(score_png, dpi=300, bbox_inches="tight")
plt.savefig(score_svg, bbox_inches="tight")
plt.show()
print(f"Saved: {score_png}")

# =============================================================================
# Pairwise tests between clusters -- Mann-Whitney U (non-parametric,
# appropriate given these are skewed count/score distributions with a
# lot of zeros), BH-corrected across all C(n,2) pairs, done separately
# per metric. 28 pairs for 8 clusters is too many to annotate as
# brackets directly on the boxplot without it becoming unreadable, so
# results are saved to CSV and shown as a separate -log10(padj) heatmap
# per metric instead.
# =============================================================================
def pairwise_tests(df, value_col, cluster_order):
    rows = []
    for cl_a, cl_b in itertools.combinations(cluster_order, 2):
        vals_a = df.loc[df["meta_cluster"] == cl_a, value_col].values
        vals_b = df.loc[df["meta_cluster"] == cl_b, value_col].values
        stat, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        rows.append({"cluster_a": cl_a, "cluster_b": cl_b, "n_a": len(vals_a), "n_b": len(vals_b),
                     "U_stat": stat, "p_value": p})
    result = pd.DataFrame(rows)
    result["padj"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result.sort_values("padj")


def plot_pairwise_heatmap(pairwise_df, cluster_order, metric_label, out_name):
    n = len(cluster_order)
    padj_matrix = pd.DataFrame(np.nan, index=cluster_order, columns=cluster_order)
    for _, row in pairwise_df.iterrows():
        padj_matrix.loc[row["cluster_a"], row["cluster_b"]] = row["padj"]
        padj_matrix.loc[row["cluster_b"], row["cluster_a"]] = row["padj"]

    neg_log10_padj = -np.log10(padj_matrix.astype(float).clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(neg_log10_padj, annot=padj_matrix.round(3), fmt="", cmap="Reds",
                mask=padj_matrix.isna(), cbar_kws={"label": "-log10(padj)"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"Pairwise cluster comparisons -- {metric_label}\n"
                 f"Mann-Whitney U, BH-corrected across all {len(pairwise_df)} pairs "
                 f"(cell values = padj)")
    ax.set_xlabel("Meta-cluster")
    ax.set_ylabel("Meta-cluster")
    plt.tight_layout()

    heat_png = f"{out_dir}/{out_name}.png"
    heat_svg = f"{out_dir}/{out_name}.svg"
    plt.savefig(heat_png, dpi=300, bbox_inches="tight")
    plt.savefig(heat_svg, bbox_inches="tight")
    plt.show()
    print(f"Saved: {heat_png}")


pairwise_hit_count = pairwise_tests(meta_clus_df, "hit_count", cluster_order)
pairwise_hit_count.to_csv(f"{out_dir}/metaCluster_STAT_hitCount_pairwise_tests.csv", index=False)
print(f"\nSaved: {out_dir}/metaCluster_STAT_hitCount_pairwise_tests.csv")
print("Significant pairs (padj < 0.05), hit_count:")
print(pairwise_hit_count[pairwise_hit_count["padj"] < 0.05].to_string(index=False))
plot_pairwise_heatmap(pairwise_hit_count, cluster_order, "motif occurrence count",
                      "metaCluster_STAT_hitCount_pairwise_heatmap")

pairwise_score = pairwise_tests(meta_clus_df, "best_score", cluster_order)
pairwise_score.to_csv(f"{out_dir}/metaCluster_STAT_score_pairwise_tests.csv", index=False)
print(f"\nSaved: {out_dir}/metaCluster_STAT_score_pairwise_tests.csv")
print("Significant pairs (padj < 0.05), best_score:")
print(pairwise_score[pairwise_score["padj"] < 0.05].to_string(index=False))
plot_pairwise_heatmap(pairwise_score, cluster_order, "motif score",
                      "metaCluster_STAT_score_pairwise_heatmap")

print("\nDone.")