"""
Reactome pathway (m2.cp MSigDB GMT file) over-representation analysis
for EACH meta-cluster from the Stat3Y705H NTC-DEG heatmap
(NTC_degs_STAT3Y705H_heatmap.py's saved
common_degs_14clusters_8metaClusters_ordered.csv) -- one group per
meta-cluster (not per edit x category, since there's a single edit
here and meta-cluster IS the grouping). ONE individual bar-chart plot
per meta-cluster, using that cluster's own meta_cluster_color from the
saved CSV for visual consistency with the original heatmap.

Standard ORA: one-sided Fisher's exact test per (meta_cluster, pathway),
background = all genes actually measured in this RNA-seq experiment
intersected with the GMT's own gene universe -- this background is
inherently a fixed constant already (it never depends on which groups
are being compared). BH-corrected across all tests.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ---- data sources ----
meta_cluster_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/NTCdegs_STAT3Y705H_heatmap/common_degs_14clusters_8metaClusters_ordered.csv"
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
GMT_FILE = "/ix/djishnu/Aaron_F/proteomics_proj/Data/gene_sets/m2.cp.v2026.1.Mm.symbols.gmt"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig4/STAT3Y705H_metaCluster_pathway_ORA"
os.makedirs(out_dir, exist_ok=True)

TOP_N_PATHWAYS = 15

# ---- background: every gene actually measured in this experiment ----
raw_data = pd.read_csv(raw_data_f, index_col=0)
measured_genes = set(raw_data.index)
print(f"Measured genes (RNA-seq background): {len(measured_genes)}")

# ---- parse the GMT file ----
def parse_gmt(path):
    """GMT format: pathway_name \\t description \\t gene1 \\t gene2 \\t ..."""
    pathway_genes = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            pathway_name = fields[0]
            genes = set(fields[2:])
            pathway_genes[pathway_name] = genes
    return pathway_genes


pathway_genes_raw = parse_gmt(GMT_FILE)
print(f"Pathways in GMT: {len(pathway_genes_raw)}")

# subset to Reactome-sourced pathways only -- MSigDB's C2:CP collection
# bundles multiple databases together, prefixing each pathway name with
# its source (REACTOME_..., KEGG_..., BIOCARTA_..., PID_..., WP_...)
pathway_genes_raw = {p: g for p, g in pathway_genes_raw.items() if p.startswith("REACTOME_")}
print(f"Reactome-sourced pathways: {len(pathway_genes_raw)}")
if len(pathway_genes_raw) == 0:
    raise SystemExit("No pathways matched the 'REACTOME_' prefix -- check the GMT file's actual naming convention "
                      "(e.g. print a few raw pathway_genes_raw keys before filtering).")

# restrict each pathway to genes actually measured in this experiment --
# this IS the background correction (a pathway's "size" for the test is
# only the portion of it we could have possibly detected)
pathway_genes = {p: genes & measured_genes for p, genes in pathway_genes_raw.items()}
pathway_genes = {p: g for p, g in pathway_genes.items() if len(g) >= 3}  # skip tiny/unmeasurable pathways
print(f"Pathways with >=3 measured genes: {len(pathway_genes)}")

gmt_universe = set().union(*pathway_genes.values()) & measured_genes
N = len(gmt_universe)
print(f"ORA background (measured genes appearing in >=1 tested pathway): {N}")

# ---- load meta-cluster assignments -- ONE group per meta-cluster ----
meta_clus_df = pd.read_csv(meta_cluster_f)
print(f"\nGenes with a meta-cluster assignment: {len(meta_clus_df)}")

cluster_order = sorted(meta_clus_df["meta_cluster"].unique())
cluster_color = meta_clus_df.groupby("meta_cluster")["meta_cluster_color"].first().to_dict()

groups = {}
for cl in cluster_order:
    groups[cl] = set(meta_clus_df.loc[meta_clus_df["meta_cluster"] == cl, "gene"])

for cl, genes in groups.items():
    in_bg = genes & gmt_universe
    print(f"Meta-cluster {cl}: {len(genes)} genes, {len(in_bg)} in ORA background "
          f"({len(genes) - len(in_bg)} not measured/not in any tested pathway)")

# =============================================================================
# ORA: one-sided Fisher's exact per (meta_cluster, pathway)
# =============================================================================
all_results = []
for cl, genes in groups.items():
    group_genes = genes & gmt_universe
    n_group = len(group_genes)

    for pathway, p_genes in pathway_genes.items():
        K = len(p_genes)
        a = len(group_genes & p_genes)
        if a == 0:
            continue  # skip zero-overlap pathways -- nothing to test meaningfully
        b = n_group - a
        c = K - a
        d = N - n_group - c

        odds_ratio, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        all_results.append({
            "meta_cluster": cl, "pathway": pathway,
            "k_overlap": a, "n_group_genes": n_group, "K_pathway_size": K, "N_background": N,
            "odds_ratio": odds_ratio, "p_value": p,
        })

results_df = pd.DataFrame(all_results)
results_df["padj"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
results_df["neg_log10_padj"] = -np.log10(results_df["padj"].clip(lower=1e-300))
results_df = results_df.sort_values(["meta_cluster", "padj"])

out_csv = f"{out_dir}/STAT3Y705H_metaCluster_pathway_ORA.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print("\nTop hits (padj < 0.05):")
print(results_df[results_df["padj"] < 0.05]
      [["meta_cluster", "pathway", "k_overlap", "K_pathway_size", "odds_ratio", "padj"]]
      .to_string(index=False))

# =============================================================================
# ONE individual plot per meta-cluster -- one file per cluster
# =============================================================================
for cl in cluster_order:
    sub = results_df[results_df["meta_cluster"] == cl].nsmallest(TOP_N_PATHWAYS, "padj")
    sub = sub.sort_values("neg_log10_padj")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(sub["pathway"], sub["neg_log10_padj"], color=cluster_color[cl])
    for i, (k, K) in enumerate(zip(sub["k_overlap"], sub["K_pathway_size"])):
        ax.text(sub["neg_log10_padj"].iloc[i] + 0.05, i, f"{k}/{K}", va="center", fontsize=7)

    ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("-log10(padj)")
    ax.set_title(f"Meta-cluster {cl}\ntop {TOP_N_PATHWAYS} pathways by padj", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()

    fig_out = f"{out_dir}/metaCluster{cl}_pathway_ORA_barplot.png"
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/metaCluster{cl}_pathway_ORA_barplot.svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {fig_out}")

print("\nDone.")
