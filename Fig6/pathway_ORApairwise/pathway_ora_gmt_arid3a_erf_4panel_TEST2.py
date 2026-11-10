"""
Arid3aT132A and ErfT529A, all FOUR separate categories from the original
enhanced/dampened x up/down classification (enhanced_up, dampened_up,
enhanced_down, dampened_down) -- not collapsed/paired into two combined
groups, each of the 4 tested individually per edit (8 groups total).
Tested for Reactome pathway (from the m2.cp MSigDB GMT file)
over-representation. ONE individual bar-chart plot per (edit, category)
group, 8 separate files total.

Standard ORA: one-sided Fisher's exact test per (group, pathway),
background = all genes actually measured in this RNA-seq experiment
(raw_data.index) intersected with the GMT's own gene universe -- this
background is inherently a fixed constant already (it never depends on
which groups are being compared), so no further "shared background"
fix is needed here, unlike the claude-term version. BH-corrected across
all tests.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ---- data sources ----
combined_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/enhance_dampen_analysis/enhance_dampen_combined_union.csv"
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
GMT_FILE = "/ix/djishnu/Aaron_F/proteomics_proj/Data/gene_sets/m2.cp.v2026.1.Mm.symbols.gmt"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6/pairwise_pathway_ORA_TEST2"
os.makedirs(out_dir, exist_ok=True)

EDIT_A, EDIT_B = "Arid3aT132A", "ErfT529A"
EDIT_COLOR = {EDIT_A: "#6A3D9A", EDIT_B: "#FF7F00"}

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

# ---- load all 8 groups: each edit x each of the 4 original categories ----
combined_df = pd.read_csv(combined_f)

CATEGORIES = ["enhanced_up", "dampened_up", "enhanced_down", "dampened_down"]

def get_gene_set(edit, category):
    mask = (combined_df["edit"] == edit) & (combined_df["category"] == category)
    return set(combined_df.loc[mask, "gene"])

groups = {}
for edit in [EDIT_A, EDIT_B]:
    for category in CATEGORIES:
        label = f"{edit} ({category})"
        groups[label] = get_gene_set(edit, category)

for label, genes in groups.items():
    in_bg = genes & gmt_universe
    print(f"{label}: {len(genes)} genes, {len(in_bg)} in ORA background "
          f"({len(genes) - len(in_bg)} not measured/not in any tested pathway)")

# =============================================================================
# ORA: one-sided Fisher's exact per (group, pathway)
# =============================================================================
all_results = []
for label, genes in groups.items():
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
            "group": label, "pathway": pathway,
            "k_overlap": a, "n_group_genes": n_group, "K_pathway_size": K, "N_background": N,
            "odds_ratio": odds_ratio, "p_value": p,
        })

results_df = pd.DataFrame(all_results)
results_df["padj"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
results_df["neg_log10_padj"] = -np.log10(results_df["padj"].clip(lower=1e-300))
results_df = results_df.sort_values(["group", "padj"])

out_csv = f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_fourCategory_pathway_ORA.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print("\nTop hits (padj < 0.05):")
print(results_df[results_df["padj"] < 0.05]
      [["group", "pathway", "k_overlap", "K_pathway_size", "odds_ratio", "padj"]]
      .to_string(index=False))

# =============================================================================
# ONE individual plot per (edit, category) -- 8 separate files
# =============================================================================
for edit in [EDIT_A, EDIT_B]:
    for category in CATEGORIES:
        label = f"{edit} ({category})"

        sub = results_df[results_df["group"] == label].nsmallest(TOP_N_PATHWAYS, "padj")
        sub = sub.sort_values("neg_log10_padj")

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(sub["pathway"], sub["neg_log10_padj"], color=EDIT_COLOR[edit])
        for i, (k, K) in enumerate(zip(sub["k_overlap"], sub["K_pathway_size"])):
            ax.text(sub["neg_log10_padj"].iloc[i] + 0.05, i, f"{k}/{K}", va="center", fontsize=7)

        ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("-log10(padj)")
        ax.set_title(f"{label}\ntop {TOP_N_PATHWAYS} pathways by padj", fontsize=11)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()

        fig_out = f"{out_dir}/{edit}_{category}_pathway_ORA_barplot.png"
        plt.savefig(fig_out, dpi=300, bbox_inches="tight")
        plt.savefig(f"{out_dir}/{edit}_{category}_pathway_ORA_barplot.svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {fig_out}")

print("\nDone.")