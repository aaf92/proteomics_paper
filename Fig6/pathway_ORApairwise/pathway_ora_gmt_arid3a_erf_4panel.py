"""
Arid3aT132A and ErfT529A, both enhanced AND dampened gene sets (combined
IL6 U IL10 classification), tested for Reactome pathway (from the m2.cp
MSigDB GMT file) over-representation. Same ORA structure as
pathway_ora_claude_term_arid3a_erf_4panel.py (2x2 grid: columns = edit,
rows = enhanced/dampened), but the term source is the curated GMT
pathway file instead of the claude-annotated term column.

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

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6/pairwise_pathway_ORA"
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

# ---- load the four groups: each edit x each effect ----
combined_df = pd.read_csv(combined_f)

def get_gene_set(edit, effect):
    mask = (combined_df["edit"] == edit) & (combined_df["category"].str.startswith(effect))
    return set(combined_df.loc[mask, "gene"])

groups = {}
for edit in [EDIT_A, EDIT_B]:
    for effect in ["enhanced", "dampened"]:
        label = f"{edit} ({effect})"
        groups[label] = get_gene_set(edit, effect)

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

out_csv = f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_pathway_ORA.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print("\nTop hits (padj < 0.05):")
print(results_df[results_df["padj"] < 0.05]
      [["group", "pathway", "k_overlap", "K_pathway_size", "odds_ratio", "padj"]]
      .to_string(index=False))

# =============================================================================
# 2x2 grid: columns = edit (Arid3a, Erf), rows = enhanced (top) / dampened (bottom)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

panel_layout = [
    [(EDIT_A, "enhanced"), (EDIT_B, "enhanced")],
    [(EDIT_A, "dampened"), (EDIT_B, "dampened")],
]

for row_idx, row in enumerate(panel_layout):
    for col_idx, (edit, effect) in enumerate(row):
        ax = axes[row_idx][col_idx]
        label = f"{edit} ({effect})"

        sub = results_df[results_df["group"] == label].nsmallest(TOP_N_PATHWAYS, "padj")
        sub = sub.sort_values("neg_log10_padj")

        ax.barh(sub["pathway"], sub["neg_log10_padj"], color=EDIT_COLOR[edit])
        for i, (k, K) in enumerate(zip(sub["k_overlap"], sub["K_pathway_size"])):
            ax.text(sub["neg_log10_padj"].iloc[i] + 0.05, i, f"{k}/{K}", va="center", fontsize=7)

        ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("-log10(padj)")
        ax.set_title(f"{label}\ntop {TOP_N_PATHWAYS} pathways by padj", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
fig_out = f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_pathway_ORA_barplot.png"
plt.savefig(f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_pathway_ORA_barplot.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_pathway_ORA_barplot.svg", dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_out}")

print("\nDone.")
