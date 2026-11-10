"""
Arid3aT132A and ErfT529A, both enhanced AND dampened gene sets (combined
IL6 U IL10 classification), tested for combined_claude_term over-
representation -- same ORA structure as pathway_ora_arid3a_vs_erf.py
(one-sided Fisher's exact, BH-corrected), now shown as a 2x2 grid:
columns = edit (Arid3a, Erf), rows = enhanced (top) / dampened (bottom).

Background: FIXED = the union of ALL FOUR edits' own orange/yellow gene
sets (both cytokines), intersected with genes that have a
combined_claude_term annotation -- a constant, independent of which two
edits are being compared, so a given edit's stats never shift just
because the comparison partner changed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ---- data sources ----
combined_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6b/ntc_cytokine_edit_effect_combined_union.csv"
terms_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/enhance_dampen_analysis/enhance_dampen_combined_union_genes_annotated.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6c"
os.makedirs(out_dir, exist_ok=True)

EDIT_A, EDIT_B = "Arid3aT132A", "ErfT529A"
ALL_EDITS = ["ErfT529A", "TfebS392P", "Arid3aT132A", "Etv3S29P"]
STIMS = ["IL6", "IL10"]

TOP_N_TERMS = 15

# ---- load gene-category assignments and term annotations ----
combined_df = pd.read_csv(combined_f)

terms_df = pd.read_csv(terms_f)
gene_to_term = dict(zip(terms_df["gene_symbol"], terms_df["combined_claude_term"]))
annotated_genes = set(gene_to_term.keys())

term_to_genes = {}
for gene, term in gene_to_term.items():
    term_to_genes.setdefault(term, set()).add(gene)
print(f"combined_claude_term categories: {len(term_to_genes)}")


def get_gene_set(edit, effect_prefix):
    mask = (combined_df["edit"] == edit) & (combined_df["category"].str.startswith(effect_prefix))
    return set(combined_df.loc[mask, "gene"])


# ---- FIXED background: the exact gene universe that was sent to Claude
# for annotation (the deduplicated gene_symbol list from
# enhance_dampen_combined_union_genes_annotated.csv) -- this file was
# itself built from the union of ALL FOUR edits' orange/yellow sets, both
# cytokines (see enhance_dampen_NTC_genes.py), so using it directly here
# means the background is unambiguously "the population these enhanced/
# dampened subsets and their term annotations were actually drawn from,"
# rather than relying on a separately-recomputed set that happens to
# coincide with it. A constant, independent of which two edits are being
# compared, so a given edit's stats never shift just because the
# comparison partner changed.
background = annotated_genes
N = len(background)
print(f"Fixed background: {N} genes (the annotated IL6 U IL10 orange/yellow "
      f"universe across all 4 edits)")

# restrict term_to_genes to the shared background too (a term's "size" for
# this test is only the portion of it inside the actual background)
term_to_genes_bg = {t: g & background for t, g in term_to_genes.items()}
term_to_genes_bg = {t: g for t, g in term_to_genes_bg.items() if len(g) >= 2}

# ---- the four groups being compared: each edit x each effect ----
groups = {}
for edit in [EDIT_A, EDIT_B]:
    for effect in ["enhanced", "dampened"]:
        label = f"{edit} ({effect})"
        groups[label] = get_gene_set(edit, effect) & background
        print(f"{label}: {len(groups[label])} genes in background")

# =============================================================================
# ORA: one-sided Fisher's exact per (group, term), SHARED background
# =============================================================================
all_results = []
for label, group_genes in groups.items():
    n_group = len(group_genes)

    for term, term_genes in term_to_genes_bg.items():
        K = len(term_genes)
        a = len(group_genes & term_genes)
        if a == 0:
            continue
        b = n_group - a
        c = K - a
        d = N - n_group - c

        odds_ratio, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        all_results.append({
            "group": label, "term": term,
            "k_overlap": a, "n_group_genes": n_group, "K_term_size": K, "N_background": N,
            "odds_ratio": odds_ratio, "p_value": p,
        })

results_df = pd.DataFrame(all_results)
results_df["padj"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
results_df["neg_log10_padj"] = -np.log10(results_df["padj"].clip(lower=1e-300))
results_df = results_df.sort_values(["group", "padj"])

out_csv = f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_combined_claude_term_ORA.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print("\nTop hits (padj < 0.05):")
print(results_df[results_df["padj"] < 0.05]
      [["group", "term", "k_overlap", "K_term_size", "odds_ratio", "padj"]]
      .to_string(index=False))

# =============================================================================
# 2x2 grid: columns = edit (Arid3a, Erf), rows = enhanced (top) / dampened (bottom)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
EDIT_COLOR = {EDIT_A: "#6A3D9A", EDIT_B: "#FF7F00"}

panel_layout = [
    [(EDIT_A, "enhanced"), (EDIT_B, "enhanced")],
    [(EDIT_A, "dampened"), (EDIT_B, "dampened")],
]

for row_idx, row in enumerate(panel_layout):
    for col_idx, (edit, effect) in enumerate(row):
        ax = axes[row_idx][col_idx]
        label = f"{edit} ({effect})"

        sub = results_df[results_df["group"] == label].nsmallest(TOP_N_TERMS, "padj")
        sub = sub.sort_values("neg_log10_padj")

        ax.barh(sub["term"], sub["neg_log10_padj"], color=EDIT_COLOR[edit])
        for i, (k, K) in enumerate(zip(sub["k_overlap"], sub["K_term_size"])):
            ax.text(sub["neg_log10_padj"].iloc[i] + 0.05, i, f"{k}/{K}", va="center", fontsize=7)

        ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("-log10(padj)")
        ax.set_title(f"{label}\ntop {TOP_N_TERMS} terms by padj", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
fig_out = f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_combined_claude_term_ORA_barplot.png"
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.savefig(f"{out_dir}/{EDIT_A}_vs_{EDIT_B}_enhanced_dampened_combined_claude_term_ORA_barplot.svg", dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_out}")

print("\nDone.")