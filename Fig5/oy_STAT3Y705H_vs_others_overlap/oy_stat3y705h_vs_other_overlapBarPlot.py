"""
Fig5: Stat3Y705H vs each of the other 4 edits, raw orange/yellow overlap
(IL6 U IL10 -- each edit's set is the union of its IL6 and IL10
classifications). Hypergeometric test against a fixed background,
N=4110 (verified count: union of NTC's own IL6-vs-PBS and IL10-vs-PBS
DESeq2 DEGs, adjusted_padj<0.05, from 03_stim_delta_analysis_allTFs.ipynb).

Annotated with raw shared (intersection) count and p-value only -- no
fold-enrichment.

Overlapping-bar style: Stat3's bar (unique + shared) drawn OPEN and
HATCHED, spanning [0, n_stat3_total]. The other edit's bar drawn as a
SOLID light-colored fill, spanning [n_unique_stat3, n_unique_stat3 +
n_edit_total] -- starting right at the beginning of the shared region
and extending past it into the edit's own unique region.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import hypergeom

interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig5/orange_yellow_stat3y705h_vs_edits_overlap"
os.makedirs(out_dir, exist_ok=True)

STAT3 = "Stat3Y705H"
OTHER_EDITS = ["ErfT529A", "TfebS392P", "Arid3aT132A", "Etv3S29P", "Hivep3S1949P", "ErfT3A"]

STIMS = ["IL6", "IL10"]

BASE_CATEGORIES = ["orange", "yellow"]
CAT = "orange_yellow"

EDIT_FILL_COLOR = {
    "ErfT529A": "#A6CEE3", "TfebS392P": "#FB9A99",
    "Arid3aT132A": "#B2DF8A", "Etv3S29P": "#FDBF6F",
    "Hivep3S1949P": "#CAB2D6", "ErfT3A": "#FFFF99",
}

bar_height = 0.5
IL6_IL10_DEG_UNIVERSE_N = 4110  # verified count, see docstring

# ---- load orange/yellow per TF, UNIONED across IL6 and IL10 ----
tf_category_genes = {}

for tf in [STAT3] + OTHER_EDITS:
    tf_union = set()
    for stim in STIMS:
        class_f = f"{interaction_dir_base}/{tf}/{tf}_{stim}_fourCategory_classification_data.csv"
        if not os.path.exists(class_f):
            print(f"WARNING: missing {class_f}, {tf}/{stim} contributes nothing to the union")
            continue
        class_df = pd.read_csv(class_f, index_col=0)
        stim_genes = set(class_df.index[class_df["category"].isin(BASE_CATEGORIES)])
        print(f"  {tf} / {stim}: {CAT} = {len(stim_genes)}")
        tf_union |= stim_genes
    tf_category_genes[tf] = tf_union
    print(f"{tf}: {CAT} (IL6 U IL10) = {len(tf_union)}")


def hypergeom_test_fixed_N(set_a, set_b, N):
    """N is a fixed constant (the verified ~4110-gene IL6/IL10 NTC DEG
    universe), not an actual gene set we can intersect against -- so K,
    n, and k are each set's/overlap's RAW size, with no restriction step."""
    k = len(set_a & set_b)
    K = len(set_a)
    n = len(set_b)

    if N == 0 or K == 0 or n == 0:
        return k, N, K, n, float("nan")

    pval = hypergeom.sf(k - 1, N, K, n)  # P(X >= k)
    return k, N, K, n, pval


def draw_overlap_bar(ax, i, n_unique_a, n_shared, n_b_total, color):
    n_a_total = n_unique_a + n_shared
    ax.barh(i, n_b_total, left=n_unique_a, height=bar_height,
            color=color, edgecolor="black", linewidth=0.8, zorder=1)
    ax.barh(i, n_a_total, left=0, height=bar_height,
            facecolor="none", hatch="///", edgecolor="black", linewidth=1.2, zorder=2)
    return max(n_a_total, n_unique_a + n_b_total)


def make_legend(ax, other_edits_present):
    stat3_legend = mpatches.Patch(facecolor="none", edgecolor="black", hatch="///", label=STAT3)
    edit_legend_handles = [
        mpatches.Patch(facecolor=EDIT_FILL_COLOR[e], edgecolor="black", label=e)
        for e in other_edits_present
    ]
    ax.legend(handles=[stat3_legend] + edit_legend_handles, loc="lower right", fontsize=8)


# =========================================================================
# Stat3 vs each edit's raw orange/yellow set (IL6 U IL10)
# =========================================================================
rows = []
for edit in OTHER_EDITS:
    set_stat3, set_edit = tf_category_genes[STAT3], tf_category_genes[edit]

    k, N, K, n, pval = hypergeom_test_fixed_N(set_stat3, set_edit, IL6_IL10_DEG_UNIVERSE_N)

    print(f"{STAT3} vs {edit}: N={N}, K(Stat3)={K}, n(edit)={n}, k(shared)={k}, pval={pval:.3g}")

    rows.append({
        "edit": edit, "N_background": N, "K_stat3": K, "n_edit": n, "k_shared": k,
        "n_unique_stat3": K - k, "n_unique_edit": n - k,
        "pvalue": pval,
    })

df = pd.DataFrame(rows).sort_values("k_shared", ascending=True).reset_index(drop=True)
out_csv = f"{out_dir}/fig5_orange_yellow_stat3_vs_other_overlapBarPlot.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

fig, ax = plt.subplots(figsize=(11, max(4, 0.8 * len(df))))
for i, row in df.iterrows():
    total_extent = draw_overlap_bar(ax, i, row["n_unique_stat3"], row["k_shared"], row["n_edit"],
                                     EDIT_FILL_COLOR[row["edit"]])
    ax.text(total_extent + 1, i,
            f"shared={row['k_shared']}, p={row['pvalue']:.2g}",
            va="center", fontsize=8)

ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"{STAT3} vs {row['edit']}" for _, row in df.iterrows()], fontsize=9)
ax.set_xlabel("number of genes")
ax.set_title(f"{CAT} genes (IL6 U IL10): {STAT3} vs each other edit\n"
             f"raw overlap, hypergeometric test vs N={IL6_IL10_DEG_UNIVERSE_N} "
             f"(IL6/IL10 NTC DEG universe) -- "
             f"{STAT3} = open/hatched, other edit = filled\n"
             f"(sorted by shared gene count, low to high, top-to-bottom)")
make_legend(ax, OTHER_EDITS)
plt.tight_layout()

png_f = f"{out_dir}/fig5_orange_yellow_stat3_vs_other_overlapBarPlot.png"
svg_f = f"{out_dir}/fig5_orange_yellow_stat3_vs_other_overlapBarPlot.svg"
plt.savefig(png_f, dpi=300, bbox_inches="tight")
plt.savefig(svg_f, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {png_f}")

print("\nDone.")