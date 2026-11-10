"""
Fig6: single 3-way Venn diagram, Stat3Y705H vs ErfT529A vs Arid3aT132A,
raw orange_yellow gene overlap (no pairwise significance filter). Each
edit's set is the UNION of its IL6 and IL10 classifications (not IL10
alone, unlike the earlier venn script this is subset from).

Requires matplotlib-venn: pip install matplotlib-venn --break-system-packages
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig6a"
os.makedirs(out_dir, exist_ok=True)

TF_A, TF_B, TF_C = "Stat3Y705H", "ErfT529A", "Arid3aT132A"
STIMS = ["IL6", "IL10"]

BASE_CATEGORIES = ["orange", "yellow"]
CAT = "orange_yellow"

# ---- load orange/yellow per TF, UNIONED across IL6 and IL10 ----
tf_category_genes = {}

for tf in [TF_A, TF_B, TF_C]:
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

set_a, set_b, set_c = tf_category_genes[TF_A], tf_category_genes[TF_B], tf_category_genes[TF_C]

# Okabe-Ito colorblind-safe palette (Wong 2011) -- commonly recommended
# for publication figures, stays distinguishable in grayscale print too
VENN_COLORS = ("#E69F00", "#56B4E9", "#009E73")  # orange, sky blue, green

fig, ax = plt.subplots(figsize=(7, 7))
v = venn3([set_a, set_b, set_c], set_labels=(TF_A, TF_B, TF_C), set_colors=VENN_COLORS, alpha=0.7, ax=ax)

ax.set_title(f"{CAT} genes (IL6 U IL10): {TF_A} vs {TF_B} vs {TF_C}\nraw overlap, no pairwise significance filter")
plt.tight_layout()

png_f = f"{out_dir}/fig6_orange_yellow_venn3_stat3_erf_arid3a.png"
svg_f = f"{out_dir}/fig6_orange_yellow_venn3_stat3_erf_arid3a.svg"
plt.savefig(png_f, dpi=300, bbox_inches="tight")
plt.savefig(svg_f, bbox_inches="tight")
plt.show()
print(f"Saved: {png_f}")

# print region counts for reference (venn3 region codes: '100'=A only,
# '010'=B only, '001'=C only, '110'=A&B not C, etc, '111'=all three)
region_labels = {
    "100": f"{TF_A} only", "010": f"{TF_B} only", "001": f"{TF_C} only",
    "110": f"{TF_A} & {TF_B} only", "101": f"{TF_A} & {TF_C} only",
    "011": f"{TF_B} & {TF_C} only", "111": "all three",
}
print(f"\nRegion counts for {TF_A} vs {TF_B} vs {TF_C}:")
for code, label in region_labels.items():
    label_obj = v.get_label_by_id(code)
    text = label_obj.get_text() if label_obj is not None else "0"
    print(f"  {label}: {text}")

print("\nDone.")