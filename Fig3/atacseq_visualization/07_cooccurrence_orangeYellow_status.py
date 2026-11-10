"""
Fig3 ATAC-seq visualization, step 7: for every (gene, other_family)
combination in the STAT3/TF co-occurrence results
(05_STAT_family_motif_proximity_check.py's saved
STAT_family_motif_proximity_50bp.csv), check whether that gene is
classified orange/yellow (unioned across IL6/IL10, same convention
used throughout this project) for Stat3Y705H AND for the other edit
(Tfeb/Erf/Etv3/Arid3a).

NOTE on naming: the co-occurrence results use motif-level gene symbols
(STAT3, TFEB, ERF, ETV3, ARID3A), but the orange/yellow classification
files are keyed by the specific EDIT name (Stat3Y705H, TfebS392P,
ErfT529A, Etv3S29P, Arid3aT132A) -- TF_TO_EDIT_NAME maps between them.
"""

import os
import pandas as pd

# ---- data sources ----
proximity_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/STAT_family_motif_proximity_50bp.csv"
interaction_dir_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/edit_effect_v2/interaction_model"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

STIMS = ["IL6", "IL10"]
BASE_CATEGORIES = ["orange", "yellow"]

# motif-level gene symbol -> actual edit name used in the classification files
TF_TO_EDIT_NAME = {
    "STAT3": "Stat3Y705H",
    "TFEB": "TfebS392P",
    "ERF": "ErfT529A",
    "ETV3": "Etv3S29P",
    "ARID3A": "Arid3aT132A",
}


def load_orange_yellow_union(edit_name):
    """Union of orange/yellow genes across IL6 and IL10 for this edit
    -- same convention as oy_venn3_stat3_erf_arid3a.py and others."""
    tf_union = set()
    for stim in STIMS:
        f = f"{interaction_dir_base}/{edit_name}/{edit_name}_{stim}_fourCategory_classification_data.csv"
        if not os.path.exists(f):
            print(f"  WARNING: missing {f}, {edit_name}/{stim} contributes nothing to the union")
            continue
        df = pd.read_csv(f, index_col=0)
        stim_genes = set(df.index[df["category"].isin(BASE_CATEGORIES)])
        tf_union |= stim_genes
    return tf_union


# ---- load the co-occurrence results ----
proximity_df = pd.read_csv(proximity_f)
print(f"Loaded {len(proximity_df)} co-occurring pairs")

# derive each row's actual TF gene symbol from other_motif_id (e.g.
# "TFEB.H14CORE.0.SM.B" -> "TFEB"), and the STAT side too for consistency
proximity_df["other_tf"] = proximity_df["other_motif_id"].str.split(".").str[0].str.upper()
proximity_df["stat_tf"] = proximity_df["stat_motif_id"].str.split(".").str[0].str.upper()

# ---- reduce to unique (gene, other_family) combinations -- orange/yellow
# status is a per-gene, per-edit property, independent of which
# SPECIFIC motif pair/position a gene had multiple co-occurrence rows for ----
unique_combos = proximity_df[["gene", "other_family", "stat_tf", "other_tf"]].drop_duplicates().reset_index(drop=True)
print(f"Unique (gene, other_family) combinations: {len(unique_combos)}")

# ---- load orange/yellow sets for every TF actually present, once each ----
all_tfs_needed = set(unique_combos["stat_tf"]) | set(unique_combos["other_tf"])
print(f"\nLoading orange/yellow classification for: {sorted(all_tfs_needed)}")

oy_sets = {}
for tf in all_tfs_needed:
    edit_name = TF_TO_EDIT_NAME.get(tf)
    if edit_name is None:
        print(f"  WARNING: no edit-name mapping for TF symbol '{tf}' -- add it to TF_TO_EDIT_NAME")
        oy_sets[tf] = set()
        continue
    oy_sets[tf] = load_orange_yellow_union(edit_name)
    print(f"  {tf} ({edit_name}): orange/yellow (IL6 U IL10) = {len(oy_sets[tf])}")

# ---- build the final per-gene/per-family status table ----
unique_combos["stat3_orange_yellow"] = unique_combos.apply(
    lambda row: row["gene"] in oy_sets.get(row["stat_tf"], set()), axis=1
)
unique_combos["other_tf_orange_yellow"] = unique_combos.apply(
    lambda row: row["gene"] in oy_sets.get(row["other_tf"], set()), axis=1
)

result_df = unique_combos.rename(columns={"other_family": "family"})[
    ["gene", "family", "stat_tf", "stat3_orange_yellow", "other_tf", "other_tf_orange_yellow"]
]
result_df = result_df.sort_values(["family", "gene"]).reset_index(drop=True)

print(f"\n{result_df.to_string(index=False)}")

print("\nSummary -- both orange/yellow (Stat3 AND the other TF):")
both_oy = result_df[result_df["stat3_orange_yellow"] & result_df["other_tf_orange_yellow"]]
print(f"  {len(both_oy)} / {len(result_df)} gene-family combinations have BOTH edits orange/yellow")
print(both_oy.groupby("family").size().to_string())

out_csv = f"{out_dir}/cooccurrence_orangeYellow_status.csv"
result_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print("\nDone.")
