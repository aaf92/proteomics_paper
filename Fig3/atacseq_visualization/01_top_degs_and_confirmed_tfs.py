"""
Fig3 ATAC-seq visualization, step 1:
  (a) Top 10 differentially expressed genes from IL10 (1 ng/mL), 2hr vs
      PBS -- DEG criteria: adjusted p < 0.05 AND 2**abs(log2FoldChange)
      > 1.5 (same convention as RNAseq_volcano_plots.py), ranked by
      padj ascending among genes passing that filter.
  (b) TF list from the motif scatter plot's merged dataframe --
      significant in BOTH HOMER and TF-MoDISco (q < 0.05 each) AND
      carrying a differential phosphosite (in_deg == True) -- the
      "top-right quadrant, colored dots" set.
"""

import os
import numpy as np
import pandas as pd

# ---- data sources ----
il10_2hr_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/RNAseq/DESeq2_results/res_il10_1_2.csv"
merged_motif_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/sig_motif_res/out_files/motif_based_scatterPlot_data.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

padj_thresh = 0.05
fc_thresh = 1.5  # linear fold change, applied as 2**abs(log2FoldChange) > fc_thresh


def find_padj_col(df):
    for candidate in ["padj", "adjusted_padj"]:
        if candidate in df.columns:
            return candidate
    raise SystemExit(f"Could not find a recognized padj column (checked 'padj', 'adjusted_padj'). "
                      f"Actual columns: {list(df.columns)}")


# =============================================================================
# (a) Top 10 DEGs, IL10 1ng/mL 2hr vs PBS
# =============================================================================
deg_df = pd.read_csv(il10_2hr_f, index_col=0)
padj_col = find_padj_col(deg_df)

deg_df = deg_df.dropna(subset=[padj_col, "log2FoldChange"]).copy()
deg_df["fold_change"] = 2 ** deg_df["log2FoldChange"].abs()
deg_df["is_deg"] = (deg_df[padj_col] < padj_thresh) & (deg_df["fold_change"] > fc_thresh)

sig_degs = deg_df[deg_df["is_deg"]].sort_values(padj_col)
print(f"IL10 2hr: {len(sig_degs)} genes pass padj<{padj_thresh} AND FC>{fc_thresh}x "
      f"(of {len(deg_df)} total tested)")

top10_degs = sig_degs.head(10).copy()
top10_degs = top10_degs.reset_index().rename(columns={top10_degs.index.name or "index": "gene"})
top10_degs = top10_degs[["gene", "log2FoldChange", padj_col, "fold_change"]]

print("\nTop 10 DEGs (IL10, 2hr, ranked by padj):")
print(top10_degs.to_string(index=False))

top10_csv = f"{out_dir}/top10_IL10_2hr_DEGs.csv"
top10_degs.to_csv(top10_csv, index=False)
print(f"\nSaved: {top10_csv}")

# =============================================================================
# (b) TF list: significant in BOTH HOMER and TF-MoDISco, AND has a
# differential phosphosite (in_deg) -- the top-right quadrant colored
# dots from the motif scatter plot
# =============================================================================
merged = pd.read_csv(merged_motif_f)

required_cols_base = ["tf_symbol", "q_value_homer", "q_value_modisco"]
missing = [c for c in required_cols_base if c not in merged.columns]
if missing:
    raise SystemExit(f"Merged motif file is missing expected column(s): {missing}. "
                      f"Actual columns: {list(merged.columns)}")

# column is named "in_des" in the actual saved file (differential
# phosphosite flag), not "in_deg" -- check both for robustness in case
# this changes again later
if "in_des" in merged.columns:
    deg_flag_col = "in_des"
elif "in_deg" in merged.columns:
    deg_flag_col = "in_deg"
else:
    raise SystemExit(f"Merged motif file is missing a differential-phosphosite flag column "
                      f"(checked 'in_des', 'in_deg'). Actual columns: {list(merged.columns)}")

q_thresh = 0.05
merged["sig_in_both"] = (merged["q_value_homer"] < q_thresh) & (merged["q_value_modisco"] < q_thresh)

confirmed_tfs_df = merged[merged["sig_in_both"] & merged[deg_flag_col]].copy()
confirmed_tfs = confirmed_tfs_df["tf_symbol"].tolist()

print(f"\nTFs significant in both HOMER and TF-MoDISco (q<{q_thresh} each) "
      f"AND carrying a differential phosphosite: {len(confirmed_tfs)}")
print(confirmed_tfs)

confirmed_csv = f"{out_dir}/confirmed_tfs_bothMethods_withPhosphosite.csv"
confirmed_tfs_df.to_csv(confirmed_csv, index=False)
print(f"\nSaved: {confirmed_csv}")

print("\nDone.")