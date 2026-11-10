"""
Builds the "expressed TFs with DEsites" background list (~100 TFs,
per conversation) -- the intersection of the full expressed-TF list
(used for the broad 1004-TF background run) and the genes present in
the DEsite cluster file (Fig1's phosphosite clustering results).

Output: a plain-text, one-gene-per-line file in the SAME format as
log2_thresh_2.5_expressed_tfs.txt, so it can be used as a drop-in
TF_LIST for a second run of run_fimo_array.sh (different TF_LIST path,
different OUT_BASE, --array range recalculated for the smaller count).
"""

import os
import pandas as pd

# ---- data sources ----
expressed_tfs_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/TF_enrichment/Results/log2_thresh_2.5_expressed_tfs.txt"
desites_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig1/des_heatmap/out_files/pairwise_DEsites_9_clusters_ordered.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results"
os.makedirs(out_dir, exist_ok=True)
out_f = f"{out_dir}/expressed_DEsite_TFs.txt"

for f, label in [(expressed_tfs_f, "expressed TFs list"), (desites_f, "DEsites cluster CSV")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label}: {f}")

# ---- load the expressed TF list ----
with open(expressed_tfs_f) as f:
    expressed_tfs = set(line.strip() for line in f if line.strip())
print(f"Expressed TFs (full list): {len(expressed_tfs)}")

# ---- load the DEsites cluster file -- print columns since the exact
# gene-column name isn't confirmed ----
desites_df = pd.read_csv(desites_f)
print(f"\nDEsites file columns: {list(desites_df.columns)}")
print(f"DEsites file shape: {desites_df.shape}")
print(f"First few rows:\n{desites_df.head()}")

# confirmed column name: "geneSymbol" -- NOTE: the "id" column here
# starts with a UniProt accession (e.g. "P83741-3_S2282s_..."), NOT the
# gene symbol, so it would be the WRONG thing to parse even as a
# fallback -- geneSymbol is the only correct source for this file.
if "geneSymbol" not in desites_df.columns:
    raise SystemExit(f"Expected column 'geneSymbol' not found. Actual columns: {list(desites_df.columns)}")

desite_genes = set(desites_df["geneSymbol"].dropna().unique())

print(f"\nUnique genes in DEsites file: {len(desite_genes)}")

# ---- intersect ----
expressed_desite_tfs = sorted(expressed_tfs & desite_genes)
print(f"\nExpressed TFs WITH a DEsite: {len(expressed_desite_tfs)}")
print(expressed_desite_tfs)

with open(out_f, "w") as f:
    for tf in expressed_desite_tfs:
        f.write(tf + "\n")

print(f"\nSaved: {out_f}")
print(f"SLURM array range for a second run_fimo_array.sh pass: --array=0-{(len(expressed_desite_tfs) - 1) // 25}")
print("\nDone.")