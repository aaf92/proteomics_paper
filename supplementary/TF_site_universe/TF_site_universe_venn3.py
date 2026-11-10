"""
Supplementary: 3-way Venn diagram of phosphosite universes --
  A. IL6-differential sites  -- significant in AT LEAST ONE of IL6
     20/45/90min vs 0min.
  B. IL10-differential sites -- same, for IL10.
  C. TF sites -- ALL phosphosites (regardless of significance) whose
     gene is a transcription factor (Mus_musculus_TF.txt).

Requires matplotlib-venn: pip install matplotlib-venn --break-system-packages

Multi-site rows are split into individual sites first (a row's id like
"Srrm2_Q8BTI8_S1438sS1439s_2_2_1438_1439" becomes two rows,
"Srrm2_Q8BTI8_S1438s_2_1438" and "Srrm2_Q8BTI8_S1439s_2_1439") -- each
split site inherits the SAME significance calls as its parent row,
since there's no way to separately attribute significance to one
specific co-occurring site over the other.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# ---- data sources ----
phospho_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/phosphoproteomics/stat_results_IL6IL10phosphopairwiseBMDMheatmap.csv"
tf_list_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/tf_motif_info/Mus_musculus_TF.txt"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/TF_site_universe"
os.makedirs(out_dir, exist_ok=True)



# ---- load the mouse TF list ----
tf_df = pd.read_csv(tf_list_f, sep="\t")
tf_genes = set(tf_df["Symbol"].str.upper())
print(f"TF gene symbols loaded: {len(tf_genes)}")

# ---- load phosphoproteomics data -- id, geneSymbol, and EVERY column
# starting with "significant." (not just the vs-0min comparisons --
# also picks up cross-cytokine comparisons like
# significant.IL6_20min_over_IL10_20min) ----
phospho_df_raw = pd.read_csv(phospho_f)

sig_cols = [c for c in phospho_df_raw.columns if c.startswith("significant.")]
if not sig_cols:
    raise SystemExit("No columns starting with 'significant.' found -- check the source file's column naming.")
print(f"Found {len(sig_cols)} 'significant.' columns:")
for c in sig_cols:
    print(f"  {c}")

phospho_df = phospho_df_raw[["id", "geneSymbol"] + sig_cols].copy()
print(f"\nPhosphoproteomics rows loaded: {len(phospho_df)}")

# ---- diagnostic: RAW (unsplit) ids, restricted to ONLY the vs-0min
# columns for IL6/IL10 (no cross-cytokine comparisons) -- unique id
# count before any site-splitting ----
vs0min_cols = [c for c in sig_cols if "_over_0min" in c and ("IL6" in c or "IL10" in c)]
print(f"\n[diagnostic] vs-0min-only columns (IL6/IL10, no cross-cytokine): {vs0min_cols}")

vs0min_mask = phospho_df[vs0min_cols].any(axis=1)
vs0min_unique_ids = set(phospho_df.loc[vs0min_mask, "id"])
print(f"[diagnostic] Unique RAW ids (no site-splitting), significant in >=1 of "
      f"IL6/IL10 20/45/90min_over_0min: {len(vs0min_unique_ids)}")

# ---- same diagnostic, but using ALL significant. columns (including
# cross-cytokine comparisons like IL6_20min_over_IL10_20min) ----
all_sig_mask = phospho_df[sig_cols].any(axis=1)
all_sig_unique_ids = set(phospho_df.loc[all_sig_mask, "id"])
print(f"[diagnostic] Unique RAW ids (no site-splitting), significant in >=1 of "
      f"ALL {len(sig_cols)} significant. columns (including cross-cytokine comparisons): "
      f"{len(all_sig_unique_ids)}")


# ---- split multi-site rows into individual sites ----
def split_id(id_str):
    """A combined multi-site id like 'Srrm2_Q8BTI8_S1438sS1439s_2_2_1438_1439'
    becomes ['Srrm2_Q8BTI8_S1438s_2_1438', 'Srrm2_Q8BTI8_S1439s_2_1439'].
    A single-site id (e.g. 'Ahnak_E9Q616_S42s_1_1_42_42') becomes a
    length-1 list, ['Ahnak_E9Q616_S42s_1_42'].

    Position is extracted DIRECTLY from each site token itself (the
    digits between the residue letter and the lowercase suffix), not
    from the trailing numeric fields -- those trailing fields vary in
    count (e.g. a single site's position gets duplicated as a
    collapsed start=end "range": '..._42_42') and aren't a reliable
    match against the number of site tokens, so they're ignored here
    entirely for position purposes."""
    fields = id_str.split("_")
    if len(fields) < 4:
        return None  # unexpected format -- flagged, not silently dropped
    gene, uniprot, combined_token, count_field = fields[0], fields[1], fields[2], fields[3]
    site_tokens = re.findall(r"[A-Z]\d+[a-z]", combined_token)
    positions = re.findall(r"[A-Z](\d+)[a-z]", combined_token)
    if not site_tokens:
        return None
    return [f"{gene}_{uniprot}_{tok}_{count_field}_{pos}" for tok, pos in zip(site_tokens, positions)]


expanded_rows = []
n_malformed = 0
for _, row in phospho_df.iterrows():
    split_ids = split_id(row["id"])
    if split_ids is None:
        n_malformed += 1
        continue
    for site_id in split_ids:
        new_row = row.to_dict()
        new_row["id"] = site_id  # each split site inherits the SAME significance calls
        expanded_rows.append(new_row)

if n_malformed:
    print(f"WARNING: {n_malformed} rows had an unexpected id format and were skipped")

site_df = pd.DataFrame(expanded_rows)
print(f"Expanded to {len(site_df)} individual-site rows")

dup_mask = site_df["id"].duplicated(keep=False)
if dup_mask.any():
    n_dup = dup_mask.sum()
    print(f"NOTE: {n_dup} duplicate site ids after splitting (same site reported in multiple "
          f"source rows) -- kept as-is, sets below are still correct since set() dedupes them")

# ---- build the three sets ----
il6_cols = [c for c in sig_cols if "IL6" in c and "_over_0min" in c]
il10_cols = [c for c in sig_cols if "IL10" in c and "_over_0min" in c]
print(f"\nColumns treated as IL6-related ({len(il6_cols)}): {il6_cols}")
print(f"Columns treated as IL10-related ({len(il10_cols)}): {il10_cols}")
overlap_cols = set(il6_cols) & set(il10_cols)
if overlap_cols:
    print(f"NOTE: {len(overlap_cols)} column(s) mention BOTH IL6 and IL10 "
          f"(e.g. a cross-cytokine comparison) and contribute to both sets: {sorted(overlap_cols)}")

il6_diff_mask = site_df[il6_cols].any(axis=1)
il10_diff_mask = site_df[il10_cols].any(axis=1)
tf_mask = site_df["geneSymbol"].str.upper().isin(tf_genes)

# ---- per-timepoint significant site counts, before taking the union
# across timepoints within each cytokine ----
print("\nSignificant sites per individual timepoint-vs-0min comparison:")
for stim, cols in [("IL6", il6_cols), ("IL10", il10_cols)]:
    for c in cols:
        n_sig = site_df[c].sum()
        print(f"  {c}: {n_sig}")

il6_diff_sites = set(site_df.loc[il6_diff_mask, "id"])
il10_diff_sites = set(site_df.loc[il10_diff_mask, "id"])
tf_sites = set(site_df.loc[tf_mask, "id"])

print(f"\nIL6-differential sites (>=1 of 20/45/90min): {len(il6_diff_sites)}")
print(f"IL10-differential sites (>=1 of 20/45/90min): {len(il10_diff_sites)}")

# ---- unique union across BOTH cytokines -- a site counts once even if
# differential in both IL6 and IL10 ----
il6_or_il10_sites = il6_diff_sites | il10_diff_sites
print(f"IL6 U IL10 differential sites (unique union, either cytokine): {len(il6_or_il10_sites)}")

print(f"TF sites (all, regardless of significance): {len(tf_sites)}")

# =============================================================================
# 3-way Venn
# =============================================================================
LABEL_A, LABEL_B, LABEL_C = "IL6-differential sites", "IL10-differential sites", "TF sites"

# Okabe-Ito colorblind-safe palette (Wong 2011), same convention as
# oy_venn3_stat3_erf_arid3a.py
VENN_COLORS = ("#E69F00", "#56B4E9", "#009E73")  # orange, sky blue, green

fig, ax = plt.subplots(figsize=(7, 7))
v = venn3([il6_diff_sites, il10_diff_sites, tf_sites],
          set_labels=(LABEL_A, LABEL_B, LABEL_C), set_colors=VENN_COLORS, alpha=0.7, ax=ax)

ax.set_title("Phosphosite universe: IL6-differential vs IL10-differential vs TF sites\n"
             "site-level (post multi-site-splitting), raw overlap")
plt.tight_layout()

png_f = f"{out_dir}/TF_site_universe_venn3.png"
svg_f = f"{out_dir}/TF_site_universe_venn3.svg"
plt.savefig(png_f, dpi=300, bbox_inches="tight")
plt.savefig(svg_f, bbox_inches="tight")
plt.show()
print(f"\nSaved: {png_f}")

# print region counts for reference (venn3 region codes: '100'=A only,
# '010'=B only, '001'=C only, '110'=A&B not C, etc, '111'=all three)
region_labels = {
    "100": f"{LABEL_A} only", "010": f"{LABEL_B} only", "001": f"{LABEL_C} only",
    "110": f"{LABEL_A} & {LABEL_B} only", "101": f"{LABEL_A} & {LABEL_C} only",
    "011": f"{LABEL_B} & {LABEL_C} only", "111": "all three",
}
print(f"\nRegion counts:")
for code, label in region_labels.items():
    label_obj = v.get_label_by_id(code)
    text = label_obj.get_text() if label_obj is not None else "0"
    print(f"  {label}: {text}")

# ---- save the underlying per-site set membership for reference ----
membership_df = pd.DataFrame({
    "id": sorted(il6_diff_sites | il10_diff_sites | tf_sites),
})
membership_df["IL6_differential"] = membership_df["id"].isin(il6_diff_sites)
membership_df["IL10_differential"] = membership_df["id"].isin(il10_diff_sites)
membership_df["TF_site"] = membership_df["id"].isin(tf_sites)
membership_csv = f"{out_dir}/TF_site_universe_membership.csv"
membership_df.to_csv(membership_csv, index=False)
print(f"\nSaved: {membership_csv}")

print("\nDone.")