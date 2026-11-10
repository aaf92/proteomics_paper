"""
Fig3 ATAC-seq visualization, supplementary check: for every STAT motif
hit (8000_TSS_8000_stats_all/fimo.tsv), restricted to genes in the
common IL6/IL10 1&2hr DEG list, check whether any OTHER TF family's
motif hit (same gene, same 8000_TSS_8000 window) falls within +/-50bp.

"Within 50bp" = the gap between the two motif intervals is <=50bp
(0 if they overlap at all).

NOTE on significance filtering: an earlier version of this script
filtered on q-value < 0.05, which turned out to be the WRONG metric --
FIMO's q-value is BH-corrected across EVERY position tested in that
entire run (every base, both strands, every gene's full window), which
for a promoterome-wide scan is tens to hundreds of millions of tests.
That correction burden crushes q-values across the board regardless of
whether a given hit is biologically real, so q<0.05 throws out nearly
everything. Using the raw P-VALUE instead, matching FIMO's own
--thresh convention (the same thing the original FIMO run already
filtered on), is the standard/appropriate significance metric for this
kind of per-locus question.

NOTE: your FIMO_results listing has BOTH "8000_TSS_8000_arid_family"
and "8000_TSS_8000_arid3_family" as separate directories -- only
"arid_family" is used here, per instruction.

RESTRICTED TO SPECIFIC TFs, not whole families: STAT hits are
restricted to Stat3 specifically (not all STATs), and each other
family's hits are restricted to one named TF (Arid3a, Erf, Etv3, Tfeb)
via a prefix match on motif_id -- NOT the entire family's combined hit
set. This matters a lot: a whole-family scan (e.g. all ~25+ ETS-family
members in erf_ets_family) produces such a dense background of hits
that most "proximity" co-occurrences with another dense motif set are
plausibly just chance overlap from sheer positional density, not real
co-regulation -- restricting to one specific TF's own motif removes
most of that inflation.

Back to using the full 8000_TSS_8000 window (not restricted to a
narrower -1000/+500 TSS range) -- proximal hits within this full
window are intended to be cross-referenced against ATAC-seq peak
signal separately, rather than narrowed by position here.

OVERLAPPING/DUPLICATE HIT MERGING: after restricting to one specific
TF (via motif_id prefix), overlapping or adjacent hits for that SAME
TF are merged into one representative hit per site (merge_overlapping_hits).
This matters because HOCOMOCO sometimes carries multiple subtype PWMs
for the same gene (e.g. TFEB.H14CORE.0.SM.B and TFEB.H14CORE.1.P.C),
and/or FIMO can report two overlapping near-optimal matches of the
exact same PWM -- without merging, one real physical binding site
could get counted as 2-3 separate "co-occurrences" downstream, purely
as a reporting artifact rather than independent evidence.
"""

import os
import numpy as np
import pandas as pd

# ---- data sources ----
fimo_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results"
deg_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/il6_il10_1and2_hour_common_DEGs.txt"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

STAT_DIR = "8000_TSS_8000_stats_all"
STAT_MOTIF_PREFIX = "STAT3."  # restrict STAT hits to Stat3 specifically, not all STATs

OTHER_FAMILY_DIRS = {
    "arid_family": "8000_TSS_8000_arid_family",
    "erf_family": "8000_TSS_8000_erf_family",
    "erf_ets_family": "8000_TSS_8000_erf_ets_family",
    "mitf_tfe_family": "8000_TSS_8000_mitf_tfe_family",
}

# restrict each family's hits to ONE specific TF's motif, by gene-symbol
# PREFIX match on motif_id (HOCOMOCO ids always start with "GENE.") --
# NOT hardcoding a full motif_id string since subtype/quality suffix
# (e.g. ".0.P.B") isn't known in advance. NOTE: per the earlier
# conversation, Arid3a may not exist in H14CORE at all -- this will
# print a clear 0-hits warning rather than silently matching nothing
FAMILY_MOTIF_OF_INTEREST = {
    "arid_family": "ARID3A.",
    "erf_family": "ERF.",
    "erf_ets_family": "ETV3.",
    "mitf_tfe_family": "TFEB.",
}

PROXIMITY_THRESH = 50  # bp
PVAL_THRESH = 1e-4  # only keep FIMO hits with raw p-value < this (FIMO's own
                     # default --thresh), applied to BOTH the STAT hits and
                     # every other family's hits -- NOT q-value, see docstring

# ---- load the common DEG list -- plain .txt, no header, one gene per line ----
deg_genes = set(np.loadtxt(deg_f, dtype=str).tolist())
print(f"Loaded {len(deg_genes)} common DEGs from {deg_f}")
print(f"  Example genes: {list(deg_genes)[:10]}")


def load_fimo(dir_name):
    f = f"{fimo_base}/{dir_name}/fimo.tsv"
    if not os.path.exists(f):
        print(f"  WARNING: missing {f}")
        return None
    df = pd.read_csv(f, sep="\t", comment="#")
    df = df.dropna(subset=["sequence_name", "start", "stop", "p-value"])

    n_before = len(df)
    df = df[df["p-value"] < PVAL_THRESH]
    print(f"  {dir_name}: {n_before} hits -> {len(df)} after p-value < {PVAL_THRESH} filter")

    return df


def restrict_to_motif_prefix(df, prefix, label):
    """Prefix match on motif_id (e.g. 'ARID3A.' matches
    'ARID3A.H14CORE.0.P.B' regardless of subtype/quality suffix)."""
    if df is None or prefix is None:
        return df
    n_before = len(df)
    restricted = df[df["motif_id"].str.upper().str.startswith(prefix.upper())].copy()
    print(f"  {label}: {n_before} hits -> {len(restricted)} after restricting to motif_id "
          f"starting with '{prefix}'")
    if len(restricted) == 0:
        print(f"    WARNING: zero hits for prefix '{prefix}' -- check this motif actually "
              f"exists in this FIMO run (e.g. print df['motif_id'].unique())")
    return restricted


def merge_overlapping_hits(df, label):
    """Collapse overlapping/adjacent hits for the SAME TF (regardless
    of exact motif_id subtype, e.g. TFEB.H14CORE.0.SM.B vs
    TFEB.H14CORE.1.P.C) into one representative hit per merged region,
    keeping the lowest-p-value row. This is run AFTER
    restrict_to_motif_prefix, so every row here is already restricted
    to one TF -- multiple subtype PWMs or overlapping repeat-hits of
    the same PWM at essentially the same physical site would otherwise
    get counted as separate, independent co-occurrences downstream."""
    if df is None or len(df) == 0:
        return df

    merged_rows = []
    for gene, sub in df.groupby("sequence_name"):
        sub = sub.sort_values("start")
        current_stop = None
        current_group = []

        for _, row in sub.iterrows():
            if current_stop is None or row["start"] > current_stop:
                if current_group:
                    merged_rows.append(min(current_group, key=lambda r: r["p-value"]))
                current_group = [row]
                current_stop = row["stop"]
            else:
                current_group.append(row)
                current_stop = max(current_stop, row["stop"])

        if current_group:
            merged_rows.append(min(current_group, key=lambda r: r["p-value"]))

    merged_df = pd.DataFrame(merged_rows)
    print(f"  {label}: {len(df)} hits -> {len(merged_df)} after merging overlapping/adjacent "
          f"hits of the same TF into one representative hit per site")
    return merged_df


def restrict_to_degs(df, dir_name):
    if df is None:
        return None
    restricted = df[df["sequence_name"].isin(deg_genes)].copy()
    print(f"  {dir_name}: {len(df)} total hits -> {len(restricted)} in DEG genes "
          f"({restricted['sequence_name'].nunique()} unique genes)")
    return restricted


def interval_distance(start_a, stop_a, start_b, stop_b):
    """0 if intervals overlap, else the gap between their closest edges."""
    if stop_a < start_b:
        return start_b - stop_a
    elif stop_b < start_a:
        return start_a - stop_b
    else:
        return 0


# ---- load STAT hits, restricted to DEGs ----
print(f"\nLoading STAT hits ({STAT_DIR}):")
stat_hits = load_fimo(STAT_DIR)
stat_hits = restrict_to_motif_prefix(stat_hits, STAT_MOTIF_PREFIX, STAT_DIR)
stat_hits = merge_overlapping_hits(stat_hits, STAT_DIR)
stat_hits = restrict_to_degs(stat_hits, STAT_DIR)
if stat_hits is None or len(stat_hits) == 0:
    raise SystemExit("No STAT hits found in DEG genes -- nothing to check proximity against.")

# =============================================================================
# For each other family, check every STAT hit against every hit in that
# family WITHIN THE SAME GENE for a gap <= PROXIMITY_THRESH
# =============================================================================
all_results = []

for family_label, dir_name in OTHER_FAMILY_DIRS.items():
    print(f"\nLoading {family_label} hits ({dir_name}):")
    family_df = load_fimo(dir_name)
    family_df = restrict_to_motif_prefix(family_df, FAMILY_MOTIF_OF_INTEREST.get(family_label), dir_name)
    family_df = merge_overlapping_hits(family_df, dir_name)
    family_hits = restrict_to_degs(family_df, dir_name)
    if family_hits is None or len(family_hits) == 0:
        continue

    # group both by gene for efficient same-gene comparison
    stat_by_gene = {g: sub for g, sub in stat_hits.groupby("sequence_name")}
    family_by_gene = {g: sub for g, sub in family_hits.groupby("sequence_name")}

    shared_genes = set(stat_by_gene.keys()) & set(family_by_gene.keys())
    print(f"  Genes with BOTH a STAT hit and a {family_label} hit: {len(shared_genes)}")

    for gene in shared_genes:
        for _, stat_row in stat_by_gene[gene].iterrows():
            for _, fam_row in family_by_gene[gene].iterrows():
                dist = interval_distance(stat_row["start"], stat_row["stop"],
                                          fam_row["start"], fam_row["stop"])
                if dist <= PROXIMITY_THRESH:
                    all_results.append({
                        "gene": gene,
                        "stat_motif_id": stat_row["motif_id"],
                        "stat_start": stat_row["start"], "stat_stop": stat_row["stop"],
                        "stat_strand": stat_row["strand"],
                        "other_family": family_label,
                        "other_motif_id": fam_row["motif_id"],
                        "other_start": fam_row["start"], "other_stop": fam_row["stop"],
                        "other_strand": fam_row["strand"],
                        "distance_bp": dist,
                    })

results_df = pd.DataFrame(all_results)
print(f"\nTotal co-occurring pairs (<= {PROXIMITY_THRESH}bp) across all families: {len(results_df)}")

if len(results_df) > 0:
    results_df = results_df.sort_values(["gene", "distance_bp"])
    print("\nCounts per family:")
    print(results_df["other_family"].value_counts().to_string())
    print(f"\nUnique genes with at least one co-occurrence: {results_df['gene'].nunique()}")

out_csv = f"{out_dir}/STAT_family_motif_proximity_{PROXIMITY_THRESH}bp.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print("\nDone.")