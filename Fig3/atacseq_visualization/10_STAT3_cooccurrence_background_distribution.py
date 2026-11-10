"""
Fig3 ATAC-seq visualization: formal background comparison for STAT3
co-occurrence. For EVERY TF with FIMO results (run_fimo_array.sh,
8000_TSS_8000_all_tfs/{TF}/fimo.tsv), computes an ENRICHMENT ratio,
not just a raw co-occurrence rate:

    DEG_rate(X)     = (# DEG genes where X's motif is within 50bp of a
                        STAT3 motif) / (# DEG genes where X has ANY hit)
    non_DEG_rate(X) = the SAME calculation, restricted to non-DEG
                       expressed genes instead of DEGs
    enrichment(X)   = DEG_rate(X) / non_DEG_rate(X)

This asks a sharper question than raw DEG co-occurrence alone: is
TF X's co-occurrence with STAT3 SPECIFICALLY CONCENTRATED in genes
that actually responded to IL6/IL10 (signal-dependent), rather than
just a general sequence-level tendency that would show up everywhere
regardless of whether the gene responded to cytokine signaling at
all. The background distribution is built on this enrichment ratio,
not the raw DEG rate.

"Expressed genes" universe = the gene index of one DESeq2 result
table (same file family as RNAseq_volcano_plots.py) -- DESeq2 only
reports genes that passed its own internal expression filtering, so
this is a reasonable proxy for "genes we actually measured", without
needing a separate dedicated expressed-genes file. non-DEG genes =
this expressed universe MINUS the DEG list (not "all genes genome-
wide", which would still contain the DEGs mixed in and dilute the
comparison).

THIS VERSION additionally computes a MACS3-peak-restricted version of
everything above: enrichment_in_peak(X), same formula, but a
co-occurring pair only counts if BOTH the TF hit and the STAT3 hit
fall inside a called ATAC peak (same overlap logic as
06_cooccurrence_ATAC_signal_tracks.py / 08's venn diagram). Both the
unrestricted and peak-restricted versions are computed in ONE pass
per TF (compute_rate_in_geneset handles both simultaneously), so the
expensive step -- loading and merging each TF's fimo.tsv -- only
happens once per TF regardless of how many restrictions are checked,
and everything needed for BOTH analyses is saved to one CSV so this
never needs to be rerun just to look at a different metric.

ONE clean background: all ~1004 expressed TFs MINUS every one of the
~65 expressed TFs with a DEsite MINUS STAT3 itself. All 65 DEsite TFs
are then evaluated against this SAME background -- deliberately not
restricted to pre-selecting ARID3A/ERF/ETV3/TFEB ahead of time; every
DEsite TF gets ranked the same way, and the ranking itself shows which
ones (if any) stand out. This avoids the selection bias of only ever
testing the 4 TFs already suspected of being interesting.

Same core logic as 05_STAT_family_motif_proximity_check.py (p-value
filter, same-TF-subtype merging, 50bp proximity), generalized to loop
over every TF instead of just the 4 of interest, and to compute the
rate in TWO gene sets (DEG and non-DEG) instead of one.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- data sources ----
all_tfs_fimo_base = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results/8000_TSS_8000_all_tfs"
stat3_fimo_dir = f"{all_tfs_fimo_base}/Stat3"  # STAT3's own per-TF directory from the same array run
deg_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/il6_il10_1and2_hour_common_DEGs.txt"
promoters_bed_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/8000_TSS_8000_promoters_mm10.bed"
narrowpeak_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/peakcall_res/UT/UT_merged_peaks.narrowPeak"
UPSTREAM = 8000
DOWNSTREAM = 8000
expressed_tfs_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/TF_enrichment/Results/log2_thresh_2.5_expressed_tfs.txt"
desite_tfs_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/FIMO_results/expressed_DEsite_TFs.txt"
# any one DESeq2 result table -- used purely for its gene index as the
# "expressed genes" universe, same file family as RNAseq_volcano_plots.py
expressed_genes_ref_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/RNAseq/DESeq2_results/res_il6_1_2.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

TFS_OF_INTEREST = ["Arid3a", "Erf", "Etv3", "Tfeb"]
PVAL_THRESH = 1e-4
PROXIMITY_THRESH = 50  # bp

for f, label in [(stat3_fimo_dir, "STAT3 FIMO dir"), (deg_f, "DEG list"),
                  (expressed_tfs_f, "expressed TFs list"), (desite_tfs_f, "DEsite TFs list")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label}: {f}")

deg_genes = set(np.loadtxt(deg_f, dtype=str).tolist())
print(f"Loaded {len(deg_genes)} common DEGs")

if not os.path.exists(expressed_genes_ref_f):
    raise SystemExit(f"Missing expressed-genes reference file: {expressed_genes_ref_f}")
expressed_genes_df = pd.read_csv(expressed_genes_ref_f, index_col=0)
expressed_genes = set(expressed_genes_df.index)
non_deg_genes = expressed_genes - deg_genes
print(f"Expressed genes (reference universe): {len(expressed_genes)}")
print(f"Non-DEG genes (expressed minus DEGs): {len(non_deg_genes)}")

with open(expressed_tfs_f) as f:
    all_expressed_tfs = [line.strip() for line in f if line.strip()]
with open(desite_tfs_f) as f:
    desite_tfs = [line.strip() for line in f if line.strip()]

# STAT3 is the reference every other TF is compared AGAINST -- it
# cannot also be a member of the test set, since checking "does
# STAT3's motif co-occur with STAT3's motif" is a trivial self-
# comparison (distance to itself is always 0), giving a meaningless
# enrichment of exactly 1.0 by construction, not a real result.
if "Stat3" in desite_tfs:
    desite_tfs.remove("Stat3")
    print("Removed Stat3 from the DEsite TF test set (it's the reference, not a comparison target)")

print(f"Broad background pool: {len(all_expressed_tfs)} TFs")
print(f"Narrow background pool: {len(desite_tfs)} TFs")


load_tf_fimo_diagnostics = {"no_file": 0, "empty_file": 0, "zero_after_filter": 0}


def load_tf_fimo(tf_dir, label):
    """Load a TF's fimo.tsv, apply p-value filter (defensive re-check
    -- the FIMO run already filtered at --thresh 1e-4, but re-applying
    here costs nothing and protects against a future change in how the
    FIMO run was configured), then merge overlapping/adjacent hits
    (same logic as 05's merge_overlapping_hits, needed for TFs with
    multiple HOCOMOCO subtype PWMs).

    Tracks WHY a TF produced no usable data, since "no fimo.tsv at all"
    (motif wasn't in H14CORE -- expected/benign, run_fimo_array.sh
    skips these) is a very different situation from "file exists but
    zero hits passed the threshold" (worth knowing if this happens a
    lot, vs. being rare/expected for short motifs)."""
    f = f"{tf_dir}/fimo.tsv"
    if not os.path.exists(f):
        load_tf_fimo_diagnostics["no_file"] += 1
        return None
    df = pd.read_csv(f, sep="\t", comment="#")
    if len(df) == 0:
        load_tf_fimo_diagnostics["empty_file"] += 1
        return None
    df = df.dropna(subset=["sequence_name", "start", "stop", "p-value"])
    df = df[df["p-value"] < PVAL_THRESH]
    if len(df) == 0:
        load_tf_fimo_diagnostics["zero_after_filter"] += 1
        return None

    merged_rows = []
    for gene, sub in df.groupby("sequence_name"):
        sub = sub.sort_values("start")
        current_stop, current_group = None, []
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

    merged = pd.DataFrame(merged_rows)
    return merged


def interval_distance(start_a, stop_a, start_b, stop_b):
    if stop_a < start_b:
        return start_b - stop_a
    elif stop_b < start_a:
        return start_a - stop_b
    else:
        return 0


# ---- load STAT3 hits once (unrestricted), then build separate
# per-gene-set views for DEG and non-DEG ----
stat3_hits_all = load_tf_fimo(stat3_fimo_dir, "Stat3")
if stat3_hits_all is None:
    raise SystemExit("No STAT3 hits found -- check the FIMO run completed for Stat3.")

stat3_by_gene_deg = {g: sub for g, sub in stat3_hits_all[stat3_hits_all["sequence_name"].isin(deg_genes)]
                      .groupby("sequence_name")}
stat3_by_gene_nondeg = {g: sub for g, sub in stat3_hits_all[stat3_hits_all["sequence_name"].isin(non_deg_genes)]
                         .groupby("sequence_name")}
print(f"STAT3 hits (after merge): {len(stat3_by_gene_deg)} DEG genes, "
      f"{len(stat3_by_gene_nondeg)} non-DEG genes")


for f, label in [(promoters_bed_f, "promoters BED"), (narrowpeak_f, "narrowPeak")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label}: {f}")

promoters_bed = pd.read_csv(
    promoters_bed_f, sep="\t", header=None,
    names=["chrom", "start", "end", "gene", "score", "strand"]
).set_index("gene")

narrowpeak_cols = ["chrom", "start", "end", "name", "score", "strand",
                    "signalValue", "pValue", "qValue", "summit"]
peaks_df = pd.read_csv(narrowpeak_f, sep="\t", header=None, names=narrowpeak_cols)
print(f"Loaded {len(peaks_df)} MACS3 peak calls")

# cache of per-gene MACS3 peak spans, IN GENOMIC COORDINATES (computed
# once per gene the first time it's needed, reused for every TF after
# that -- avoids re-filtering peaks_df repeatedly for the same gene)
_gene_peak_cache = {}


def get_gene_peaks_genomic(gene):
    """Genomic (start, end) tuples for MACS3 peaks overlapping this
    gene's promoter window. Cached per gene."""
    if gene in _gene_peak_cache:
        return _gene_peak_cache[gene]
    if gene not in promoters_bed.index:
        _gene_peak_cache[gene] = []
        return []
    row = promoters_bed.loc[gene]
    chrom, bed_start, bed_end = row["chrom"], int(row["start"]), int(row["end"])
    overlapping = peaks_df[
        (peaks_df["chrom"] == chrom) & (peaks_df["start"] < bed_end) & (peaks_df["end"] > bed_start)
    ]
    spans = list(zip(overlapping["start"], overlapping["end"]))
    _gene_peak_cache[gene] = spans
    return spans


def fimo_pos_to_genomic(fimo_pos, gene):
    """Converts a FIMO in-sequence position to a genomic coordinate,
    using this gene's own window info (bed_start/bed_end/strand).
    Mirrors for - strand genes, since their extracted sequence was
    reverse-complemented at extraction time (bedtools getfasta -s)."""
    row = promoters_bed.loc[gene]
    bed_start, bed_end, strand = int(row["start"]), int(row["end"]), row["strand"]
    if strand == "+":
        return bed_start + fimo_pos
    else:
        return bed_end - fimo_pos


def hit_in_peak(row, gene):
    """True if this FIMO hit's genomic span overlaps ANY MACS3 peak
    for this gene."""
    peak_spans = get_gene_peaks_genomic(gene)
    if not peak_spans:
        return False
    g_start = fimo_pos_to_genomic(row["start"], gene)
    g_stop = fimo_pos_to_genomic(row["stop"], gene)
    g_lo, g_hi = min(g_start, g_stop), max(g_start, g_stop)
    for (p_start, p_end) in peak_spans:
        if g_lo < p_end and g_hi > p_start:
            return True
    return False


def compute_rate_in_geneset(tf_hits, gene_set, stat3_by_gene):
    """Co-occurrence restricted to ONE gene set (DEGs or non-DEGs).
    Computes THREE things in one pass, since the expensive part
    (loading/merging each TF's fimo.tsv) only needs to happen once:

    1. GENE-LEVEL rate, unrestricted (one gene = one observation,
       regardless of how many co-occurring pairs it has).
    2. GENE-LEVEL rate, peak-restricted -- CORRECTED denominator.
       Previously this used the same denominator as (1) (all genes
       with ANY hit), which confounds "does X's motif tend to sit in
       open chromatin at all" with "does STAT3 co-occur with X given
       both are accessible" -- two different properties. Fixed here:
       denominator is now genes where X's hit is ALREADY in a peak,
       and the numerator additionally requires the co-occurring STAT3
       hit to also be in a peak. This isolates the conditional
       question properly.
    3. OCCURRENCE-LEVEL rate -- counts individual motif occurrences,
       not unique genes, so a gene with 5 X-occurrences contributes up
       to 5 observations instead of 1. Reported alongside gene-level
       for direct comparison; they can legitimately diverge if a TF's
       co-occurrence is concentrated at a few dense loci rather than
       spread across many genes.

    Returns a dict, or None if this TF has zero hits in this gene set."""
    sub_hits = tf_hits[tf_hits["sequence_name"].isin(gene_set)]
    if len(sub_hits) == 0:
        return None

    tf_by_gene = {g: sub for g, sub in sub_hits.groupby("sequence_name")}
    n_genes_with_hit = len(tf_by_gene)
    n_occurrences_total = len(sub_hits)

    n_cooccurring = 0                  # gene-level, unrestricted
    n_genes_with_hit_in_peak = 0       # CORRECTED denominator
    n_cooccurring_in_peak = 0          # CORRECTED numerator
    n_occurrences_cooccurring = 0      # occurrence-level

    for gene, tf_sub in tf_by_gene.items():
        stat3_sub = stat3_by_gene.get(gene)

        gene_has_any_hit_in_peak = False
        gene_cooccurs = False
        gene_cooccurs_in_peak = False

        for _, trow in tf_sub.iterrows():
            trow_in_peak = hit_in_peak(trow, gene)
            if trow_in_peak:
                gene_has_any_hit_in_peak = True

            this_occurrence_cooccurs = False
            if stat3_sub is not None:
                for _, srow in stat3_sub.iterrows():
                    if interval_distance(trow["start"], trow["stop"], srow["start"], srow["stop"]) <= PROXIMITY_THRESH:
                        this_occurrence_cooccurs = True
                        gene_cooccurs = True
                        if trow_in_peak and hit_in_peak(srow, gene):
                            gene_cooccurs_in_peak = True

            if this_occurrence_cooccurs:
                n_occurrences_cooccurring += 1

        if gene_has_any_hit_in_peak:
            n_genes_with_hit_in_peak += 1
        if gene_cooccurs:
            n_cooccurring += 1
        if gene_has_any_hit_in_peak and gene_cooccurs_in_peak:
            n_cooccurring_in_peak += 1

    rate = n_cooccurring / n_genes_with_hit
    rate_in_peak = _safe_ratio(n_cooccurring_in_peak, n_genes_with_hit_in_peak)
    occurrence_rate = n_occurrences_cooccurring / n_occurrences_total

    return {
        "rate": rate, "n_genes_with_hit": n_genes_with_hit, "n_cooccurring": n_cooccurring,
        "rate_in_peak": rate_in_peak, "n_genes_with_hit_in_peak": n_genes_with_hit_in_peak,
        "n_cooccurring_in_peak": n_cooccurring_in_peak,
        "occurrence_rate": occurrence_rate, "n_occurrences_total": n_occurrences_total,
        "n_occurrences_cooccurring": n_occurrences_cooccurring,
    }


def _safe_ratio(numerator, denominator):
    """NaN (not 0 or inf) if the denominator is exactly 0 or missing --
    an undefined enrichment should be excluded from downstream
    distributions, not silently treated as 0 or infinite."""
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def compute_enrichment(tf, tf_dir):
    """Returns a dict with BOTH the unrestricted and MACS3-peak-
    restricted DEG rate, non-DEG rate, and enrichment ratio -- all
    computed in ONE pass per TF (see compute_rate_in_geneset), so the
    expensive step (loading + merging each TF's fimo.tsv) only
    happens once regardless of how many restrictions are checked.
    Returns None if this TF has no usable FIMO data at all."""
    tf_hits = load_tf_fimo(tf_dir, tf)
    if tf_hits is None:
        return None

    deg_res = compute_rate_in_geneset(tf_hits, deg_genes, stat3_by_gene_deg)
    nondeg_res = compute_rate_in_geneset(tf_hits, non_deg_genes, stat3_by_gene_nondeg)

    if deg_res is None and nondeg_res is None:
        return None

    empty = {"rate": np.nan, "n_genes_with_hit": 0, "n_cooccurring": 0,
              "rate_in_peak": np.nan, "n_genes_with_hit_in_peak": 0, "n_cooccurring_in_peak": 0,
              "occurrence_rate": np.nan, "n_occurrences_total": 0, "n_occurrences_cooccurring": 0}
    deg_res = deg_res or empty
    nondeg_res = nondeg_res or empty

    enrichment = _safe_ratio(deg_res["rate"], nondeg_res["rate"])
    enrichment_in_peak = _safe_ratio(deg_res["rate_in_peak"], nondeg_res["rate_in_peak"])
    occurrence_enrichment = _safe_ratio(deg_res["occurrence_rate"], nondeg_res["occurrence_rate"])

    return {
        # gene-level, unrestricted
        "deg_rate": deg_res["rate"], "deg_n_genes_with_hit": deg_res["n_genes_with_hit"],
        "deg_n_cooccurring": deg_res["n_cooccurring"],
        "nondeg_rate": nondeg_res["rate"], "nondeg_n_genes_with_hit": nondeg_res["n_genes_with_hit"],
        "nondeg_n_cooccurring": nondeg_res["n_cooccurring"],
        "enrichment": enrichment,

        # gene-level, peak-restricted (CORRECTED denominator -- see
        # compute_rate_in_geneset docstring)
        "deg_rate_in_peak": deg_res["rate_in_peak"],
        "deg_n_genes_with_hit_in_peak": deg_res["n_genes_with_hit_in_peak"],
        "deg_n_cooccurring_in_peak": deg_res["n_cooccurring_in_peak"],
        "nondeg_rate_in_peak": nondeg_res["rate_in_peak"],
        "nondeg_n_genes_with_hit_in_peak": nondeg_res["n_genes_with_hit_in_peak"],
        "nondeg_n_cooccurring_in_peak": nondeg_res["n_cooccurring_in_peak"],
        "enrichment_in_peak": enrichment_in_peak,

        # occurrence-level (individual motif instances, not unique genes)
        "deg_occurrence_rate": deg_res["occurrence_rate"],
        "deg_n_occurrences_total": deg_res["n_occurrences_total"],
        "deg_n_occurrences_cooccurring": deg_res["n_occurrences_cooccurring"],
        "nondeg_occurrence_rate": nondeg_res["occurrence_rate"],
        "nondeg_n_occurrences_total": nondeg_res["n_occurrences_total"],
        "nondeg_n_occurrences_cooccurring": nondeg_res["n_occurrences_cooccurring"],
        "occurrence_enrichment": occurrence_enrichment,
    }


# =============================================================================
# Compute rates for everything: 4 TFs of interest, broad background,
# narrow background. Both backgrounds exclude the TFs of interest and
# STAT3 itself.
# =============================================================================
# CLEAN background = all expressed TFs MINUS every one of the 65
# DEsite TFs (not just the 4 TFs of interest) MINUS STAT3 itself. The
# 65 DEsite TFs are the TEST SET, evaluated against this background --
# this is deliberately NOT restricted to pre-selecting which of the 65
# to highlight; every one of them gets checked against the same clean
# background, letting the data show which (if any) stand out.
exclude_from_background = set(desite_tfs) | {"Stat3"}

results = []
all_tfs_to_check = set(desite_tfs) | set(all_expressed_tfs)
print(f"\nComputing co-occurrence rate for {len(all_tfs_to_check)} total TFs...")

# reset diagnostics so the breakdown below only reflects the background/
# TFs-of-interest loop, not the earlier STAT3 load
load_tf_fimo_diagnostics["no_file"] = 0
load_tf_fimo_diagnostics["empty_file"] = 0
load_tf_fimo_diagnostics["zero_after_filter"] = 0

n_done, n_skipped = 0, 0
sorted_tfs = sorted(all_tfs_to_check)
n_total_tfs = len(sorted_tfs)
loop_start_time = time.time()

for i, tf in enumerate(sorted_tfs, start=1):
    tf_dir = f"{all_tfs_fimo_base}/{tf}"
    res = compute_enrichment(tf, tf_dir)

    elapsed = time.time() - loop_start_time
    rate = elapsed / i
    remaining = rate * (n_total_tfs - i)
    print(f"  [{i}/{n_total_tfs}] {tf} -- elapsed {elapsed:.1f}s, "
          f"~{rate:.3f}s/TF, est. remaining {remaining:.1f}s")

    if res is None:
        n_skipped += 1
        continue
    results.append({
        "tf": tf,
        "is_tf_of_interest": tf in TFS_OF_INTEREST,
        "is_desite_tf": tf in desite_tfs,  # the test set -- all 65, not pre-selected
        "in_background": tf in all_expressed_tfs and tf not in exclude_from_background,
        **res,
    })
    n_done += 1

print(f"Computed rates for {n_done} TFs, skipped {n_skipped} (no usable FIMO data)")
print(f"  Breakdown of skipped TFs:")
print(f"    No fimo.tsv at all (motif not in H14CORE, skipped by run_fimo_array.sh): "
      f"{load_tf_fimo_diagnostics['no_file']}")
print(f"    fimo.tsv exists but completely empty (0 rows):               "
      f"{load_tf_fimo_diagnostics['empty_file']}")
print(f"    fimo.tsv has rows but 0 pass p<{PVAL_THRESH} after re-filter: "
      f"{load_tf_fimo_diagnostics['zero_after_filter']}")

results_df = pd.DataFrame(results).sort_values("enrichment", ascending=False)
results_csv = f"{out_dir}/STAT3_cooccurrence_enrichment_all_TFs_withPeaks.csv"
results_df.to_csv(results_csv, index=False)
print(f"\nSaved: {results_csv}")

n_nan_enrichment = results_df["enrichment"].isna().sum()
print(f"\n{n_nan_enrichment}/{len(results_df)} TFs have an undefined enrichment ratio "
      f"(zero co-occurrence in non-DEG genes -- can't divide by zero; these are EXCLUDED "
      f"from the background distribution below, not treated as 0 or infinite)")

print(f"\nTFs of interest:\n"
      f"{results_df[results_df['is_tf_of_interest']][['tf', 'deg_rate', 'nondeg_rate', 'enrichment']].to_string(index=False)}")


def percentile_rank(value, background_values):
    """Fraction of background values <= this value -- i.e. this TF's
    percentile position in the background distribution."""
    background_values = np.array(background_values)
    return (background_values <= value).mean() * 100


TF_COLOR_MARK = {"Arid3a": "#8E6BA8", "Erf": "#2E6F95", "Etv3": "#D08B3A", "Tfeb": "#4C9F70"}
OTHER_DESITE_COLOR = "#444444"


def rank_and_plot(metric_col, metric_label, file_tag):
    """Percentile-ranks every DEsite TF against the clean background
    for ONE metric column (either 'enrichment' or
    'enrichment_in_peak'), prints the ranking, saves a CSV, and saves
    a histogram+rug-plot figure. Parameterized so the unrestricted and
    peak-restricted versions share identical logic rather than
    duplicating it."""
    bg_mask = results_df["in_background"] & results_df[metric_col].notna()
    bg_vals = results_df.loc[bg_mask, metric_col].values
    n_bg_nan = (results_df["in_background"] & results_df[metric_col].isna()).sum()
    print(f"\n--- Background for {metric_label} (n={len(bg_vals)}, "
          f"{n_bg_nan} excluded for undefined ratio) ---")
    if len(bg_vals) == 0:
        print(f"    No background TFs have a defined {metric_label} -- skipping ranking/plot for this metric.")
        return None
    print(f"    mean={bg_vals.mean():.3f}, median={np.median(bg_vals):.3f}, std={bg_vals.std():.3f}")

    desite_rows = results_df[results_df["is_desite_tf"] & results_df[metric_col].notna()].copy()
    if len(desite_rows) == 0:
        print(f"    No DEsite TFs have a defined {metric_label} -- skipping ranking/plot for this metric.")
        return None

    desite_rows["percentile_vs_background"] = desite_rows[metric_col].apply(
        lambda v: percentile_rank(v, bg_vals))
    desite_rows["empirical_p_vs_background"] = desite_rows[metric_col].apply(
        lambda v: (bg_vals >= v).mean())
    desite_rows = desite_rows.sort_values("percentile_vs_background", ascending=False)

    print(f"\nAll {len(desite_rows)} DEsite TFs ranked against the clean background "
          f"for {metric_label} ({TFS_OF_INTEREST} marked with *):")
    for _, row in desite_rows.iterrows():
        marker = " *" if row["tf"] in TFS_OF_INTEREST else ""
        print(f"    {row['tf']}{marker}: {metric_col}={row[metric_col]:.3f}, "
              f"percentile={row['percentile_vs_background']:.1f}, "
              f"empirical p={row['empirical_p_vs_background']:.4f}")

    ranked_csv = f"{out_dir}/DEsite_TFs_ranked_vs_background_{file_tag}.csv"
    desite_rows.to_csv(ranked_csv, index=False)
    print(f"\nSaved: {ranked_csv}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bg_vals, bins=30, color="#B0B0B0", alpha=0.8, edgecolor="white",
            label=f"Background (n={len(bg_vals)})")

    y_rug = -ax.get_ylim()[1] * 0.04
    for _, row in desite_rows.iterrows():
        tf = row["tf"]
        is_interest = tf in TFS_OF_INTEREST
        color = TF_COLOR_MARK.get(tf, OTHER_DESITE_COLOR)
        ax.plot(row[metric_col], y_rug, marker="|", color=color,
                 markersize=14 if is_interest else 8,
                 markeredgewidth=2.2 if is_interest else 1.0, zorder=5)
        if is_interest:
            ax.annotate(tf, (row[metric_col], y_rug), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_ylim(bottom=y_rug * 1.8)
    ax.set_title(f"All {len(desite_rows)} expressed TFs with a DEsite, ranked against the "
                 f"clean background\n({metric_label}; background = all expressed TFs minus "
                 f"every DEsite TF minus STAT3)", fontsize=10)
    ax.set_xlabel(f"STAT3 co-occurrence {metric_label} (DEG rate / non-DEG rate)")
    ax.set_ylabel("Number of background TFs")

    legend_handles = [plt.Line2D([0], [0], color="#B0B0B0", lw=6, label="Background distribution"),
                      plt.Line2D([0], [0], marker="|", color=OTHER_DESITE_COLOR, linestyle="none",
                                  markersize=8, label="Other DEsite TFs (n=" +
                                  str(len(desite_rows) - sum(desite_rows["tf"].isin(TFS_OF_INTEREST))) + ")")]
    for tf in TFS_OF_INTEREST:
        if tf in desite_rows["tf"].values:
            legend_handles.append(plt.Line2D([0], [0], marker="|", color=TF_COLOR_MARK[tf], linestyle="none",
                                               markersize=12, markeredgewidth=2, label=tf))
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="upper right")

    plt.tight_layout()
    hist_png = f"{out_dir}/STAT3_cooccurrence_background_distribution_{file_tag}.png"
    hist_pdf = f"{out_dir}/STAT3_cooccurrence_background_distribution_{file_tag}.pdf"
    plt.savefig(hist_png, dpi=300, bbox_inches="tight")
    plt.savefig(hist_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {hist_png}")
    return desite_rows


# =============================================================================
# Run both versions -- unrestricted, then MACS3-peak-restricted.
# Nothing here re-loads or re-merges any TF's fimo.tsv; everything
# needed for both was already computed in the main loop above and is
# sitting in results_df, so this reuses that saved work directly.
# =============================================================================
rank_and_plot("enrichment", "unrestricted", "unrestricted")
rank_and_plot("enrichment_in_peak", "MACS3-peak-restricted", "withinPeaks")

# =============================================================================
# Gene-level vs. occurrence-level comparison, for the DEsite TF test
# set specifically. These CAN legitimately disagree: gene-level treats
# a gene with 1 co-occurring pair the same as a gene with 5 (avoids
# pseudo-replication from one dense locus dominating), occurrence-
# level counts every individual pair (more sensitive to density, but
# risks exactly that pseudo-replication). Plotting both lets a
# divergence be SEEN rather than silently picking one.
# =============================================================================
compare_df = results_df[results_df["is_desite_tf"] &
                          results_df["enrichment"].notna() &
                          results_df["occurrence_enrichment"].notna()].copy()

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_box_aspect(1)

other_mask = ~compare_df["tf"].isin(TFS_OF_INTEREST)
ax.scatter(compare_df.loc[other_mask, "enrichment"], compare_df.loc[other_mask, "occurrence_enrichment"],
           color=OTHER_DESITE_COLOR, alpha=0.6, s=30, label=f"Other DEsite TFs (n={other_mask.sum()})")

for tf in TFS_OF_INTEREST:
    row = compare_df[compare_df["tf"] == tf]
    if len(row) == 0:
        continue
    ax.scatter(row["enrichment"], row["occurrence_enrichment"], color=TF_COLOR_MARK[tf],
               s=100, edgecolors="black", linewidths=1, zorder=5, label=tf)

axis_min = min(compare_df["enrichment"].min(), compare_df["occurrence_enrichment"].min()) * 0.9
axis_max = max(compare_df["enrichment"].max(), compare_df["occurrence_enrichment"].max()) * 1.1
ax.plot([axis_min, axis_max], [axis_min, axis_max], color="grey", linestyle="--", linewidth=0.8, zorder=1,
        label="y = x (perfect agreement)")
ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)

ax.set_xlabel("Gene-level enrichment")
ax.set_ylabel("Occurrence-level enrichment")
ax.set_title("Gene-level vs. occurrence-level enrichment, DEsite TFs\n"
             "(points far from y=x indicate the two metrics disagree for that TF)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")

plt.tight_layout()
compare_png = f"{out_dir}/geneLevel_vs_occurrenceLevel_enrichment.png"
compare_pdf = f"{out_dir}/geneLevel_vs_occurrenceLevel_enrichment.pdf"
plt.savefig(compare_png, dpi=300, bbox_inches="tight")
plt.savefig(compare_pdf, bbox_inches="tight")
plt.show()
print(f"\nSaved: {compare_png}")

print("\nDone.")