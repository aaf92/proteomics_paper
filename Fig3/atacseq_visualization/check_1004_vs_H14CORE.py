"""
Quick check: of the 1004 expressed TFs, how many have a matching
motif entry in H14CORE at all? Same prefix-match logic as
run_fimo_array.sh's grep check (matching "MOTIF <symbol>.") -- this directly
tests whether the 716/1004 "no usable data" count from
10_STAT3_cooccurrence_background_distribution.py is mostly explained
by "motif doesn't exist in H14CORE" (expected/benign) rather than
something else.
"""

expressed_tfs_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/TF_enrichment/Results/log2_thresh_2.5_expressed_tfs.txt"
meme_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_motif_enrichment/H14CORE_meme_format.meme"

with open(expressed_tfs_f) as f:
    expressed_tfs = [line.strip() for line in f if line.strip()]
print(f"Expressed TFs: {len(expressed_tfs)}")

# collect every motif's gene-symbol prefix (text before the first ".")
with open(meme_f) as f:
    motif_symbols = set()
    for line in f:
        if line.startswith("MOTIF "):
            motif_id = line.split()[1]
            symbol = motif_id.split(".")[0]
            motif_symbols.add(symbol)

print(f"Unique gene symbols with a motif in H14CORE: {len(motif_symbols)}")

symbols_upper = {s.upper() for s in motif_symbols}

# EXACT match -- same case-sensitivity as run_fimo_array.sh's actual
# grep matching "MOTIF <symbol>." (no -i flag)
has_motif_exact = [tf for tf in expressed_tfs if tf in motif_symbols]
no_motif_exact = [tf for tf in expressed_tfs if tf not in motif_symbols]

# case-INsensitive match -- checks whether case mismatch (e.g. "Stat3"
# vs "STAT3") is the actual bottleneck, separate from genuinely
# missing motifs
has_motif_ci = [tf for tf in expressed_tfs if tf.upper() in symbols_upper]
no_motif_ci = [tf for tf in expressed_tfs if tf.upper() not in symbols_upper]

print(f"\n--- EXACT case match (matches run_fimo_array.sh's real grep behavior) ---")
print(f"Expressed TFs WITH a matching H14CORE motif: {len(has_motif_exact)}")
print(f"Expressed TFs with NO matching H14CORE motif: {len(no_motif_exact)}")

print(f"\n--- Case-INSENSITIVE match (checks if case mismatch alone explains gaps) ---")
print(f"Expressed TFs WITH a matching H14CORE motif: {len(has_motif_ci)}")
print(f"Expressed TFs with NO matching H14CORE motif: {len(no_motif_ci)}")

if len(has_motif_ci) > len(has_motif_exact):
    print(f"\n*** Case mismatch is losing {len(has_motif_ci) - len(has_motif_exact)} TFs that DO have a "
          f"motif in H14CORE but fail the exact-case grep in run_fimo_array.sh. ***")
    case_mismatch_examples = [tf for tf in expressed_tfs if tf not in motif_symbols and tf.upper() in symbols_upper]
    print(f"Examples: {case_mismatch_examples[:15]}")

print(f"\nFirst 30 genuinely missing from H14CORE (no match either way):")
print(no_motif_ci[:30])
