import os
import numpy as np
import pandas as pd

import kinase_library as kl
import pickle

# =============================================================================
# PATHS
# =============================================================================
phospho_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/phosphoproteomics/stat_results_IL6IL10phosphopairwiseBMDMheatmap.csv"
out_path       = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig2/KL_motif_enrichment_res"

os.makedirs(f"{out_path}/figures", exist_ok=True)
os.makedirs(f"{out_path}/out_files", exist_ok=True)

# =============================================================================
# STEP 1: LOAD PHOSPHOPROTEOMICS DATA
# =============================================================================

prot_df = pd.read_csv(phospho_data_f)
prot_df['geneSymbol'] = prot_df['id'].str.split('_').str[0]
prot_df['id'] = prot_df['id'].str.split('_', n=1).str[1]

sign_logp_cols = [
    "sign.logP.IL6_20min_over_IL10_20min",
    "sign.logP.IL6_45min_over_IL10_45min",
    "sign.logP.IL6_90min_over_IL10_90min",
    "sign.logP.IL6_20min_over_0min",
    "sign.logP.IL6_45min_over_0min",
    "sign.logP.IL6_90min_over_0min",
    "sign.logP.IL10_20min_over_0min",
    "sign.logP.IL10_45min_over_0min",
    "sign.logP.IL10_90min_over_0min",
]

prot_df = prot_df[["id", "geneSymbol", "VMsiteFlanks"] + sign_logp_cols]
prot_df = prot_df.dropna().reset_index(drop=True)

# =============================================================================
# STEP 2: EXPLODE MULTI-SITE FLANKING SEQUENCES
# =============================================================================

prot_df_exploded = (
    prot_df[["id", "geneSymbol", "VMsiteFlanks"] + sign_logp_cols]
    .copy()
    .assign(VMsiteFlanks=lambda df: df["VMsiteFlanks"].str.split("|"))
    .explode("VMsiteFlanks")
    .dropna(subset=["VMsiteFlanks"])
    .assign(VMsiteFlanks=lambda df: df["VMsiteFlanks"].str.strip())
    .reset_index(drop=True)
)

print(f"Sites before explode: {len(prot_df)}, after: {len(prot_df_exploded)}")

# =============================================================================
# STEP 3: MULTI-CONDITION MEA
# =============================================================================

kin_configs = {
    'ser_thr': {'method': 'percentile_rank', 'thresh': 15},
    'tyrosine': {'method': 'percentile_rank', 'thresh': 8},
}

# Pre-compute percentiles for both kinase types upfront
pps_data = kl.PhosphoProteomics(prot_df_exploded, seq_col='VMsiteFlanks')
ser_thr_percentiles = pps_data.percentile('ser_thr', values_only=True)
tyr_percentiles     = pps_data.percentile('tyrosine', values_only=True)

percentiles_map = {
    'ser_thr':  ser_thr_percentiles,
    'tyrosine': tyr_percentiles,
}

# NOTE: only need `mea_res.enrichment_results`
enrichment_tables = {'ser_thr': {}, 'tyrosine': {}}

for kin_type, config in kin_configs.items():
    print(f"\n=== {kin_type} ===")
    for i, contrast in enumerate(sign_logp_cols):
        print(f"  {i}: {contrast}")
        ranked_phos = kl.RankedPhosData(dp_data=prot_df_exploded, rank_col=contrast, seq_col='VMsiteFlanks')
        ranked_phos.submit_percentiles(kin_type, percentiles_map[kin_type], suppress_messages=True)
        mea_res = ranked_phos.mea(kin_type=kin_type, kl_method=config['method'], kl_thresh=config['thresh'])

        # keep only what's actually needed downstream
        enrichment_tables[kin_type][contrast] = mea_res.enrichment_results.copy()

        # save per-contrast CSV too -- tiny, human-readable, easy to spot-check
        # without needing kinase_library or pickle at all
        csv_out = f"{out_path}/out_files/enrichment_{kin_type}_{contrast.split('.')[-1]}.csv"
        mea_res.enrichment_results.to_csv(csv_out)

        del mea_res, ranked_phos  # explicit, though loop reassignment would do this anyway

# =============================================================================
# STEP 4: CACHE KINASE FAMILY INFO ONCE (static across contrasts, no need to
# store/recompute it per contrast downstream)
# =============================================================================
all_kinases_seen = set()
for kin_type in enrichment_tables:
    for contrast, tbl in enrichment_tables[kin_type].items():
        all_kinases_seen.update(tbl.index.tolist())

kinase_family_map = {}
for kinase in all_kinases_seen:
    kin = kl.get_kinase(kinase)
    kinase_family_map[kinase] = kin.family

# =============================================================================
# SAVE — lightweight pickle: just the enrichment DataFrames + family lookup.
# This should be a few MB at most, not ~2GB.
# =============================================================================
lightweight_out = {
    "enrichment_tables": enrichment_tables,   # {kin_type: {contrast: DataFrame}}
    "kinase_family_map": kinase_family_map,   # {kinase_name: family}
}

with open(f"{out_path}/out_files/kl_en_res_lightweight.pkl", "wb") as f:
    pickle.dump(lightweight_out, f)

