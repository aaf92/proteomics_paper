"""
Single-gene dot plot -- both IL6 and IL10, 5-6 groups: PBS_NTC,
PBS_edit, IL6_NTC, IL10_NTC, IL6_edit, IL10_edit.

Data source depends on the edit:
  - Stat3Y705H (spans B1 and B2): uses the UPSTREAM ComBat-corrected
    file (il6_il10_NTC_ctrl_centered_combatCorrected.csv), same file
    NTC_degs_STAT3Y705H_heatmap.py uses. B1/B2 are pooled directly --
    ComBat already made them comparable, so no further batch handling
    is needed or done here.
  - Single-batch edits (Arid3aT132A, ErfT529A, Etv3S29P, TfebS392P):
    uses raw counts (size-factor normalized, NO ComBat) -- there's no
    batch issue to correct for a single-batch edit in the first place.

NOTE: the ComBat file's values are already control-centered (per its
own filename/prior script convention), NOT raw log2-normalized
expression -- so Stat3's y-axis is not directly on the same absolute
scale as the single-batch edits' y-axis (raw log2 expression). This is
an intentional consequence of using the two different, appropriate-
for-each-case data sources, not a bug -- don't compare the two
y-axes' absolute values directly across different TFs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- data sources ----
raw_data_f = "/ix/djishnu/Aaron_F/proteomics_proj/Data/be_validation_RNAseq/feature_counts_output/mRNA_raw_reads_ordered.csv"
combat_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/be_validation_RNAseq/be_effect_analysis/out_files/il6_il10_NTC_ctrl_centered_combatCorrected.csv"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Supplement/singleGene_dotplots"
os.makedirs(out_dir, exist_ok=True)

TF = "Stat3Y705H"  # Arid3aT132A, Etv3S29P, ErfT529A, TfebS392P, Stat3Y705H all supported
NTC_LABEL = "NTC"
GENE = "Tnfrsf1a"
USE_COMBAT = (TF == "Stat3Y705H")

CYTOKINE_COLOR = {"IL6": "#F4C430", "IL10": "#88CCEE"}
GROUP_ORDER = [("PBS", NTC_LABEL), ("PBS", TF), ("IL6", NTC_LABEL), ("IL10", NTC_LABEL), ("IL6", TF), ("IL10", TF)]
GROUP_COLOR = {"PBS": "#B0B0B0", "IL6": CYTOKINE_COLOR["IL6"], "IL10": CYTOKINE_COLOR["IL10"]}
GROUP_MARKER = {NTC_LABEL: "o", TF: "s"}


def deseq2_size_factors(counts):
    """Median-of-ratios size factors, DESeq2's default method."""
    log_counts = np.log(counts.replace(0, np.nan))
    log_geo_means = log_counts.mean(axis=1)
    valid = np.isfinite(log_geo_means)
    ratios = log_counts.loc[valid].sub(log_geo_means.loc[valid], axis=0)
    return np.exp(ratios.median(axis=0))


if USE_COMBAT:
    print(f"{TF} spans multiple batches -- using ComBat-corrected data: {combat_f}")
    if not os.path.exists(combat_f):
        raise SystemExit(f"ComBat file not found: {combat_f}")
    data = pd.read_csv(combat_f, index_col=0)
    y_label = "ctrl-centered expression\n(ComBat-corrected)"
else:
    print(f"{TF} is single-batch -- using raw counts (NO ComBat): {raw_data_f}")
    raw_data = pd.read_csv(raw_data_f, index_col=0)
    sf = deseq2_size_factors(raw_data)
    data = np.log2(raw_data.div(sf, axis=1) + 1)
    y_label = "log2 normalized expression\n(raw counts, no ComBat)"

if GENE not in data.index:
    raise SystemExit(f"'{GENE}' not found in {'the ComBat file' if USE_COMBAT else 'the expression data'} "
                      f"-- check gene symbol/capitalization, or whether this file's gene coverage is restricted "
                      f"(e.g. to a DEG subset) if using ComBat.")

col_tokens = [c.split("_") for c in data.columns]
col_meta = pd.DataFrame({
    "sample": data.columns.tolist(),
    "batch": [t[0] for t in col_tokens],
    "stim": [t[1] for t in col_tokens],
    "pert": [t[2] for t in col_tokens],
})


def get_group_cols(stim_g, pert_g):
    """Pools across ALL batches present for this group -- safe for
    both cases: ComBat already made B1/B2 comparable for Stat3, and
    single-batch edits only ever have one batch anyway."""
    mask = (col_meta["stim"] == stim_g) & (col_meta["pert"] == pert_g)
    return col_meta.loc[mask, "sample"].tolist()


group_cols_cache = {(stim, pert): get_group_cols(stim, pert) for stim, pert in GROUP_ORDER}

# ---- fold changes, for reference in the title -- simple group-mean
# subtraction is valid here for BOTH cases: ComBat already resolved
# the batch issue for Stat3, and there's no batch issue at all for
# single-batch edits ----
def group_mean(stim_g, pert_g):
    cols = group_cols_cache.get((stim_g, pert_g), [])
    return data.loc[GENE, cols].mean() if cols else np.nan


il6_fc = group_mean("IL6", TF) - group_mean("IL6", NTC_LABEL)
il10_fc = group_mean("IL10", TF) - group_mean("IL10", NTC_LABEL)
print(f"{GENE}: IL6 delta (edit vs NTC) = {il6_fc:.3f}  |  IL10 delta (edit vs NTC) = {il10_fc:.3f}")

# ---- save the underlying per-sample values plotted below ----
csv_rows = []
for stim, pert in GROUP_ORDER:
    cols = group_cols_cache[(stim, pert)]
    for c in cols:
        batch = col_meta.loc[col_meta["sample"] == c, "batch"].iloc[0]
        csv_rows.append({
            "gene": GENE, "stim": stim, "pert": pert, "sample": c, "batch": batch,
            "value": data.loc[GENE, c], "data_source": "combat" if USE_COMBAT else "raw_no_combat",
        })
values_df = pd.DataFrame(csv_rows)
values_csv = f"{out_dir}/{TF}_{GENE}_dotplot_values.csv"
values_df.to_csv(values_csv, index=False)
print(f"Saved: {values_csv}")

fig, ax = plt.subplots(figsize=(6, 5))

for x, (stim, pert) in enumerate(GROUP_ORDER):
    cols = group_cols_cache[(stim, pert)]
    if len(cols) == 0:
        continue
    vals = data.loc[GENE, cols].values.astype(float)
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), x) + jitter, vals,
               color=GROUP_COLOR[stim], marker=GROUP_MARKER[pert],
               s=60, edgecolors="black", linewidths=0.5, alpha=0.85, zorder=3)
    ax.plot([x - 0.15, x + 0.15], [vals.mean(), vals.mean()],
            color="black", linewidth=1.5, zorder=4)

ax.axhline(0, color="grey", linewidth=0.6, linestyle=":", zorder=0)
ax.set_xticks(range(len(GROUP_ORDER)))
ax.set_xticklabels([f"{pert}\n{stim}" for stim, pert in GROUP_ORDER], fontsize=9)
ax.set_ylabel(y_label, fontsize=10)
ax.set_title(f"{GENE} -- {TF}\nIL6 delta={il6_fc:.2f}, IL10 delta={il10_fc:.2f} (edit vs NTC)\n"
             f"data source: {'ComBat-corrected' if USE_COMBAT else 'raw counts, no ComBat'}",
             fontsize=10, fontweight="bold")
ax.tick_params(labelsize=9)

group_legend = [plt.Line2D([0], [0], marker=GROUP_MARKER[p], color="grey", linestyle="none",
                            markersize=8, label=p) for p in [NTC_LABEL, TF]]
cytokine_legend = [plt.Line2D([0], [0], marker="o", color=GROUP_COLOR[s], linestyle="none",
                               markersize=8, label=s) for s in ["PBS", "IL6", "IL10"]]
ax.legend(handles=group_legend + cytokine_legend, loc="upper left", bbox_to_anchor=(1.02, 1),
          fontsize=9, frameon=False)

plt.tight_layout()

dot_png = f"{out_dir}/{TF}_{GENE}_dotplot.png"
dot_svg = f"{out_dir}/{TF}_{GENE}_dotplot.svg"
plt.savefig(dot_png, dpi=300, bbox_inches="tight")
plt.savefig(dot_svg, bbox_inches="tight")
plt.show()

print(f"Saved: {dot_png}")
print("\nDone.")