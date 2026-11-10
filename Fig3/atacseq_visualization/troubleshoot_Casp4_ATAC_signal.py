"""
Minimal single-gene troubleshooting script -- Casp4 ONLY, ATAC signal
panel ONLY. For diagnosing why the signal track doesn't render in
Adobe Illustrator when placing the SVG.

Two candidate fixes applied together:
  1. clip_on=False on fill_between -- avoids a known matplotlib SVG
     clip-path rendering bug in Illustrator (clip paths technically
     present in the SVG, but Illustrator sometimes fails to apply them,
     rendering the fill as invisible).
  2. agg.path.chunksize = 0 -- disables AGG's default long-path
     chunking. For a signal track with many data points (16000bp
     window), AGG can split the path into chunks for rendering
     performance; this chunking is invisible in PNG but can produce
     broken/incomplete path data in SVG specifically -- matches the
     "fine on screen/PNG, missing in Illustrator" symptom exactly.

Run this FIRST before re-running the full 06 script, to confirm which
fix (or both) actually solves it.
"""

import os
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib
matplotlib.rcParams["agg.path.chunksize"] = 0  # candidate fix -- disable AGG path chunking
matplotlib.rcParams["path.simplify"] = False   # candidate fix -- the more likely actual cause:
                                                 # path simplification is what leads matplotlib's
                                                 # SVG backend to cache a path once in <defs> and
                                                 # reuse it via <use>, which Illustrator has
                                                 # long-standing bugs resolving. Both MUST be set
                                                 # before importing pyplot.
import matplotlib.pyplot as plt

# ---- data sources ----
bigwig_f = "/ix/djishnu/Aaron_F/Cleaned_Proteomics_Aaron/20240906/Data/ATAC-seq/trimmed_data/bigwig/UT_merged.bw"
promoters_bed_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_results/out_files/8000_TSS_8000_promoters_mm10.bed"

out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization/troubleshooting"
os.makedirs(out_dir, exist_ok=True)

GENE = "Casp4"
UPSTREAM = 8000
DOWNSTREAM = 8000

for f, label in [(bigwig_f, "bigwig"), (promoters_bed_f, "promoters BED")]:
    if not os.path.exists(f):
        raise SystemExit(f"Missing {label} file: {f}")

promoters_bed = pd.read_csv(
    promoters_bed_f, sep="\t", header=None,
    names=["chrom", "start", "end", "gene", "score", "strand"]
).set_index("gene")

if GENE not in promoters_bed.index:
    raise SystemExit(f"'{GENE}' not found in promoters BED.")

row = promoters_bed.loc[GENE]
chrom, bed_start, bed_end, strand = row["chrom"], int(row["start"]), int(row["end"]), row["strand"]
print(f"{GENE}: {chrom}:{bed_start}-{bed_end} ({strand})")

bw = pyBigWig.open(bigwig_f)
if chrom not in bw.chroms():
    raise SystemExit(f"Chrom '{chrom}' not found in bigwig.")

values = bw.values(chrom, bed_start, bed_end, numpy=True)
values = np.nan_to_num(values, nan=0.0)
genomic_positions = np.arange(bed_start, bed_end)
bw.close()

if strand == "+":
    tss_genomic = bed_start + UPSTREAM
    relative_positions = genomic_positions - tss_genomic
else:
    tss_genomic = bed_start + DOWNSTREAM
    relative_positions = tss_genomic - genomic_positions
    relative_positions = relative_positions[::-1]
    values = values[::-1]

print(f"Signal: {len(values)} points, range [{values.min():.3f}, {values.max():.3f}]")

# =============================================================================
# Single panel, both fixes applied
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 3))
ax.fill_between(relative_positions, values, color="#7B9EC9", alpha=0.85, linewidth=0, clip_on=False)
ax.axvline(0, color="#777777", linewidth=1.0, linestyle="--", zorder=3)
ax.set_xlim(-UPSTREAM, DOWNSTREAM)
ax.set_ylabel("ATAC signal")
ax.set_xlabel("Position relative to TSS (bp)")
ax.set_title(f"{GENE} -- ATAC signal (troubleshooting: clip_on=False + agg.path.chunksize=0)")

plt.tight_layout()

png_f = f"{out_dir}/{GENE}_troubleshoot.png"
svg_f = f"{out_dir}/{GENE}_troubleshoot.svg"
pdf_f = f"{out_dir}/{GENE}_troubleshoot.pdf"
plt.savefig(png_f, dpi=300, bbox_inches="tight")
plt.savefig(svg_f, bbox_inches="tight")
plt.savefig(pdf_f, bbox_inches="tight")  # PDF is often more reliable for Illustrator than SVG --
                                           # Illustrator's native format is PDF-based, so its PDF
                                           # import path is more mature than its SVG import path
plt.show()

print(f"\nSaved: {png_f}")
print(f"Saved: {svg_f}")
print(f"Saved: {pdf_f}  <- try placing THIS one in Illustrator first")
print("\nOpen the SVG in Illustrator and check if the fill now shows up.")
print("If it STILL doesn't show, next things to check:")
print("  1. Open the SVG in a text editor -- search for 'clip-path' and 'path d=' near the")
print("     top of the file. If the fill's <path> element has an empty or missing 'd' attribute,")
print("     that confirms path corruption (chunking) rather than a clip-path display issue.")
print("  2. Try File > Place in Illustrator vs. opening the SVG directly -- these can behave")
print("     differently depending on Illustrator version.")
print("  3. As a fallback, re-export at a coarser resolution (larger binSize in bamCoverage,")
print("     e.g. 50bp instead of 10bp) to reduce the point count substantially.")