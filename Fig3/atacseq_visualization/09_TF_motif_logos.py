"""
Fig3 ATAC-seq visualization, supplementary: sequence logos for the 4
TFs of interest (STAT3, TFEB, ETV3, ERF), pulled directly from the
H14CORE MEME-format motif file used for all of this project's FIMO
runs. One logo panel per TF, arranged in a single figure.

Logos use INFORMATION CONTENT scaling (Schneider & Stephens
convention, bits) -- the standard sequence-logo style, not raw
probability height.

Requires logomaker: pip install logomaker --break-system-packages
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

# ---- data sources ----
meme_f = "/ix/djishnu/Aaron_F/proteomics_proj/Results/DEG_motif_enrichment/H14CORE_meme_format.meme"
out_dir = "/ix/djishnu/Aaron_F/proteomics_proj/Results/Figures/Fig3/atacseq_visualization"
os.makedirs(out_dir, exist_ok=True)

# exact motif IDs, matching the FIMO runs used throughout this project
MOTIFS_OF_INTEREST = {
    "STAT3": "STAT3.H14CORE.0.P.B",
    "TFEB": "TFEB.H14CORE.0.SM.B",
    "ETV3": "ETV3.H14CORE.0.SM.B",
    "ERF": "ERF.H14CORE.0.PS.A",
}

TF_COLOR_SCHEME = "classic"  # logomaker's built-in DNA color scheme


def parse_meme_motifs(meme_path, motif_ids_wanted):
    """Parses a MEME-format motif file, returning
    {motif_id: probability_matrix (DataFrame, columns A C G T)} for
    just the requested motif IDs.

    Reads the matrix width directly from the header line (e.g.
    "w= 13") and takes EXACTLY that many subsequent lines as the
    matrix rows, rather than guessing where the block ends via blank
    lines -- this file's blocks are followed by a "URL ..." line
    BEFORE the blank line, which a blank-line/next-MOTIF lookahead
    would incorrectly swallow into the matrix rows."""
    with open(meme_path) as f:
        lines = f.readlines()

    found = {}
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.startswith("MOTIF "):
            motif_id = line.split()[1]

            j = i + 1
            while j < len(lines) and "letter-probability matrix" not in lines[j]:
                j += 1
            if j >= len(lines):
                break

            w_match = re.search(r"w=\s*(\d+)", lines[j])
            if w_match is None:
                i = j + 1
                continue
            w = int(w_match.group(1))

            if motif_id in motif_ids_wanted:
                rows = []
                for k in range(j + 1, j + 1 + w):
                    vals = [float(x) for x in lines[k].split()]
                    rows.append(vals)
                found[motif_id] = pd.DataFrame(rows, columns=["A", "C", "G", "T"])

            i = j + 1 + w
        else:
            i += 1

    return found


print(f"Parsing MEME file for motifs: {list(MOTIFS_OF_INTEREST.values())}")
motif_matrices = parse_meme_motifs(meme_f, set(MOTIFS_OF_INTEREST.values()))

missing = set(MOTIFS_OF_INTEREST.values()) - set(motif_matrices.keys())
if missing:
    print(f"WARNING: these motif IDs were not found in the MEME file: {missing}")

n_found = len(motif_matrices)
print(f"Found {n_found}/{len(MOTIFS_OF_INTEREST)} requested motifs")

# =============================================================================
# One logo panel per TF, information-content scaled
# =============================================================================
tf_names = [tf for tf, mid in MOTIFS_OF_INTEREST.items() if mid in motif_matrices]

fig, axes = plt.subplots(len(tf_names), 1, figsize=(6, 2.2 * len(tf_names)), squeeze=False)
axes = axes.flatten()

for i, tf in enumerate(tf_names):
    motif_id = MOTIFS_OF_INTEREST[tf]
    prob_df = motif_matrices[motif_id]

    # convert probabilities -> information content (bits), standard
    # sequence-logo scaling
    info_df = logomaker.transform_matrix(prob_df, from_type="probability", to_type="information")

    ax = axes[i]
    logomaker.Logo(info_df, ax=ax, color_scheme=TF_COLOR_SCHEME)
    ax.set_ylabel("bits", fontsize=9)
    ax.set_title(f"{tf}  ({motif_id})", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 2)

plt.tight_layout()

logo_png = f"{out_dir}/TF_motif_logos_STAT3_TFEB_ETV3_ERF.png"
logo_svg = f"{out_dir}/TF_motif_logos_STAT3_TFEB_ETV3_ERF.svg"
plt.savefig(logo_png, dpi=300, bbox_inches="tight")
plt.savefig(logo_svg, bbox_inches="tight")
plt.show()

print(f"\nSaved: {logo_png}")
print("\nDone.")