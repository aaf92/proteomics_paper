"""
get_expressed_proteins.py

For each macrophage type in Qi2022, identify expressed proteins:
  - detected in >= 2/3 replicates within that cell type
  - saves one gene symbol list per cell type
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
EXCEL_PATH = '/ix/djishnu/Aaron_F/proteomics_proj/Data/macs_proteomics/Qi2022_macrophage_MSdata.xlsx'
OUTPUT_DIR = Path('/ix/djishnu/Aaron_F/proteomics_proj/Data/macs_proteomics/expressed_proteins')

GENE_COL   = 'Symbol'
MIN_REPS   = 2   # require detection in at least this many of 3 replicates

# ── Macrophage types → their 3 replicate columns ───────────────────────────────
CELL_TYPES = {
    'Microglia':            ['Microglia_1',            'Microglia_2',            'Microglia_3'],
    'Alveolar_M':          ['Alveolar MΦ_1',          'Alveolar MΦ_2',          'Alveolar MΦ_3'],
    'Lung_recruited_M':    ['Lung recruited MΦ_1',    'Lung recruited MΦ_2',    'Lung recruited MΦ_3'],
    'Kupffer_Cells':        ['Kupffer Cells_1',        'Kupffer Cells_2',        'Kupffer Cells_3'],
    'Liver_recruited_M':   ['Liver recruited MΦ_1',   'Liver recruited MΦ_2',   'Liver recruited MΦ_3'],
    'Red_pulp_M':          ['Red pulp MΦ_1',          'Red pulp MΦ_2',          'Red pulp MΦ_3'],
    'Spleen_recruited_M':  ['Spleen recruited MΦ_1',  'Spleen recruited MΦ_2',  'Spleen recruited MΦ_3'],
    'Small_Intestinal_M':  ['Small Intestinal MΦ_1',  'Small Intestinal MΦ_2',  'Small Intestinal MΦ_3'],
    'Large_Intestinal_M':  ['Large Intestinal MΦ_1',  'Large Intestinal MΦ_2',  'Large Intestinal MΦ_3'],
    'Peritoneal_M':        ['Peritoneal MΦ_1',        'Peritoneal MΦ_2',        'Peritoneal MΦ_3'],
    'BMDM':                 ['BMDM_1',                 'BMDM_2',                 'BMDM_3'],
    'RAW264.7':             ['RAW264.7_1',              'RAW264.7_2',              'RAW264.7_3'],
}

# ── Load ───────────────────────────────────────────────────────────────────────
print(f'Loading {EXCEL_PATH} ...')
df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[GENE_COL]).reset_index(drop=True)
print(f'Proteins with gene symbol: {len(df)}\n')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Process each cell type ─────────────────────────────────────────────────────
summary = []

for cell_type, rep_cols in CELL_TYPES.items():
    n_detected = df[rep_cols].notna().sum(axis=1)
    mask = n_detected >= MIN_REPS
    df_expressed = df.loc[mask, GENE_COL].astype(str)

    # flatten semicolon-delimited multi-gene entries
    gene_symbols = sorted(set(
        g.strip()
        for entry in df_expressed
        for g in entry.split(';')
        if g.strip() and g.strip() != 'nan'
    ))

    out_path = OUTPUT_DIR / f'{cell_type}_expressed_protein_gene_symbols.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(gene_symbols))

    summary.append((cell_type, mask.sum(), len(gene_symbols)))
    print(f'{cell_type:<25}  {mask.sum():>5} proteins  →  {len(gene_symbols):>5} gene symbols')

# ── Summary table ──────────────────────────────────────────────────────────────
print(f'\nAll files saved to: {OUTPUT_DIR}')
print(f'\n{"Cell type":<25}  {"Proteins":>8}  {"Gene symbols":>13}')
print('-' * 50)
for cell_type, n_prot, n_genes in summary:
    print(f'{cell_type:<25}  {n_prot:>8}  {n_genes:>13}')