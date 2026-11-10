# Generation of `string_DES_gene_pairwise_interactions.tsv`

**Generated:** 2026-05-05 11:39

**Output file:** `/ix/djishnu/Aaron_F/proteomics_proj/Results/phosphoNetwork_res/out_files/string_DES_gene_pairwise_interactions.tsv`

This file was **not** produced by a script — it was pulled manually from the [STRING database](https://string-db.org) web interface. This document records the exact steps taken so the process can be reproduced.

## Steps

1. Started from the differentially phosphorylated site table:
   `/ix/djishnu/Aaron_F/proteomics_proj/Data/phosphoproteomics/significant_phospho_pairwise_res.csv`
2. Mapped each site `id` (format `geneSymbol_uniprotID_siteInfo`, e.g. `Ahnak_E9Q616_S136s_1_1_136_136`) to its gene symbol by splitting on `_` and taking the first token.
3. Took the list of unique gene symbols from the DE sites.
4. Went to [string-db.org](https://string-db.org) and used **Multiple proteins** search, pasting in the gene symbol list as a CSV, with organism set to *Mus musculus* (taxon 10090 — matches the `10090.ENSMUSP...` prefixes in `node1_string_id`/`node2_string_id` in the output).
5. In the resulting network's **Settings** panel:
   - Set **minimum required interaction score** to **0.4** (medium confidence).
   - Under **evidence considerations**, restricted to **Textmining**, **Experiments**, and **Databases** only.
6. Exported the network edge table from the **Exports** tab as a TSV (simple tabular text output) — one row per interacting gene pair, with per-evidence-type scores and a `combined_score` column.
7. Saved the downloaded TSV directly to:
   `Results/phosphoNetwork_res/out_files/string_DES_gene_pairwise_interactions.tsv`

No further processing was applied to the downloaded file — it's used as-is by the downstream notebooks (`string_based_phosphoNetwork*.ipynb`, `tf_kinase_analysis.ipynb`, `fig1c_umap_only.ipynb`), which read it via:

```python
string_net = pd.read_csv(string_net_f, sep="\t")
```
