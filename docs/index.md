---
layout: default
title: AMLAnnotator
---

# AMLAnnotator

**Hierarchical annotation of malignant cells, blast groups, and LSC types in paediatric AML single-cell RNA-seq data.**

AMLAnnotator is a Python tool that provides automated, three-level hierarchical classification of cells in acute myeloid leukaemia (AML) scRNA-seq datasets. It takes an AnnData object as input and returns cell-level annotations with confidence scores at each level of the hierarchy.

---

## Overview

AMLAnnotator performs three sequential classification steps:

| Level | Task | Classes |
|-------|------|---------|
| **1** | Malignant vs Normal | `malignant`, `normal` |
| **2** | Blast Group | `Primitive_blasts`, `GMP_like_blasts`, `Erythroid_like_blasts`, `Mature_myeloid_like_blasts`, `High_LSC_score_shared` |
| **3** | LSC Type | `P-LSC`, `M-LSC`, `E-LSC`, `Patient Specific` |

Each level uses an **ensemble of three classifiers** (Random Forest, Extra Trees, Logistic Regression) with majority voting and averaged probability scores. Only cells classified as malignant at Level 1 proceed to Level 2, and only cells classified as `High_LSC_score_shared` at Level 2 proceed to Level 3.

---

## Installation

```bash
pip install git+https://github.com/sinakane/AMLAnnotator.git
```

---

## Quick Start

```python
import scanpy as sc
from amlannotator import AMLAnnotator

# Load your AML scRNA-seq data
adata = sc.read_h5ad("my_aml_data.h5ad")

# Initialize the annotator (models load automatically)
annotator = AMLAnnotator()

# Annotate — specify the layer with raw counts if needed
adata = annotator.annotate(adata, layer="raw_counts")

# View results
print(adata.obs[["aml_malignant_normal", "aml_blast_group", "aml_lsc_type"]].value_counts())
```

---

## Output Columns

After annotation, the following columns are added to `adata.obs`:

### Annotations
- `aml_malignant_normal` — `'malignant'` or `'normal'`
- `aml_blast_group` — blast group label (malignant cells only; `'N/A'` for normal)
- `aml_lsc_type` — LSC type label (High_LSC_score_shared cells only; `'N/A'` otherwise)

### Confidence Scores
- `aml_malignant_confidence` — ensemble confidence for Level 1 (0–1)
- `aml_blast_confidence` — ensemble confidence for Level 2 (0–1)
- `aml_lsc_confidence` — ensemble confidence for Level 3 (0–1)

### Per-class Probabilities
- `aml_prob_malignant`, `aml_prob_normal`
- `aml_prob_Primitive_blasts`, `aml_prob_GMP_like_blasts`, etc.
- `aml_prob_P-LSC`, `aml_prob_M-LSC`, `aml_prob_E-LSC`, `aml_prob_Patient Specific`

---

## How It Works

### Preprocessing
The tool normalises your expression matrix internally: raw counts are total-count normalised to 10,000 per cell and log1p-transformed. No prior normalisation is required — provide raw counts.

### Feature Selection
- **Levels 1 & 2**: The top 500 genes were selected by **mutual information** between gene expression and the target labels from 3,000 highly variable genes.
- **Level 3**: The top 80 differentially expressed genes per class were selected using the **Wilcoxon rank-sum test** via `scanpy.tl.rank_genes_groups`, yielding 308 unique genes.

Only these selected genes are used at prediction time.

### Ensemble Classification
Each level uses three classifiers trained on the selected features:

1. **Random Forest** (300 trees, max depth 25, balanced class weights)
2. **Extra Trees** (300 trees, max depth 25, balanced class weights)
3. **Logistic Regression** (L2 regularisation, SAGA solver, balanced class weights)

The final prediction is determined by **majority vote** across the three classifiers. The confidence score is the **maximum of the averaged probability vector** across models.

### Hierarchical Flow

```
All cells
  │
  ├─ Level 1: Malignant vs Normal
  │     │
  │     ├─ normal → done
  │     │
  │     └─ malignant
  │           │
  │           ├─ Level 2: Blast Group
  │           │     │
  │           │     ├─ Primitive_blasts → done
  │           │     ├─ GMP_like_blasts → done
  │           │     ├─ Erythroid_like_blasts → done
  │           │     ├─ Mature_myeloid_like_blasts → done
  │           │     │
  │           │     └─ High_LSC_score_shared
  │           │           │
  │           │           └─ Level 3: LSC Type
  │           │                 ├─ P-LSC
  │           │                 ├─ M-LSC
  │           │                 ├─ E-LSC
  │           │                 └─ Patient Specific
```

---

## Training Data

Models were trained on a paediatric AML cohort:

- **Level 1**: 259,817 cells × 3,000 HVGs → 500 MI-selected features — **99.1% accuracy**
- **Level 2**: 160,295 malignant cells × 3,000 HVGs → 500 MI-selected features — **98.9% accuracy**
- **Level 3**: 19,659 High_LSC_score_shared cells × 36,601 genes → 308 DE genes — **96.6% accuracy**

All metrics are from a held-out 20% test set (stratified split).

---

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.21
- scikit-learn ≥ 1.0
- anndata ≥ 0.8
- scanpy ≥ 1.9
- scipy ≥ 1.7
- joblib ≥ 1.1

---

## Citation

If you use AMLAnnotator in your research, please cite:

> Kanannejad, S. et al. (2026). AMLAnnotator: Hierarchical annotation of malignant cells in paediatric AML single-cell data. GitHub. https://github.com/sinakane/AMLAnnotator

---

## License

MIT License. See [LICENSE](https://github.com/sinakane/AMLAnnotator/blob/main/LICENSE).
