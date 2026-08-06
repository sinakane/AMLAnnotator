# AMLAnnotator

**Hierarchical annotation of malignant cells, blast groups, and LSC types in paediatric AML single-cell RNA-seq data.**

AMLAnnotator provides automated, three-level classification of cells in AML scRNA-seq datasets using an ensemble of pre-trained classifiers.

## Classification Hierarchy

```
All cells
  ├─ Level 1: Malignant vs Normal
  │     ├─ normal → done
  │     └─ malignant
  │           ├─ Level 2: Blast Group
  │           │     ├─ Primitive_blasts
  │           │     ├─ GMP_like_blasts
  │           │     ├─ Erythroid_like_blasts
  │           │     ├─ Mature_myeloid_like_blasts
  │           │     └─ High_LSC_score_shared
  │           │           └─ Level 3: LSC Type
  │           │                 ├─ P-LSC
  │           │                 ├─ M-LSC
  │           │                 ├─ E-LSC
  │           │                 └─ Patient Specific
```

## Installation

```bash
pip install git+https://github.com/sinakane/AMLAnnotator.git
```

## Quick Start

```python
import scanpy as sc
from amlannotator import AMLAnnotator

adata = sc.read_h5ad("my_aml_data.h5ad")
annotator = AMLAnnotator()
adata = annotator.annotate(adata, layer="raw_counts")

# Results are in adata.obs
print(adata.obs[["aml_malignant_normal", "aml_blast_group", "aml_lsc_type"]])
```

## Output

| Column | Description |
|--------|-------------|
| `aml_malignant_normal` | `malignant` or `normal` |
| `aml_malignant_confidence` | Ensemble confidence (0–1) |
| `aml_blast_group` | Blast group (malignant only) |
| `aml_blast_confidence` | Ensemble confidence (0–1) |
| `aml_lsc_type` | LSC type (High_LSC_score_shared only) |
| `aml_lsc_confidence` | Ensemble confidence (0–1) |
| `aml_prob_*` | Per-class probabilities |

## Method

Each classification level uses an **ensemble of three classifiers**:
- Random Forest (300 trees)
- Extra Trees (300 trees)
- Logistic Regression (L2, SAGA)

Final labels are determined by **majority vote**. Confidence is the maximum of the **averaged probability** across models.

### Feature Selection

- **Levels 1 & 2**: Mutual information on 3,000 highly variable genes → top 500 genes per level
- **Level 3**: Wilcoxon rank-sum test via `scanpy.tl.rank_genes_groups` → top 80 DE genes per class (308 unique genes)

### Performance (held-out 20% test set)

| Level | Task | Classes | Cells | Ensemble Accuracy |
|-------|------|---------|-------|-------------------|
| L1 | Malignant vs Normal | 2 | 259,817 | 99.1% |
| L2 | Blast Group | 5 | 160,295 | 98.9% |
| L3 | LSC Type | 4 | 19,659 | 96.6% |

## Documentation

Full documentation and tutorials at: https://sinakane.github.io/AMLAnnotator

## Requirements

- Python >= 3.8
- scanpy >= 1.9
- scikit-learn >= 1.0
- anndata >= 0.8

## License

MIT
