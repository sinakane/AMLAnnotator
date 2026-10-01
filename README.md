# AMLAnnotator

**Hierarchical annotation of malignant cells, blast groups, and LSC types in paediatric AML (and, for malignant/normal calling, B-ALL) single-cell RNA-seq data.**

AMLAnnotator provides automated classification of cells in AML/B-ALL scRNA-seq datasets using an ensemble of pre-trained classifiers. Two separate model bundles are available, selected via `AMLAnnotator(model="...")`:

| `model=` | Diseases | Levels available | Notes |
|----------|----------|-------------------|-------|
| `"full"` (default) | AML | L1 malignant/normal → L2 blast group → L3 LSC type | The original three-level hierarchy. |
| `"pan_leukemia"` | AML + B-ALL | L1 malignant/normal only | Blast group and LSC type for this track are planned for a future release. |

Call `AMLAnnotator.list_models()` at any time to see this list without downloading anything.

## Classification Hierarchy (`model="full"`)

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

`model="pan_leukemia"` currently stops after Level 1.

## Installation

```bash
pip install git+https://github.com/sinakane/AMLAnnotator.git
```

## Quick Start

```python
import scanpy as sc
from amlannotator import AMLAnnotator

# See what's available before picking one
AMLAnnotator.list_models()

adata = sc.read_h5ad("my_data.h5ad")

# Full AML hierarchy (malignant/normal -> blast group -> LSC type)
annotator = AMLAnnotator(model="full")
adata = annotator.annotate(adata, layer="raw_counts")
print(adata.obs[["aml_malignant_normal", "aml_blast_group", "aml_lsc_type"]])

# Malignant/normal only, works on AML or B-ALL data
annotator = AMLAnnotator(model="pan_leukemia")
adata = annotator.annotate(adata, layer="raw_counts")
print(adata.obs[["aml_malignant_normal"]])
```

## Output

Which columns are populated depends on which levels the selected `model=` supports.

| Column | Description | Available for |
|--------|-------------|----------------|
| `aml_malignant_normal` | `malignant` or `normal` | both |
| `aml_malignant_confidence` | Ensemble confidence (0–1) | both |
| `aml_blast_group` | Blast group (malignant cells only) | `full` only |
| `aml_blast_confidence` | Ensemble confidence (0–1) | `full` only |
| `aml_lsc_type` | LSC type (High_LSC_score_shared cells only) | `full` only |
| `aml_lsc_confidence` | Ensemble confidence (0–1) | `full` only |
| `aml_prob_*` | Per-class probabilities | per available level |

## Method

Each available level uses an **ensemble of three classifiers**:
- Random Forest (300 trees)
- Extra Trees (300 trees)
- Logistic Regression (L2, SAGA)

Final labels are determined by **majority vote**. Confidence is the maximum of the **averaged probability** across models.

### Feature Selection

- **`full`, Levels 1 & 2**: Mutual information on highly variable genes → top 500 genes per level
- **`full`, Level 3**: Wilcoxon rank-sum test via `scanpy.tl.rank_genes_groups` → top 80 DE genes per class (308 unique genes)
- **`pan_leukemia`, Level 1**: Mutual information on precomputed highly variable genes (from the source object) → top 500 genes

### Performance (held-out 20% test set)

| Model | Level | Task | Classes | Cells | Ensemble Accuracy |
|-------|-------|------|---------|-------|-------------------|
| `full` | L1 | Malignant vs Normal | 2 | 259,817 | 99.1% |
| `full` | L2 | Blast Group | 5 | 160,295 | 98.9% |
| `full` | L3 | LSC Type | 4 | 19,659 | 96.6% |
| `pan_leukemia` | L1 | Malignant vs Normal | 2 | 1,744,555 | 95.6% |

`pan_leukemia`'s L1 is trained on a much larger, multi-study cohort spanning
both AML and B-ALL (vs. `full`'s AML-only, single-cohort L1) — the lower
accuracy reflects the harder, more heterogeneous task, not a worse model for
either disease individually.

## Model hosting

Model files are too large for GitHub (the largest single file exceeds
GitHub's 2GB release-asset limit) and are hosted on the
[Hugging Face Hub](https://huggingface.co/AgSin/AMLAnnotator-models)
instead, one subfolder per `model=` bundle (`full/`, `pan_leukemia/`).
Only the files for the bundle you actually request are **downloaded
automatically** the first time you instantiate `AMLAnnotator(model=...)`
— no manual steps needed, just requires an internet connection on first
use per bundle (cached afterwards).

## Documentation

Full documentation and tutorials at: https://sinakane.github.io/AMLAnnotator

## Requirements

- Python >= 3.8
- scanpy >= 1.9
- scikit-learn >= 1.0
- anndata >= 0.8
- huggingface_hub >= 0.20 (for automatic model download)

## License

MIT
