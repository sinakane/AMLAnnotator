"""
Core annotator class for hierarchical AML / B-ALL cell classification.
"""

import os
import shutil
import numpy as np
import joblib
from scipy.sparse import issparse


PACKAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Models are hosted on the Hugging Face Hub rather than GitHub Releases: the
# retrained "full" L1 and "pan_leukemia" L1 models both exceed GitHub's 2GB-
# per-release-asset limit (the largest single file is ~3GB), and would also
# exceed Git LFS's free bandwidth quota on a single clone. HF Hub has no such
# per-file cap. Each model bundle lives under its own prefix in the repo
# (e.g. "full/L1_random_forest.joblib", "pan_leukemia/L1_random_forest.joblib").
HF_REPO_ID = "AgSin/AMLAnnotator-models"

LEVEL_NAMES = {
    1: "Malignant vs Normal",
    2: "Blast Group",
    3: "LSC Type",
}

MODEL_TYPES = ["random_forest", "extra_trees", "logistic_regression"]

# ──────────────────────────────────────────────────────────────────────────
# Model registry: each entry is a distinct, independently-downloadable model
# bundle, selected via AMLAnnotator(model="<key>"). See AMLAnnotator.list_models().
#
#   "full"         -- the original three-level hierarchy (malignant/normal ->
#                      blast group -> LSC type), trained on AML only.
#   "pan_leukemia" -- malignant/normal only (Level 1) so far, trained across
#                      AML and B-ALL. Blast group and LSC type for this model
#                      are planned for a future release -- once trained, add
#                      their level numbers to "levels" below and upload the
#                      corresponding files under the same "pan_leukemia/"
#                      prefix on the Hub; no other code changes needed.
# ──────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "full": {
        "display_name": "AMLAnnotator Full",
        "description": "Three-level hierarchy: malignant/normal -> blast group -> LSC type.",
        "diseases": ["AML"],
        "levels": [1, 2, 3],
        "hf_prefix": "full",
    },
    "pan_leukemia": {
        "display_name": "AMLAnnotator Pan-Leukemia",
        "description": (
            "Malignant vs normal only (Level 1), trained across AML and B-ALL. "
            "Blast group and LSC type are not yet available for this model "
            "(planned for a future release)."
        ),
        "diseases": ["AML", "B-ALL"],
        "levels": [1],
        "hf_prefix": "pan_leukemia",
    },
}
DEFAULT_MODEL = "full"


def _expected_files(levels):
    return (
        [f"L{l}_{m}.joblib" for l in levels for m in MODEL_TYPES]
        + [f"L{l}_label_encoder.joblib" for l in levels]
        + [f"L{l}_genes.npy" for l in levels]
        + [f"L{l}_metrics.json" for l in levels]
    )


def _is_lfs_pointer(path):
    """Check if a file is a Git LFS pointer instead of actual content."""
    try:
        with open(path, "rb") as f:
            header = f.read(40)
        return header.startswith(b"version https://git-lfs")
    except Exception:
        return False


def _ensure_models(model_key):
    """Download one model bundle's files from the Hugging Face Hub if missing/corrupted."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY)}. "
            f"See AMLAnnotator.list_models() for details."
        )
    spec = MODEL_REGISTRY[model_key]
    model_dir = os.path.join(PACKAGE_MODEL_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)

    expected = _expected_files(spec["levels"])
    missing = [
        fname for fname in expected
        if not os.path.exists(os.path.join(model_dir, fname)) or _is_lfs_pointer(os.path.join(model_dir, fname))
    ]
    if not missing:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "Model files are missing and downloading them requires the 'huggingface_hub' "
            "package: pip install huggingface_hub"
        ) from e

    print(f"AMLAnnotator: downloading {len(missing)} file(s) for model='{model_key}' "
          f"from https://huggingface.co/{HF_REPO_ID} ({spec['hf_prefix']}/) ...", flush=True)
    try:
        for fname in missing:
            print(f"  {fname} ...", flush=True)
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID, filename=f"{spec['hf_prefix']}/{fname}"
            )
            shutil.copyfile(cached_path, os.path.join(model_dir, fname))
        print("  Models ready.", flush=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download models from Hugging Face Hub: {e}\n"
            f"You can browse/download manually from: "
            f"https://huggingface.co/{HF_REPO_ID}/tree/main/{spec['hf_prefix']}\n"
            f"Place the files directly into: {model_dir}"
        ) from e


class AMLAnnotator:
    """
    Hierarchical annotator for AML / B-ALL single-cell RNA-seq data.

    Two model bundles are available -- call AMLAnnotator.list_models() for
    the full, up-to-date list and what each one covers:

      - "full" (default): three-level AML-only hierarchy (malignant/normal
        -> blast group -> LSC type).
      - "pan_leukemia": malignant/normal only (Level 1 -- calling
        .annotate() will only populate the malignant/normal columns),
        trained across both AML and B-ALL.

    Each available level uses an ensemble of three classifiers (Random
    Forest, Extra Trees, Logistic Regression) with majority voting and
    averaged probability scores.

    Parameters
    ----------
    model : str, default "full"
        Which model bundle to load. See AMLAnnotator.list_models().

    Examples
    --------
    >>> import scanpy as sc
    >>> from amlannotator import AMLAnnotator
    >>> AMLAnnotator.list_models()
    >>> adata = sc.read_h5ad("my_data.h5ad")
    >>> annotator = AMLAnnotator(model="full")
    >>> adata = annotator.annotate(adata)
    >>> adata.obs[["aml_malignant_normal", "aml_blast_group", "aml_lsc_type"]]
    """

    def __init__(self, model=DEFAULT_MODEL):
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}. "
                f"See AMLAnnotator.list_models() for details."
            )
        self.model_key = model
        self.levels = MODEL_REGISTRY[model]["levels"]
        self._models = {}
        self._label_encoders = {}
        self._genes = {}
        self._load_models()

    @staticmethod
    def list_models():
        """Print the available model bundles, what diseases/levels each covers, and how to pick one."""
        print(f"{'model=':<20}{'diseases':<14}{'levels available'}")
        print("-" * 90)
        for key, spec in MODEL_REGISTRY.items():
            marker = '"%s" (default)' % key if key == DEFAULT_MODEL else '"%s"' % key
            levels_str = ", ".join(f"L{l} ({LEVEL_NAMES[l]})" for l in spec["levels"])
            print(f"{marker:<20}{'/'.join(spec['diseases']):<14}{levels_str}")
            print(f"    {spec['description']}\n")
        print('Usage: AMLAnnotator(model="<key>")')

    def _model_dir(self):
        return os.path.join(PACKAGE_MODEL_DIR, self.model_key)

    def _load_models(self):
        """Load pre-trained models, label encoders, and gene lists for this instance's model bundle."""
        _ensure_models(self.model_key)
        model_dir = self._model_dir()
        for level in self.levels:
            prefix = f"L{level}"
            self._models[level] = {}
            for mtype in MODEL_TYPES:
                path = os.path.join(model_dir, f"{prefix}_{mtype}.joblib")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Model file not found: {path}. "
                        "Ensure the package is installed correctly."
                    )
                self._models[level][mtype] = joblib.load(path)

            le_path = os.path.join(model_dir, f"{prefix}_label_encoder.joblib")
            self._label_encoders[level] = joblib.load(le_path)

            genes_path = os.path.join(model_dir, f"{prefix}_genes.npy")
            self._genes[level] = np.load(genes_path, allow_pickle=True)

    @staticmethod
    def _normalize(X_raw):
        """Normalize raw counts: total-count to 10k, log1p."""
        if issparse(X_raw):
            X_raw = X_raw.toarray()
        X = np.array(X_raw, dtype=np.float32)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums * 10000
        return np.log1p(X)

    def _get_feature_matrix(self, adata, level, layer=None):
        """Extract and normalize expression matrix for the required genes."""
        required_genes = self._genes[level]
        adata_genes = np.array(adata.var_names)

        # Find matching gene indices
        gene_idx = []
        missing_genes = []
        for gene in required_genes:
            matches = np.where(adata_genes == gene)[0]
            if len(matches) > 0:
                gene_idx.append(matches[0])
            else:
                missing_genes.append(gene)
                gene_idx.append(None)

        if missing_genes:
            n_missing = len(missing_genes)
            n_total = len(required_genes)
            pct = n_missing / n_total * 100
            if pct > 50:
                raise ValueError(
                    f"Level {level}: {n_missing}/{n_total} ({pct:.0f}%) required "
                    f"genes are missing from the input data. Check that your "
                    f"AnnData contains standard gene symbols."
                )
            import warnings
            warnings.warn(
                f"Level {level}: {n_missing}/{n_total} ({pct:.1f}%) required "
                f"genes not found. Missing genes will be set to zero."
            )

        # Extract expression
        if layer and layer in adata.layers:
            X_full = adata.layers[layer]
        else:
            X_full = adata.X

        if issparse(X_full):
            X_full = X_full.toarray()
        X_full = np.array(X_full, dtype=np.float32)

        # Normalize
        X_norm = self._normalize(X_full)

        # Select features
        X_sel = np.zeros((X_norm.shape[0], len(required_genes)), dtype=np.float32)
        for i, idx in enumerate(gene_idx):
            if idx is not None:
                X_sel[:, i] = X_norm[:, idx]

        return X_sel

    def _predict_level(self, X, level):
        """Run ensemble prediction for a given level."""
        le = self._label_encoders[level]
        n_classes = len(le.classes_)

        all_preds = []
        all_probas = []
        for mtype in MODEL_TYPES:
            clf = self._models[level][mtype]
            preds = clf.predict(X)
            probas = clf.predict_proba(X)
            all_preds.append(preds)
            all_probas.append(probas)

        # Majority vote
        pred_matrix = np.column_stack(all_preds)
        majority = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n_classes).argmax(), 1, pred_matrix
        )
        labels = le.inverse_transform(majority)

        # Average probability
        avg_proba = np.mean(all_probas, axis=0)
        confidence = np.max(avg_proba, axis=1)

        # Per-class probabilities
        class_probas = {cls: avg_proba[:, i] for i, cls in enumerate(le.classes_)}

        return labels, confidence, class_probas

    def annotate(self, adata, layer=None, copy=False, malignant_prob_threshold=0.3):
        """
        Annotate an AnnData object with hierarchical AML/B-ALL classification.

        Which columns get populated depends on the model bundle this
        instance was constructed with (see AMLAnnotator.list_models()) --
        a model that only supports Level 1 will only populate the
        malignant/normal columns.

        Parameters
        ----------
        adata : anndata.AnnData
            Input AnnData object with raw or normalized counts.
            Must contain standard gene symbols in var_names.
        layer : str, optional
            Layer containing raw counts. If None, uses adata.X.
            Common choices: 'raw_counts', 'counts'.
        copy : bool, default False
            If True, return a copy of the AnnData object.
        malignant_prob_threshold : float, default 0.3
            Cells initially classified as normal but with an averaged
            malignant probability above this threshold are reclassified
            as malignant. Set to 0.5 to disable reclassification.

        Returns
        -------
        adata : anndata.AnnData
            Annotated AnnData with new columns in obs:
            - ``aml_malignant_normal``: 'malignant' or 'normal'
            - ``aml_malignant_confidence``: ensemble confidence [0-1]
            - ``aml_blast_group``: blast group (malignant cells only; only
              if this model's levels include Level 2)
            - ``aml_blast_confidence``: ensemble confidence [0-1]
            - ``aml_lsc_type``: LSC type (High_LSC_score_shared cells only;
              only if this model's levels include Level 3)
            - ``aml_lsc_confidence``: ensemble confidence [0-1]

            Per-class probability columns are also added:
            - ``aml_prob_malignant``, ``aml_prob_normal``
            - ``aml_prob_<blast_group>`` for each blast group, if available
            - ``aml_prob_<lsc_type>`` for each LSC type, if available
        """
        if copy:
            adata = adata.copy()

        n_cells = adata.n_obs
        print(f"AMLAnnotator (model='{self.model_key}'): annotating {n_cells} cells...")

        # --- Level 1: Malignant vs Normal ---
        print("  Level 1: Malignant vs Normal...")
        X_l1 = self._get_feature_matrix(adata, level=1, layer=layer)
        l1_labels, l1_conf, l1_probas = self._predict_level(X_l1, level=1)

        # Reclassify low-confidence normals with high malignant probability
        mal_probs = l1_probas["malignant"]
        flip_mask = (l1_labels == "normal") & (mal_probs >= malignant_prob_threshold)
        n_flipped = np.sum(flip_mask)
        if n_flipped > 0:
            l1_labels[flip_mask] = "malignant"
            l1_conf[flip_mask] = mal_probs[flip_mask]
            print(f"    Reclassified {n_flipped} low-confidence normals as malignant "
                  f"(prob_malignant >= {malignant_prob_threshold})")

        adata.obs["aml_malignant_normal"] = l1_labels
        adata.obs["aml_malignant_confidence"] = l1_conf
        for cls, probs in l1_probas.items():
            adata.obs[f"aml_prob_{cls}"] = probs

        n_mal = np.sum(l1_labels == "malignant")
        n_norm = np.sum(l1_labels == "normal")
        print(f"    {n_mal} malignant, {n_norm} normal")

        if 2 not in self.levels:
            available = ", ".join(LEVEL_NAMES[l] for l in self.levels)
            print(f"  Level 2 & 3: not available for model='{self.model_key}' "
                  f"(this model only supports: {available}). "
                  f"Use model=\"full\" for blast group / LSC type classification.")
            print("  Done.")
            return adata

        # --- Level 2: Blast Group (malignant cells only) ---
        adata.obs["aml_blast_group"] = "N/A"
        adata.obs["aml_blast_confidence"] = np.nan
        l2_classes = self._label_encoders[2].classes_

        if n_mal > 0:
            print("  Level 2: Blast Group...")
            mal_mask = l1_labels == "malignant"
            mal_idx = np.where(mal_mask)[0]
            adata_mal = adata[mal_mask]

            X_l2 = self._get_feature_matrix(adata_mal, level=2, layer=layer)
            l2_labels, l2_conf, l2_probas = self._predict_level(X_l2, level=2)

            adata.obs.iloc[mal_idx, adata.obs.columns.get_loc("aml_blast_group")] = l2_labels
            adata.obs.iloc[mal_idx, adata.obs.columns.get_loc("aml_blast_confidence")] = l2_conf

            for cls in l2_classes:
                col = f"aml_prob_{cls}"
                adata.obs[col] = np.nan
                adata.obs.iloc[mal_idx, adata.obs.columns.get_loc(col)] = l2_probas[cls]

            for cls in l2_classes:
                print(f"    {cls}: {np.sum(l2_labels == cls)}")

            # --- Level 3: LSC Type (High_LSC_score_shared cells only) ---
            adata.obs["aml_lsc_type"] = "N/A"
            adata.obs["aml_lsc_confidence"] = np.nan

            lsc_mask_in_mal = l2_labels == "High_LSC_score_shared"
            n_lsc = np.sum(lsc_mask_in_mal)

            if n_lsc > 0:
                print("  Level 3: LSC Type...")
                lsc_global_idx = mal_idx[lsc_mask_in_mal]
                adata_lsc = adata[lsc_global_idx]

                X_l3 = self._get_feature_matrix(adata_lsc, level=3, layer=layer)
                l3_labels, l3_conf, l3_probas = self._predict_level(X_l3, level=3)

                adata.obs.iloc[lsc_global_idx, adata.obs.columns.get_loc("aml_lsc_type")] = l3_labels
                adata.obs.iloc[lsc_global_idx, adata.obs.columns.get_loc("aml_lsc_confidence")] = l3_conf

                l3_classes = self._label_encoders[3].classes_
                for cls in l3_classes:
                    col = f"aml_prob_{cls}"
                    adata.obs[col] = np.nan
                    adata.obs.iloc[lsc_global_idx, adata.obs.columns.get_loc(col)] = l3_probas[cls]

                for cls in l3_classes:
                    print(f"    {cls}: {np.sum(l3_labels == cls)}")
            else:
                print("  Level 3: No High_LSC_score_shared cells found, skipping.")
        else:
            print("  Level 2 & 3: No malignant cells found, skipping.")

        print("  Done.")
        return adata

    def get_model_info(self):
        """Return a summary of the loaded model bundle's levels and their training metrics."""
        import json
        info = {"model": self.model_key, "levels": {}}
        model_dir = self._model_dir()
        for level in self.levels:
            metrics_path = os.path.join(model_dir, f"L{level}_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics = {"accuracy": "N/A"}

            info["levels"][f"Level {level}: {LEVEL_NAMES[level]}"] = {
                "classes": list(self._label_encoders[level].classes_),
                "n_features": len(self._genes[level]),
                "models": MODEL_TYPES,
                "accuracy": metrics.get("accuracy", "N/A"),
            }
        return info
