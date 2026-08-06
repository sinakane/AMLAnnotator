"""
Core annotator class for hierarchical AML cell classification.
"""

import os
import numpy as np
import joblib
from scipy.sparse import issparse


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

LEVEL_NAMES = {
    1: "Malignant vs Normal",
    2: "Blast Group",
    3: "LSC Type",
}

MODEL_TYPES = ["random_forest", "extra_trees", "logistic_regression"]


class AMLAnnotator:
    """
    Hierarchical annotator for AML single-cell RNA-seq data.

    Provides three-level classification:
      - Level 1: Malignant vs Normal
      - Level 2: Blast group among malignant cells
      - Level 3: LSC type among High_LSC_score_shared cells

    Each level uses an ensemble of three classifiers (Random Forest,
    Extra Trees, Logistic Regression) with majority voting and
    averaged probability scores.

    Parameters
    ----------
    None. Models are loaded automatically from the bundled model files.

    Examples
    --------
    >>> import scanpy as sc
    >>> from amlannotator import AMLAnnotator
    >>> adata = sc.read_h5ad("my_aml_data.h5ad")
    >>> annotator = AMLAnnotator()
    >>> adata = annotator.annotate(adata)
    >>> adata.obs[["aml_malignant_normal", "aml_blast_group", "aml_lsc_type"]]
    """

    def __init__(self):
        self._models = {}
        self._label_encoders = {}
        self._genes = {}
        self._load_models()

    def _load_models(self):
        """Load all pre-trained models, label encoders, and gene lists."""
        for level in [1, 2, 3]:
            prefix = f"L{level}"
            self._models[level] = {}
            for mtype in MODEL_TYPES:
                path = os.path.join(MODEL_DIR, f"{prefix}_{mtype}.joblib")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Model file not found: {path}. "
                        "Ensure the package is installed correctly."
                    )
                self._models[level][mtype] = joblib.load(path)

            le_path = os.path.join(MODEL_DIR, f"{prefix}_label_encoder.joblib")
            self._label_encoders[level] = joblib.load(le_path)

            genes_path = os.path.join(MODEL_DIR, f"{prefix}_genes.npy")
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

    def annotate(self, adata, layer=None, copy=False, malignant_prob_threshold=0.4):
        """
        Annotate an AnnData object with hierarchical AML classification.

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
        malignant_prob_threshold : float, default 0.4
            Cells initially classified as normal but with an averaged
            malignant probability above this threshold are reclassified
            as malignant. Set to 0.5 to disable reclassification.

        Returns
        -------
        adata : anndata.AnnData
            Annotated AnnData with new columns in obs:
            - ``aml_malignant_normal``: 'malignant' or 'normal'
            - ``aml_malignant_confidence``: ensemble confidence [0-1]
            - ``aml_blast_group``: blast group (malignant cells only)
            - ``aml_blast_confidence``: ensemble confidence [0-1]
            - ``aml_lsc_type``: LSC type (High_LSC_score_shared cells only)
            - ``aml_lsc_confidence``: ensemble confidence [0-1]

            Per-class probability columns are also added:
            - ``aml_prob_malignant``, ``aml_prob_normal``
            - ``aml_prob_<blast_group>`` for each blast group
            - ``aml_prob_<lsc_type>`` for each LSC type
        """
        if copy:
            adata = adata.copy()

        n_cells = adata.n_obs
        print(f"AMLAnnotator: annotating {n_cells} cells...")

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
        """Return a summary of the models and their training metrics."""
        import json
        info = {}
        for level in [1, 2, 3]:
            metrics_path = os.path.join(MODEL_DIR, f"L{level}_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics = {"accuracy": "N/A"}

            info[f"Level {level}: {LEVEL_NAMES[level]}"] = {
                "classes": list(self._label_encoders[level].classes_),
                "n_features": len(self._genes[level]),
                "models": MODEL_TYPES,
                "accuracy": metrics.get("accuracy", "N/A"),
            }
        return info
