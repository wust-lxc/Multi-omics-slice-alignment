import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from STAIR.utils import set_seed


def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _to_positive_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.nanmin(x, axis=0, keepdims=True)
    scale = np.nanpercentile(x, 99, axis=0, keepdims=True)
    scale[scale <= 0] = 1.0
    x = x / scale
    x = np.clip(x, 0.0, None)
    x = x + 1e-4
    return x.astype(np.float32)


def main():
    set_seed(42)

    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "Mouse_Brain_ATAC"
    result_dir = root_dir / "Mouse_brain_result" / "ATAC"
    result_dir.mkdir(parents=True, exist_ok=True)

    rna_file = data_dir / "adata_RNA.h5ad"
    peak_file = data_dir / "adata_peaks_normalized.h5ad"
    truth_file = data_dir / "3d-OT.h5ad"
    if not rna_file.exists():
        raise FileNotFoundError(f"Missing RNA file: {rna_file}")
    if not peak_file.exists():
        raise FileNotFoundError(f"Missing ATAC peak file: {peak_file}")

    adata_rna = sc.read_h5ad(rna_file)
    adata_peak = sc.read_h5ad(peak_file)

    common_obs = adata_rna.obs_names.intersection(adata_peak.obs_names)
    if len(common_obs) == 0:
        raise ValueError("No overlapping barcodes between RNA and ATAC peak AnnData.")

    adata_rna = adata_rna[common_obs].copy()
    adata_peak = adata_peak[common_obs].copy()

    if "spatial" not in adata_rna.obsm:
        if "spatial" not in adata_peak.obsm:
            raise KeyError("No spatial coordinates found in RNA or ATAC AnnData.")
        adata_rna.obsm["spatial"] = np.asarray(adata_peak.obsm["spatial"])

    if "X_lsi" in adata_peak.obsm:
        epi_feat = np.asarray(adata_peak.obsm["X_lsi"], dtype=np.float32)
        epi_source = "X_lsi"
    elif "X_pca" in adata_peak.obsm:
        epi_feat = np.asarray(adata_peak.obsm["X_pca"], dtype=np.float32)
        epi_source = "X_pca"
    else:
        epi_feat = _to_dense(adata_peak.X)
        epi_source = "X"
    epi_feat = _to_positive_features(epi_feat)
    epi_transform = "positive_99pct"
    adata_rna.obsm["EPI"] = epi_feat
    adata_rna.uns["EPI_source"] = epi_source
    adata_rna.uns["EPI_transform"] = epi_transform

    adata_rna.obs["original_barcode"] = adata_rna.obs_names.astype(str)
    adata_rna.obs["batch"] = "ATAC"
    adata_rna.obs["slice_order"] = 0
    adata_rna.obs["epigenomic_assay"] = "ATAC"
    if "ATAC_clusters" in adata_rna.obs:
        adata_rna.obs["Domain_input"] = adata_rna.obs["ATAC_clusters"].astype(str)
    if truth_file.exists():
        adata_truth = sc.read_h5ad(truth_file, backed="r")
        common_truth_obs = adata_rna.obs_names.intersection(adata_truth.obs_names)
        if "truth" in adata_truth.obs and len(common_truth_obs) > 0:
            truth_map = adata_truth.obs.loc[common_truth_obs, "truth"].astype(str).to_dict()
            adata_rna.obs["truth"] = adata_rna.obs_names.astype(str).map(truth_map)
        adata_truth.file.close()

    adata_rna.obs_names_make_unique()
    adata_rna.var_names_make_unique()

    processed_file = result_dir / "atac_processed.h5ad"
    adata_rna.write(processed_file)

    pd.DataFrame(
        [
            {
                "slice": "ATAC",
                "n_cells": int(adata_rna.n_obs),
                "n_genes": int(adata_rna.n_vars),
                "epi_dim": int(adata_rna.obsm["EPI"].shape[1]),
                "epi_source": epi_source,
                "epi_transform": epi_transform,
            }
        ]
    ).to_csv(result_dir / "input_summary.csv", index=False)

    print(f"Prepared ATAC single-slice data: cells={adata_rna.n_obs}, genes={adata_rna.n_vars}")
    print(f"Epigenomic feature source: {epi_source}, transform={epi_transform}, dim={adata_rna.obsm['EPI'].shape[1]}")
    print(f"Saved processed data to: {processed_file}")


if __name__ == "__main__":
    main()
