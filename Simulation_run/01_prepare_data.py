import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

from STAIR.utils import set_seed


def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _to_positive_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x, axis=0, keepdims=True)
    scale = np.nanpercentile(x, 99, axis=0, keepdims=True)
    scale[scale <= 0] = 1.0
    x = x / scale
    x = np.clip(x, 0.0, None)
    x = x + 1e-4
    return x.astype(np.float32)


def _merge_truth_columns(adata_rna, truth_path: str) -> None:
    if not os.path.exists(truth_path):
        return
    adata_truth = ad.read_h5ad(truth_path, backed="r")
    cols = [c for c in ("ground_truth", "truth") if c in adata_truth.obs]
    for col in cols:
        adata_rna.obs[col] = adata_truth.obs[col].astype(str).values
    adata_truth.file.close()


def main():
    set_seed(42)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    result_dir = os.path.join(root_dir, "Simulation_result")
    os.makedirs(result_dir, exist_ok=True)

    slices = [f"Simulation{i}" for i in range(1, 6)]

    gene_sets = []
    for slice_name in slices:
        rna_file = os.path.join(data_dir, slice_name, "adata_RNA.h5ad")
        if not os.path.exists(rna_file):
            raise FileNotFoundError(f"Missing RNA file: {rna_file}")
        adata_rna = sc.read_h5ad(rna_file, backed="r")
        gene_sets.append(set(map(str, adata_rna.var_names)))
        adata_rna.file.close()

    common_genes = sorted(set.intersection(*gene_sets))
    if len(common_genes) == 0:
        raise ValueError("No common RNA genes across simulations.")

    adata_list = []
    summary_rows = []

    for order, slice_name in enumerate(slices):
        rna_file = os.path.join(data_dir, slice_name, "adata_RNA.h5ad")
        adt_file = os.path.join(data_dir, slice_name, "adata_ADT.h5ad")
        truth_file = os.path.join(data_dir, slice_name, "3d-OT.h5ad")

        if not os.path.exists(adt_file):
            raise FileNotFoundError(f"Missing ADT file: {adt_file}")

        adata_rna = sc.read_h5ad(rna_file)
        adata_adt = sc.read_h5ad(adt_file)

        common_obs = adata_rna.obs_names.intersection(adata_adt.obs_names)
        if len(common_obs) == 0:
            raise ValueError(f"No overlapping cells in {slice_name} between RNA and ADT.")

        adata_rna = adata_rna[common_obs, common_genes].copy()
        adata_adt = adata_adt[common_obs].copy()

        if "spatial" not in adata_rna.obsm:
            if "spatial" not in adata_adt.obsm:
                raise KeyError(f"No 'spatial' found in RNA or ADT for {slice_name}")
            adata_rna.obsm["spatial"] = np.asarray(adata_adt.obsm["spatial"])

        adt_feat = _to_dense(adata_adt.X)
        adt_feat = _to_positive_features(adt_feat)

        adata_rna.obs_names_make_unique()
        adata_rna.var_names_make_unique()
        adata_rna.obsm["ADT"] = adt_feat
        adata_rna.obs["batch"] = slice_name
        adata_rna.obs["slice_order"] = int(order)

        _merge_truth_columns(adata_rna, truth_file)

        adata_list.append(adata_rna)
        summary_rows.append(
            {
                "slice": slice_name,
                "n_cells": int(adata_rna.n_obs),
                "n_genes_common": int(adata_rna.n_vars),
                "adt_dim": int(adata_rna.obsm["ADT"].shape[1]),
            }
        )

    merged = ad.concat(adata_list, join="inner", merge="same")
    merged.obs["batch"] = merged.obs["batch"].astype("category")
    merged.obs["batch"] = merged.obs["batch"].cat.set_categories(slices)

    merged_file = os.path.join(result_dir, "simulation_merged.h5ad")
    processed_file = os.path.join(result_dir, "simulation_processed.h5ad")

    merged.write(merged_file)
    merged.write(processed_file)

    pd.DataFrame(summary_rows).to_csv(os.path.join(result_dir, "input_summary.csv"), index=False)

    print(f"Common genes across slices: {len(common_genes)}")
    print(f"Merged shape: cells={merged.n_obs}, genes={merged.n_vars}")
    print(f"Saved merged data to: {merged_file}")
    print(f"Initialized processed data to: {processed_file}")


if __name__ == "__main__":
    main()
