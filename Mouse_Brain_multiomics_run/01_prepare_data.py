import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from STAIR.utils import set_seed


def _to_positive_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x, axis=0, keepdims=True)
    scale = np.nanpercentile(x, 99, axis=0, keepdims=True)
    scale[scale <= 0] = 1.0
    x = x / scale
    x = np.clip(x, 0.0, None)
    x = x + 1e-4
    return x.astype(np.float32)


def _align_by_obs_names(obs_src, matrix, target_obs):
    src_index = pd.Index(obs_src)
    pos = src_index.get_indexer(target_obs)
    if (pos < 0).any():
        miss = int((pos < 0).sum())
        raise ValueError(f"Failed to align by obs_names: missing {miss} cells in source matrix.")
    return np.asarray(matrix)[pos]


def main():
    # Hardcoded run profile for convenient manual tuning.
    random_seed = 42  # 随机种子。
    use_common_genes = True  # 是否仅保留四张切片共同基因。
    atac_obsm_priority = ["X_lsi", "X_pca"]  # ATAC 表征优先级，可改成 ["X_pca", "X_lsi"]。
    atac_shift_quantile = 0.0  # 列平移时使用的分位数，0.0 对应最小值。
    atac_scale_quantile = 99.0  # 列缩放时使用的分位数。
    spatial_dot_size = 2  # 输入空间图点大小。

    slices = [
        ("Mouse_Brain_ATAC", "Mouse_Brain_ATAC"),
        ("Mouse_Brain_H3K4me3", "Mouse_Brain_H3K4me3"),
        ("Mouse_Brain_H3K27ac", "Mouse_Brain_H3K27ac"),
        ("Mouse_Brain_H3K27me3", "Mouse_Brain_H3K27me3"),
    ]

    set_seed(random_seed)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    result_dir = os.path.join(root_dir, "Mouse_Brain_multiomics_result")
    os.makedirs(result_dir, exist_ok=True)

    # 1) gather common RNA genes across four slices
    gene_sets = []
    for slice_name, folder in slices:
        rna_path = os.path.join(data_dir, folder, "adata_RNA.h5ad")
        if not os.path.exists(rna_path):
            raise FileNotFoundError(f"Missing RNA file: {rna_path}")
        adata_rna = sc.read_h5ad(rna_path, backed="r")
        gene_sets.append(set(map(str, adata_rna.var_names)))
        adata_rna.file.close()

    if use_common_genes:
        common_genes = sorted(set.intersection(*gene_sets))
        if len(common_genes) == 0:
            raise ValueError("No common RNA genes found across four slices.")
    else:
        common_genes = None

    adata_list = []
    stats = []

    # 2) build per-slice paired multi-omics object (RNA in X, ATAC-like features in obsm['ATAC'])
    for order, (slice_name, folder) in enumerate(slices):
        rna_path = os.path.join(data_dir, folder, "adata_RNA.h5ad")
        peak_path = os.path.join(data_dir, folder, "adata_peaks_normalized.h5ad")
        if not os.path.exists(peak_path):
            raise FileNotFoundError(f"Missing peak file: {peak_path}")

        adata_rna = sc.read_h5ad(rna_path)
        adata_peak = ad.read_h5ad(peak_path, backed="r")

        common_obs = adata_rna.obs_names.intersection(adata_peak.obs_names)
        if len(common_obs) == 0:
            adata_peak.file.close()
            raise ValueError(f"No overlapping cells between RNA and peaks in {slice_name}.")

        if common_genes is not None:
            adata_rna = adata_rna[common_obs, common_genes].copy()
        else:
            adata_rna = adata_rna[common_obs].copy()

        atac_raw = None
        atac_key_used = None
        for rep_key in atac_obsm_priority:
            if rep_key in adata_peak.obsm:
                atac_raw = adata_peak.obsm[rep_key]
                atac_key_used = rep_key
                break

        if atac_raw is None:
            adata_peak.file.close()
            raise KeyError(
                f"None of {atac_obsm_priority} found in peaks obsm for {slice_name}."
            )

        atac_raw = _align_by_obs_names(adata_peak.obs_names, atac_raw, adata_rna.obs_names)
        atac = np.asarray(atac_raw, dtype=np.float32)
        atac = atac - np.nanpercentile(atac, atac_shift_quantile, axis=0, keepdims=True)
        scale = np.nanpercentile(atac, atac_scale_quantile, axis=0, keepdims=True)
        scale[scale <= 0] = 1.0
        atac = np.clip(atac / scale, 0.0, None) + 1e-4
        atac = atac.astype(np.float32)

        adata_rna.obsm["ATAC"] = atac
        adata_rna.obs["batch"] = slice_name
        adata_rna.obs["slice_name"] = slice_name
        adata_rna.obs["slice_order"] = order

        if "spatial" not in adata_rna.obsm and "spatial" in adata_peak.obsm:
            sp = _align_by_obs_names(adata_peak.obs_names, adata_peak.obsm["spatial"], adata_rna.obs_names)
            adata_rna.obsm["spatial"] = np.asarray(sp)

        adata_list.append(adata_rna)
        stats.append(
            {
                "slice": slice_name,
                "n_cells": int(adata_rna.n_obs),
                "n_genes_common": int(adata_rna.n_vars),
                "atac_dim": int(atac.shape[1]),
                "atac_source": atac_key_used,
            }
        )

        adata_peak.file.close()

    merged = ad.concat(adata_list, join="inner", merge="same")
    merged.obs["batch"] = merged.obs["batch"].astype("category")
    merged.obs["batch"] = merged.obs["batch"].cat.set_categories([x[0] for x in slices])

    merged_file = os.path.join(result_dir, "multiomics_merged.h5ad")
    processed_file = os.path.join(result_dir, "multiomics_processed.h5ad")
    merged.write(merged_file)
    merged.write(processed_file)

    pd.DataFrame(stats).to_csv(os.path.join(result_dir, "input_summary.csv"), index=False)

    if "spatial" in merged.obsm:
        plt.figure(figsize=(6.0, 5.0))
        sc.pl.embedding(
            merged,
            basis="spatial",
            color="batch",
            frameon=False,
            s=spatial_dot_size,
            show=False,
            title="Mouse brain multiomics slices",
        )
        plt.savefig(os.path.join(result_dir, "input_spatial.png"), dpi=300, bbox_inches="tight")
        plt.close()

    if common_genes is None:
        print("Common genes across four slices: disabled (using per-slice full genes)")
    else:
        print(f"Common genes across four slices: {len(common_genes)}")
    print(f"Merged data saved to: {merged_file}")
    print(f"Initialized processed data: {processed_file}")


if __name__ == "__main__":
    main()
