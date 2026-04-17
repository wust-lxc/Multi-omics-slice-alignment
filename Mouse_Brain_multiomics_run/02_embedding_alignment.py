import warnings
warnings.filterwarnings("ignore")

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import math
from sklearn.neighbors import NearestNeighbors

from STAIR.multi_emb_alignment import Multi_Emb_Align
from STAIR.utils import set_seed, cluster_func


def _moran_i_knn(values: np.ndarray, coords: np.ndarray, k: int = 10) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return np.nan

    k_use = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(coords)
    neigh_idx = nbrs.kneighbors(coords, return_distance=False)[:, 1:]

    z = values - values.mean()
    denom = np.sum(z * z)
    if denom <= 0:
        return np.nan

    # Row-standardized KNN weights => each row sums to 1, therefore S0 = n.
    w = 1.0 / float(k_use)
    num = np.sum(z[:, None] * z[neigh_idx] * w)
    s0 = float(n)
    return float((n / s0) * (num / denom))


def _compute_slice_moran_curve(
    adata,
    domain_key: str = "Domain",
    spatial_key: str = "spatial",
    slice_key: str = "batch",
    k_list=None,
) -> pd.DataFrame:
    if k_list is None:
        k_list = [6, 10, 15, 20, 30]

    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key} not found in adata.obsm")
    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key} not found in adata.obs")
    if slice_key not in adata.obs:
        raise KeyError(f"{slice_key} not found in adata.obs")

    rows = []
    for slice_name in sorted(adata.obs[slice_key].astype(str).unique()):
        adata_tmp = adata[adata.obs[slice_key].astype(str) == slice_name].copy()
        coords = adata_tmp.obsm[spatial_key]
        domains = adata_tmp.obs[domain_key].astype(str)
        codes = pd.Categorical(domains).codes.astype(np.float64)

        for k in k_list:
            rows.append(
                {
                    "slice": str(slice_name),
                    "k": int(k),
                    "moran_i": _moran_i_knn(codes, coords, k=int(k)),
                    "n_cells": int(adata_tmp.n_obs),
                }
            )

    return pd.DataFrame(rows)


def main():
    set_seed(42)

    # Hardcoded run profile (high-precision version) for convenient repeated runs.
    hvg_top = 4000  # 用于建模的高变基因数量。
    ae_epoch = 200  # 多组学自编码器预训练轮数。
    ae_batch_size = 256  # 自编码器训练时的批大小。
    loss_weight_rna = 1.0  # RNA 重构损失权重。
    loss_weight_atac = 5.0  # ATAC 重构损失权重。
    atac_loss = "mse"  # ATAC 损失类型：mse(LSI/PCA连续特征推荐) 或 nb/zinb(原始计数推荐)。

    hgat_epoch = 200  # HGAT 跨切片对齐训练轮数。
    hgat_batches = 10  # HGAT 训练/预测分块数。
    sim_threshold = 0.3  # 构建跨切片连接时的相似度阈值。
    c_neigh_het = 0.9  # 跨切片异质邻居的相似度阈值参数。
    n_neigh_hom = 10  # 同切片同质邻居数量。
    mini_batch = False  # 是否启用 HGAT 小批量模式。

    max_cells_per_slice = 0  # 每张切片最多保留的细胞数；0 表示不过滤。
    cluster_num = 18  # mclust 目标簇数（需安装 R 包 mclust 与 rpy2）。
    moran_k_list = [6]  # 分切片 Moran 的固定 K 列表。

    # Reuse controls to avoid retraining from scratch on small script changes.
    force_retrain_ae = False  # True 时强制重训 AE；False 时优先复用缓存。
    force_retrain_hgat = False  # True 时强制重训 HGAT；False 时优先复用缓存。

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Mouse_Brain_multiomics_result")
    embedding_dir = os.path.join(result_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)

    merged_file = os.path.join(result_dir, "multiomics_merged.h5ad")
    processed_file = os.path.join(result_dir, "multiomics_processed.h5ad")
    ae_cache_file = os.path.join(embedding_dir, "adata_after_ae.h5ad")
    hgat_cache_file = os.path.join(embedding_dir, "adata_after_hgat.h5ad")

    if not os.path.exists(merged_file):
        raise FileNotFoundError("multiomics_merged.h5ad not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(merged_file)
    adata.obs_names_make_unique()

    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )
    print("Detected slice order:", batch_order)

    hvg_top = min(hvg_top, adata.n_vars)

    if max_cells_per_slice > 0:
        kept = []
        for batch_name in batch_order:
            idx = (adata.obs["batch"].astype(str).values == batch_name).nonzero()[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_cells_per_slice:
                idx = np.random.choice(idx, size=max_cells_per_slice, replace=False)
            kept.append(idx)
        if kept:
            keep_idx = np.concatenate(kept)
            adata = adata[keep_idx].copy()
            print(
                f"Subsampled for memory: max {max_cells_per_slice} cells/slice, "
                f"new shape={adata.n_obs}x{adata.n_vars}"
            )
    reuse_ae = (not force_retrain_ae) and os.path.exists(ae_cache_file)
    if reuse_ae:
        adata_ae = sc.read_h5ad(ae_cache_file)
        if "latent" in adata_ae.obsm:
            adata = adata_ae
            print(f"Reusing cached AE latent from: {ae_cache_file}")
        else:
            reuse_ae = False

    if not reuse_ae:
        emb_align = Multi_Emb_Align(
            adata,
            batch_key="batch",
            hvg=hvg_top,
            n_hidden=128,
            n_latent=32,
            likelihood="nb",
            num_workers=0,
            result_path=result_dir,
            atac_key="ATAC",
        )

        emb_align.prepare(count_key=None, lib_size="explog", normalize=True, scale=False)
        emb_align.preprocess(
            epoch_ae=ae_epoch,
            batch_size=ae_batch_size,
            loss_weight_rna=loss_weight_rna,
            loss_weight_atac=loss_weight_atac,
            atac_loss=atac_loss,
        )
        emb_align.latent()
        adata = emb_align.adata
        adata.write(ae_cache_file)
        print(f"Saved AE cache to: {ae_cache_file}")

    # Rebuild slice order from current adata after optional cache loading.
    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )

    attention_file = os.path.join(embedding_dir, "attention.csv")
    reuse_hgat = (not force_retrain_hgat) and os.path.exists(hgat_cache_file) and os.path.exists(attention_file)
    if reuse_hgat:
        adata_hgat = sc.read_h5ad(hgat_cache_file)
        if "STAIR" in adata_hgat.obsm:
            adata = adata_hgat
            attention = pd.read_csv(attention_file, index_col=0)
            print(f"Reusing cached HGAT embedding from: {hgat_cache_file}")
            print(f"Reusing cached attention from: {attention_file}")
        else:
            reuse_hgat = False

    if not reuse_hgat:
        emb_align = Multi_Emb_Align(
            adata,
            batch_key="batch",
            hvg=hvg_top,
            n_hidden=128,
            n_latent=32,
            likelihood="nb",
            num_workers=0,
            result_path=result_dir,
            atac_key="ATAC",
        )

        emb_align.prepare_hgat(
            spatial_key="spatial",
            slice_order=batch_order,
            n_neigh_hom=n_neigh_hom,
            c_neigh_het=c_neigh_het,
            sim_threshold=sim_threshold,
        )

        emb_align.train_hgat(
            mini_batch=mini_batch,
            epoch_hgat=hgat_epoch,
            batches=hgat_batches,
        )

        adata, attention = emb_align.predict_hgat(
            mini_batch=mini_batch,
            batches=hgat_batches,
        )
        attention.to_csv(attention_file)
        adata.write(hgat_cache_file)
        print(f"Saved HGAT cache to: {hgat_cache_file}")

    # Ensure attention is always persisted (also for cache-reuse path).
    attention.to_csv(attention_file)

    adata = cluster_func(
        adata,
        clustering="mclust",
        use_rep="STAIR",
        cluster_num=cluster_num,
        key_add="STAIR",
    )
    adata.obs["Domain"] = adata.obs["STAIR"].astype(str)
    print("Clustering method: mclust")

    moran_df = _compute_slice_moran_curve(
        adata,
        domain_key="Domain",
        spatial_key="spatial",
        slice_key="batch",
        k_list=moran_k_list,
    )
    moran_file = os.path.join(result_dir, "moran_scores.csv")
    moran_df.to_csv(moran_file, index=False)

    # Keep Moran metrics in CSV only (no metric figure output).

    sc.pp.neighbors(adata, use_rep="STAIR")
    sc.tl.umap(adata, min_dist=0.2)

    plt.figure(figsize=(6.0, 4.8))
    sc.pl.umap(adata, color=["batch", "STAIR"], frameon=False, show=False)
    plt.savefig(os.path.join(result_dir, "umap_batch_stair.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.2, 4.8))
    sc.pl.umap(adata, color=["Domain"], frameon=False, show=False)
    plt.savefig(os.path.join(result_dir, "umap_domain.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Spatial domain overview by slice (similar to spatial_domains_2d style)
    if "spatial" in adata.obsm:
        n_slice = len(batch_order)
        n_col = min(4, max(1, n_slice))
        n_row = int(math.ceil(n_slice / n_col))
        fig, axs = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 3.8 * n_row), constrained_layout=True)
        axs = np.array(axs).reshape(n_row, n_col)

        x_min, x_max = adata.obsm["spatial"][:, 0].min(), adata.obsm["spatial"][:, 0].max()
        y_min, y_max = adata.obsm["spatial"][:, 1].min(), adata.obsm["spatial"][:, 1].max()

        idx = 0
        for i in range(n_row):
            for j in range(n_col):
                ax = axs[i, j]
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis("off")
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])

                if idx < n_slice:
                    batch_name = batch_order[idx]
                    adata_tmp = adata[adata.obs["batch"].astype(str) == batch_name].copy()
                    show_legend = (idx == n_slice - 1)
                    sc.pl.embedding(
                        adata_tmp,
                        basis="spatial",
                        color="Domain",
                        title=str(batch_name),
                        frameon=False,
                        legend_loc="right margin" if show_legend else None,
                        s=10,
                        show=False,
                        ax=ax,
                    )
                idx += 1

        plt.savefig(os.path.join(result_dir, "spatial_domains_2d.png"), dpi=300, bbox_inches="tight")
        plt.close()

    adata.write(processed_file)
    print(f"Saved attention to: {attention_file}")
    print(f"Saved Moran scores to: {moran_file}")
    print(f"Saved spatial domains to: {os.path.join(result_dir, 'spatial_domains_2d.png')}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
