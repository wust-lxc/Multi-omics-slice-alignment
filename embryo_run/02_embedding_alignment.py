import warnings
warnings.filterwarnings("ignore")

import os

# Limit BLAS/OpenMP threads early so standalone runs of this step remain stable.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc

from STAIR.emb_alignment import Emb_Align
from STAIR.utils import set_seed, cluster_func


def main():
    set_seed(42)

    # Hardcoded run profile (high-precision version) for convenient repeated runs.
    hvg_top = 5000  # 用于建模的高变基因数量。
    ae_epoch = 100  # 自编码器预训练轮数。
    ae_batch_size = 256  # 自编码器训练时的批大小。
    hgat_epoch = 100  # HGAT 跨切片对齐训练轮数。
    hgat_batches = 10  # HGAT 训练/预测分块数。
    sim_threshold = 0.5  # 构建跨切片连接时的相似度阈值。
    c_neigh_het = 0.93  # 跨切片异质邻居的比例/权重参数。
    n_neigh_hom = 10  # 同切片同质邻居数量。
    mini_batch = False  # 是否启用 HGAT 小批量模式。
    max_cells_per_stage = 12000  # 每个发育阶段最多保留的细胞数。
    cluster_num = 18  # mclust 目标簇数（需安装 R 包 mclust 与 rpy2）。

    result_dir = "./embryo_result"
    embedding_dir = os.path.join(result_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)

    merged_file = os.path.join(result_dir, "embryo_merged.h5ad")
    processed_file = os.path.join(result_dir, "embryo_processed.h5ad")

    if not os.path.exists(merged_file):
        raise FileNotFoundError("embryo_merged.h5ad not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(merged_file)
    adata.obs_names_make_unique()

    stage_df = adata.obs[["batch_safe", "stage"]].drop_duplicates().sort_values("stage")
    batch_order = stage_df["batch_safe"].astype(str).tolist()
    print("Detected slice order:", batch_order)

    if max_cells_per_stage > 0:
        kept = []
        for batch_name in batch_order:
            idx = np.where(adata.obs["batch_safe"].astype(str).values == batch_name)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_cells_per_stage:
                idx = np.random.choice(idx, size=max_cells_per_stage, replace=False)
            kept.append(idx)
        if kept:
            keep_idx = np.concatenate(kept)
            adata = adata[keep_idx].copy()
            print(
                f"Subsampled for memory: max {max_cells_per_stage} cells/stage, "
                f"new shape={adata.n_obs}x{adata.n_vars}"
            )

    emb_align = Emb_Align(
        adata,
        batch_key="batch_safe",
        hvg=hvg_top,
        num_workers=0,
        result_path=result_dir,
    )

    emb_align.prepare(lib_size="explog", normalize=True, scale=False)
    emb_align.preprocess(epoch_ae=ae_epoch, batch_size=ae_batch_size)
    emb_align.latent()

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

    attention_file = os.path.join(embedding_dir, "attention.csv")
    attention.to_csv(attention_file)

    adata = cluster_func(
        adata,
        clustering="mclust",
        use_rep="STAIR",
        cluster_num=cluster_num,
        key_add="STAIR",
    )
    n_clusters = int(adata.obs["STAIR"].nunique())
    adata.obs["Domain"] = adata.obs["STAIR"].astype(str)
    print(f"Mclust clustering finished: clusters={n_clusters}")

    sc.pp.neighbors(adata, use_rep="STAIR")
    sc.tl.umap(adata, min_dist=0.2)

    plt.figure(figsize=(5.5, 4.5))
    sc.pl.umap(adata, color=["batch_safe", "STAIR"], frameon=False, show=False)
    plt.savefig(os.path.join(result_dir, "umap_batch_stair.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.0, 4.5))
    sc.pl.umap(adata, color=["Domain"], frameon=False, show=False)
    plt.savefig(os.path.join(result_dir, "umap_domain.png"), dpi=300, bbox_inches="tight")
    plt.close()

    adata.write(processed_file)

    print(f"Saved attention to: {attention_file}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
