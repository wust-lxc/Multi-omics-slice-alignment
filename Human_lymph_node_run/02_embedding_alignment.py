import warnings
warnings.filterwarnings("ignore")

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
import json
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

    w = 1.0 / float(k_use)
    num = np.sum(z[:, None] * z[neigh_idx] * w)
    return float(num / denom)


def _compute_slice_moran(adata, domain_key="Domain", spatial_key="spatial", slice_key="batch", k=6):
    rows = []
    for slice_name in sorted(adata.obs[slice_key].astype(str).unique()):
        adata_tmp = adata[adata.obs[slice_key].astype(str) == slice_name].copy()
        coords = adata_tmp.obsm[spatial_key]
        codes = pd.Categorical(adata_tmp.obs[domain_key].astype(str)).codes.astype(np.float64)
        rows.append(
            {
                "metric_group": "moran",
                "metric_name": "moran_i_domain_knn",
                "slice": str(slice_name),
                "slice_pair": "",
                "value": float(_moran_i_knn(codes, coords, k=k)),
                "extra": f"k={k};n_cells={adata_tmp.n_obs}",
            }
        )
    return rows


def main():
    set_seed(42)

    hvg_top = 4000
    ae_epoch = 200
    ae_batch_size = 256
    loss_weight_rna = 1.0
    loss_weight_atac = 10.0
    atac_loss = "mse"

    hgat_epoch = 200
    hgat_batches = 6
    sim_threshold = 0.3
    c_neigh_het = 0.35
    n_neigh_hom = 10
    mini_batch = False

    cluster_num = 8

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Human_lymph_node_result")
    embedding_dir = os.path.join(result_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)

    merged_file = os.path.join(result_dir, "human_lymph_node_merged.h5ad")
    processed_file = os.path.join(result_dir, "human_lymph_node_processed.h5ad")

    if not os.path.exists(merged_file):
        raise FileNotFoundError("human_lymph_node_merged.h5ad not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(merged_file)
    adata.obs_names_make_unique()
    if "final_annot" in adata.obs.columns:
        adata.obs["final_annot"] = (adata.obs["final_annot"]
            .astype(str)
            .str.lower()
            .str.replace("vessels", "vessel"))
    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )
    print("Detected slice order:", batch_order)

    hvg_top = min(hvg_top, adata.n_vars)

    emb_align = Multi_Emb_Align(
        adata,
        batch_key="batch",
        hvg=hvg_top,
        n_hidden=128,
        n_latent=32,
        likelihood="nb",
        num_workers=0,
        result_path=result_dir,
        atac_key="ADT",
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

    emb_align.prepare_hgat(
        spatial_key="spatial",
        slice_order=batch_order,
        n_neigh_hom=n_neigh_hom,
        c_neigh_het=c_neigh_het,
        sim_threshold=sim_threshold,
    )

    emb_align.train_hgat(
        gamma=0.80,
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

    # Prefer mclust, fallback to kmeans when R-side runtime/packages are unavailable.
    try:
        adata = cluster_func(
            adata,
            clustering="mclust",
            use_rep="STAIR",
            cluster_num=cluster_num,
            key_add="STAIR",
        )
        cluster_method = "mclust"
    except Exception as exc:
        print("mclust failed, fallback to kmeans clustering.")
        print(f"Detail: {exc}")
        adata = cluster_func(
            adata,
            clustering="kmeans",
            use_rep="STAIR",
            cluster_num=cluster_num,
            key_add="STAIR",
        )
        cluster_method = "kmeans"

    adata.obs["Domain"] = adata.obs["STAIR"].astype(str)

    moran_rows = _compute_slice_moran(
        adata,
        domain_key="Domain",
        spatial_key="spatial",
        slice_key="batch",
        k=6,
    )
    adata.uns["metrics_moran_rows_json"] = json.dumps(moran_rows, ensure_ascii=True)
    adata.uns["cluster_method"] = cluster_method
    adata.obs["cluster_method"] = cluster_method

    sc.pp.neighbors(adata, use_rep="STAIR")
    sc.tl.umap(adata, min_dist=0.2)

    adata.write(processed_file)

    print(f"Clustering method: {cluster_method}")
    print(f"Saved attention to: {attention_file}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()