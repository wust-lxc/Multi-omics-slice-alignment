import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def _minmax_normalize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    p_min = points.min(axis=0, keepdims=True)
    p_max = points.max(axis=0, keepdims=True)
    span = p_max - p_min
    span[span <= 1e-12] = 1.0
    return (points - p_min) / span


def _moran_i_knn(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return np.nan

    k_use = max(1, min(int(k), n - 1))
    neigh = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(coords)
    neigh_idx = neigh.kneighbors(coords, return_distance=False)[:, 1:]

    z = values - values.mean()
    denom = np.sum(z * z)
    if denom <= 0:
        return np.nan

    w = 1.0 / float(k_use)
    num = np.sum(z[:, None] * z[neigh_idx] * w)
    return float(num / denom)


def _add_ari_row(metrics_rows: list[dict], adata, truth_key: str = "truth", pred_key: str = "Domain") -> None:
    if truth_key not in adata.obs or pred_key not in adata.obs:
        return
    valid = ~adata.obs[truth_key].isna() & ~adata.obs[pred_key].isna()
    if int(valid.sum()) == 0:
        return
    y_true = adata.obs.loc[valid, truth_key].astype(str)
    y_pred = adata.obs.loc[valid, pred_key].astype(str)
    metrics_rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "ARI",
            "slice": "ATAC",
            "slice_pair": "",
            "value": float(adjusted_rand_score(y_true, y_pred)),
            "extra": f"truth_key={truth_key};pred_key={pred_key};n_valid={int(valid.sum())};n_truth={int(y_true.nunique())};n_pred={int(y_pred.nunique())}",
        }
    )


def main():
    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "ATAC"
    processed_file = result_dir / "atac_processed.h5ad"
    final_file = result_dir / "adata.h5ad"
    metric_file = result_dir / "metrics_summary.csv"

    if not processed_file.exists():
        raise FileNotFoundError("atac_processed.h5ad not found. Run previous steps first.")

    adata = sc.read_h5ad(processed_file)
    if "Domain" not in adata.obs:
        raise KeyError("Domain not found. Run 02_embedding_and_clustering.py first.")

    z = np.zeros(adata.n_obs, dtype=np.float64)
    adata.obs["z_rec"] = z
    adata.obsm["rec_3d"] = np.column_stack([adata.obsm["spatial"], z])
    xy_plot = _minmax_normalize(adata.obsm["spatial"])
    adata.obsm["gt_3d_order_plot"] = np.column_stack([xy_plot[:, 0], xy_plot[:, 1], z])

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="Domain",
        frameon=False,
        s=8,
        show=False,
        title="Mouse brain ATAC mclust domains",
    )
    plt.savefig(result_dir / "spatial_domain.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="gt_3d_order_plot",
        projection="3d",
        color="Domain",
        s=2,
        show=False,
        title="Mouse brain ATAC reference 3D",
    )
    plt.savefig(result_dir / "reconstruction_3d_reference.png", dpi=300, bbox_inches="tight")
    plt.close()

    metrics_rows = [
        {
            "metric_group": "basic",
            "metric_name": "n_cells",
            "slice": "ATAC",
            "slice_pair": "",
            "value": float(adata.n_obs),
            "extra": "",
        },
        {
            "metric_group": "basic",
            "metric_name": "n_genes",
            "slice": "ATAC",
            "slice_pair": "",
            "value": float(adata.n_vars),
            "extra": "",
        },
        {
            "metric_group": "clustering",
            "metric_name": "cluster_method",
            "slice": "ATAC",
            "slice_pair": "",
            "value": np.nan,
            "extra": str(adata.uns.get("cluster_method", "unknown")),
        },
        {
            "metric_group": "clustering",
            "metric_name": "n_domain_clusters",
            "slice": "ATAC",
            "slice_pair": "",
            "value": float(adata.obs["Domain"].astype(str).nunique()),
            "extra": "",
        },
    ]
    domain_codes = pd.Categorical(adata.obs["Domain"].astype(str)).codes.astype(np.float64)
    metrics_rows.append(
        {
            "metric_group": "moran",
            "metric_name": "Moran's I",
            "slice": "ATAC",
            "slice_pair": "",
            "value": float(_moran_i_knn(domain_codes, adata.obsm["spatial"], k=6)),
            "extra": f"source=moran_i_domain_knn;k=6;n_cells={adata.n_obs}",
        }
    )
    _add_ari_row(metrics_rows, adata, truth_key="truth", pred_key="Domain")
    pd.DataFrame(metrics_rows).to_csv(metric_file, index=False)
    adata.write(final_file)

    print(f"Saved metrics to: {metric_file}")
    print(f"Saved final AnnData to: {final_file}")


if __name__ == "__main__":
    main()
