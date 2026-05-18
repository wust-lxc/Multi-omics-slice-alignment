import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


def _directed_chamfer_sq(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(b)
    dist, _ = nn.kneighbors(a, return_distance=True)
    return float((dist * dist).mean())


def _chamfer_row(points: np.ndarray, batches: np.ndarray, slice_a: str, slice_b: str, stage: str) -> dict:
    a = points[batches == slice_a]
    b = points[batches == slice_b]
    d_ab = _directed_chamfer_sq(a, b)
    d_ba = _directed_chamfer_sq(b, a)
    d_sym = np.nan if np.isnan(d_ab) or np.isnan(d_ba) else d_ab + d_ba
    return {
        "stage": stage,
        "slice_a": slice_a,
        "slice_b": slice_b,
        "n_a": int(a.shape[0]),
        "n_b": int(b.shape[0]),
        "cd_sq_a_to_b": d_ab,
        "cd_sq_b_to_a": d_ba,
        "cd_sq_symmetric": float(d_sym),
    }


def _minmax_normalize_by_slice(points: np.ndarray, batches: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    batches = np.asarray(batches).astype(str)
    out = np.zeros_like(points, dtype=np.float64)
    for b in np.unique(batches):
        idx = np.where(batches == b)[0]
        p = points[idx]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        span = p_max - p_min
        span[span <= 1e-12] = 1.0
        out[idx] = (p - p_min) / span
    return out


def main():
    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac_ATAC"
    processed_file = result_dir / "h3k27ac_atac_processed.h5ad"
    final_file = result_dir / "adata.h5ad"
    chamfer_file = result_dir / "chamfer_distance.csv"
    metric_file = result_dir / "metrics_summary.csv"

    if not processed_file.exists():
        raise FileNotFoundError("h3k27ac_atac_processed.h5ad not found. Run previous steps first.")

    adata = sc.read_h5ad(processed_file)
    if "transform_init" not in adata.obsm or "transform_fine" not in adata.obsm:
        raise KeyError("transform_init/transform_fine missing. Run 04_location_alignment.py first.")

    slice_a, slice_b = "H3K27ac", "ATAC"
    batches = adata.obs["batch"].astype(str).values

    rows = []
    stage_points = {
        "input_raw": np.asarray(adata.obsm["spatial"], dtype=np.float64),
        "init_raw": np.asarray(adata.obsm["transform_init"], dtype=np.float64),
        "fine_raw": np.asarray(adata.obsm["transform_fine"], dtype=np.float64),
    }
    for stage, pts in stage_points.items():
        rows.append(_chamfer_row(pts, batches, slice_a, slice_b, stage))

    for stage, pts in stage_points.items():
        pts_norm = _minmax_normalize_by_slice(pts, batches)
        rows.append(_chamfer_row(pts_norm, batches, slice_a, slice_b, f"{stage}_per_slice_minmax"))

    chamfer_df = pd.DataFrame(rows)
    chamfer_df.to_csv(chamfer_file, index=False)

    adata.obsm["rec_3d"] = np.column_stack(
        [
            np.asarray(adata.obsm["transform_fine"], dtype=np.float64),
            adata.obs["z_rec"].astype(float).values if "z_rec" in adata.obs else np.zeros(adata.n_obs),
        ]
    )
    z_rec = adata.obs["z_rec"].astype(float).values if "z_rec" in adata.obs else np.zeros(adata.n_obs)
    xy_aligned_plot = _minmax_normalize_by_slice(adata.obsm["transform_fine"], batches)
    xy_input_plot = _minmax_normalize_by_slice(adata.obsm["spatial"], batches)
    adata.obsm["rec_3d_plot"] = np.column_stack(
        [
            xy_aligned_plot[:, 0],
            xy_aligned_plot[:, 1],
            -z_rec,
        ]
    )
    adata.obsm["gt_3d_order_plot"] = np.column_stack(
        [
            xy_input_plot[:, 0],
            xy_input_plot[:, 1],
            -z_rec,
        ]
    )

    color_key = "Domain" if "Domain" in adata.obs else "batch"

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Mouse brain H3K27ac-ATAC reconstructed 3D",
    )
    plt.savefig(result_dir / "reconstruction_3d_rec.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="gt_3d_order_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Mouse brain H3K27ac-ATAC reference 3D",
    )
    plt.savefig(result_dir / "reconstruction_3d_reference.png", dpi=300, bbox_inches="tight")
    plt.close()

    metrics_rows = [
        {
            "metric_group": "basic",
            "metric_name": "n_cells",
            "slice": "ALL",
            "slice_pair": "H3K27ac|ATAC",
            "value": float(adata.n_obs),
            "extra": "",
        },
        {
            "metric_group": "basic",
            "metric_name": "n_genes",
            "slice": "ALL",
            "slice_pair": "H3K27ac|ATAC",
            "value": float(adata.n_vars),
            "extra": "",
        },
    ]
    for _, row in chamfer_df.iterrows():
        metrics_rows.append(
            {
                "metric_group": "chamfer",
                "metric_name": str(row["stage"]),
                "slice": "",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(row["cd_sq_symmetric"]),
                "extra": f"cd_sq_a_to_b={row['cd_sq_a_to_b']};cd_sq_b_to_a={row['cd_sq_b_to_a']}",
            }
        )
    if "Domain" in adata.obs:
        metrics_rows.append(
            {
                "metric_group": "clustering",
                "metric_name": "n_domain_clusters",
                "slice": "ALL",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(adata.obs["Domain"].astype(str).nunique()),
                "extra": "",
            }
        )
    metrics_rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "cluster_method",
            "slice": "ALL",
            "slice_pair": "H3K27ac|ATAC",
            "value": np.nan,
            "extra": str(adata.uns.get("cluster_method", "unknown")),
        }
    )
    if "alignment_rms_init" in adata.uns:
        metrics_rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_init_vs_input",
                "slice": "ALL",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(adata.uns["alignment_rms_init"]),
                "extra": "",
            }
        )
    if "alignment_rms_fine" in adata.uns:
        metrics_rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_fine_vs_input",
                "slice": "ALL",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(adata.uns["alignment_rms_fine"]),
                "extra": "",
            }
        )

    pd.DataFrame(metrics_rows).to_csv(metric_file, index=False)
    adata.write(final_file)

    print(f"Saved Chamfer distances to: {chamfer_file}")
    print(f"Saved metrics summary to: {metric_file}")
    print(f"Saved final AnnData to: {final_file}")


if __name__ == "__main__":
    main()
