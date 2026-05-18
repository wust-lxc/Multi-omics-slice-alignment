import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _directed_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(b)
    dist, _ = nn.kneighbors(a, return_distance=True)
    return float((dist * dist).mean())


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = _directed_chamfer(a, b)
    d_ba = _directed_chamfer(b, a)
    if np.isnan(d_ab) or np.isnan(d_ba):
        return np.nan
    return float(d_ab + d_ba)


def _adjacent_chamfer_table(points: np.ndarray, batches: np.ndarray, order_use) -> pd.DataFrame:
    rows = []
    for i in range(len(order_use) - 1):
        a_name = order_use[i]
        b_name = order_use[i + 1]
        a_pts = points[batches == a_name]
        b_pts = points[batches == b_name]
        d_ab = _directed_chamfer(a_pts, b_pts)
        d_ba = _directed_chamfer(b_pts, a_pts)
        d_sym = _symmetric_chamfer(a_pts, b_pts)
        rows.append(
            {
                "slice_a": a_name,
                "slice_b": b_name,
                "n_a": int(a_pts.shape[0]),
                "n_b": int(b_pts.shape[0]),
                "cd_sq_a_to_b": d_ab,
                "cd_sq_b_to_a": d_ba,
                "cd_sq_symmetric": d_sym,
            }
        )
    return pd.DataFrame(rows)


def _minmax_normalize_by_slice(points: np.ndarray, batches: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    batches = np.asarray(batches).astype(str)
    out = np.zeros_like(points, dtype=np.float64)

    for b in np.unique(batches):
        idx = np.where(batches == b)[0]
        if idx.size == 0:
            continue
        p = points[idx]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        span = p_max - p_min
        span[span <= 1e-12] = 1.0
        out[idx] = (p - p_min) / span
    return out


def _neg_log10_safe(v: pd.Series, eps: float = 1e-12) -> pd.Series:
    arr = np.asarray(v, dtype=np.float64)
    arr = np.clip(arr, eps, None)
    return pd.Series(-np.log10(arr), index=v.index)


def _pick_hvg_mask(adata, top_fallback: int = 2000) -> np.ndarray:
    if "highly_variable" in adata.var:
        mask = np.asarray(adata.var["highly_variable"].values).astype(bool)
        if int(mask.sum()) > 1:
            return mask

    x = adata.X
    if sparse.issparse(x):
        x2_mean = np.asarray(x.power(2).mean(axis=0)).ravel()
        x_mean = np.asarray(x.mean(axis=0)).ravel()
    else:
        x = np.asarray(x)
        x2_mean = np.mean(x * x, axis=0)
        x_mean = np.mean(x, axis=0)

    var = x2_mean - x_mean * x_mean
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    n_vars = int(var.shape[0])
    top_k = int(min(max(2, top_fallback), n_vars))
    top_idx = np.argpartition(var, -top_k)[-top_k:]
    mask = np.zeros(n_vars, dtype=bool)
    mask[top_idx] = True
    return mask


def _collect_cross_slice_nearest_pairs(coords_3d: np.ndarray, batches: np.ndarray) -> np.ndarray:
    batches = np.asarray(batches).astype(str)
    unique_batches = np.unique(batches)
    directed_pairs = []

    for batch_name in unique_batches:
        idx_src = np.where(batches == batch_name)[0]
        idx_tgt = np.where(batches != batch_name)[0]
        if idx_src.size == 0 or idx_tgt.size == 0:
            continue

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(coords_3d[idx_tgt])
        _, tgt_local = nn.kneighbors(coords_3d[idx_src], return_distance=True)
        tgt_idx = idx_tgt[tgt_local[:, 0]]
        directed_pairs.append(np.column_stack([idx_src, tgt_idx]))

    if not directed_pairs:
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.vstack(directed_pairs).astype(np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs


def _row_as_dense(x, i: int) -> np.ndarray:
    if sparse.issparse(x):
        return x.getrow(i).toarray().ravel().astype(np.float64)
    return np.asarray(x[i]).ravel().astype(np.float64)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= 1 or y.size <= 1:
        return np.nan

    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if denom <= 0:
        return np.nan
    return float(np.sum(xm * ym) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(rx, ry)


def _pairwise_pearson_for_pairs(matrix, pairs: np.ndarray) -> np.ndarray:
    out = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
    for i, (idx_a, idx_b) in enumerate(pairs):
        vec_a = _row_as_dense(matrix, int(idx_a))
        vec_b = _row_as_dense(matrix, int(idx_b))
        out[i] = _pearson_corr(vec_a, vec_b)
    return out


def _cross_slice_expression_corr_table(
    adata,
    coords_3d_key: str = "rec_3d",
    batch_key: str = "batch",
    hvg_fallback_top: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if coords_3d_key not in adata.obsm:
        raise KeyError(f"{coords_3d_key} not found in adata.obsm")
    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key} not found in adata.obs")

    coords_3d = np.asarray(adata.obsm[coords_3d_key], dtype=np.float64)
    batches = adata.obs[batch_key].astype(str).values
    pairs = _collect_cross_slice_nearest_pairs(coords_3d=coords_3d, batches=batches)

    hvg_mask = _pick_hvg_mask(adata, top_fallback=hvg_fallback_top)
    n_hvg = int(hvg_mask.sum())
    x_hvg = adata[:, hvg_mask].X

    pearson_rna = _pairwise_pearson_for_pairs(x_hvg, pairs)

    spearman_rna = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
    for i, (idx_a, idx_b) in enumerate(pairs):
        vec_a = _row_as_dense(x_hvg, int(idx_a))
        vec_b = _row_as_dense(x_hvg, int(idx_b))
        spearman_rna[i] = _spearman_corr(vec_a, vec_b)

    if "ADT" in adata.obsm:
        adt_mat = adata.obsm["ADT"]
        pearson_adt = _pairwise_pearson_for_pairs(adt_mat, pairs)
    else:
        pearson_adt = np.full((pairs.shape[0],), np.nan, dtype=np.float64)

    if "STAIR" in adata.obsm:
        stair_mat = adata.obsm["STAIR"]
        pearson_stair = _pairwise_pearson_for_pairs(stair_mat, pairs)
    else:
        pearson_stair = np.full((pairs.shape[0],), np.nan, dtype=np.float64)

    pair_df = pd.DataFrame(
        {
            "cell_a": [str(adata.obs_names[int(i)]) for i in pairs[:, 0]] if pairs.size else [],
            "cell_b": [str(adata.obs_names[int(i)]) for i in pairs[:, 1]] if pairs.size else [],
            "slice_a": [str(batches[int(i)]) for i in pairs[:, 0]] if pairs.size else [],
            "slice_b": [str(batches[int(i)]) for i in pairs[:, 1]] if pairs.size else [],
            "distance_3d": [
                float(np.linalg.norm(coords_3d[int(i)] - coords_3d[int(j)])) for i, j in pairs
            ]
            if pairs.size
            else [],
            "pearson_hvg": pearson_rna,
            "spearman_hvg": spearman_rna,
            "pearson_adt": pearson_adt,
            "pearson_stair": pearson_stair,
            "n_hvg": [n_hvg] * pairs.shape[0],
        }
    )

    summary_rows = [
        {
            "level": "global",
            "slice_pair": "ALL",
            "n_pairs": int(pair_df.shape[0]),
            "n_hvg": n_hvg,
            "pearson_mean": float(pair_df["pearson_hvg"].mean()) if not pair_df.empty else np.nan,
            "pearson_median": float(pair_df["pearson_hvg"].median()) if not pair_df.empty else np.nan,
            "spearman_mean": float(pair_df["spearman_hvg"].mean()) if not pair_df.empty else np.nan,
            "spearman_median": float(pair_df["spearman_hvg"].median()) if not pair_df.empty else np.nan,
            "distance_3d_mean": float(pair_df["distance_3d"].mean()) if not pair_df.empty else np.nan,
        }
    ]

    if not pair_df.empty:
        tmp = pair_df.copy()
        tmp["slice_pair"] = tmp.apply(
            lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])),
            axis=1,
        )
        for pair_name, g in tmp.groupby("slice_pair", sort=True):
            summary_rows.append(
                {
                    "level": "slice_pair",
                    "slice_pair": str(pair_name),
                    "n_pairs": int(g.shape[0]),
                    "n_hvg": n_hvg,
                    "pearson_mean": float(g["pearson_hvg"].mean()),
                    "pearson_median": float(g["pearson_hvg"].median()),
                    "spearman_mean": float(g["spearman_hvg"].mean()),
                    "spearman_median": float(g["spearman_hvg"].median()),
                    "distance_3d_mean": float(g["distance_3d"].mean()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return pair_df, summary_df, _adjacent_chamfer_table(coords_3d, batches, np.unique(batches))


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Simulation_result")
    export_dir = os.path.join(result_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    processed_file = os.path.join(result_dir, "simulation_processed.h5ad")
    order_file = os.path.join(result_dir, "predicted_slice_order.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("simulation_processed.h5ad not found. Run 04_location_alignment.py first.")

    adata = sc.read_h5ad(processed_file)
    if "transform_fine" not in adata.obsm:
        raise KeyError("transform_fine not found in adata.obsm. Run 04_location_alignment.py first.")

    if os.path.exists(order_file):
        order_df = pd.read_csv(order_file)
        order_use = order_df.sort_values("z_rec")["batch"].astype(str).tolist()
    else:
        order_use = sorted(adata.obs["batch"].astype(str).unique())

    coords = adata.obsm["transform_fine"]
    z_rec = adata.obs["z_rec"].astype(float).values if "z_rec" in adata.obs else None

    if z_rec is None:
        raise KeyError("z_rec not found in adata.obs. Run 03_slice_order_and_z_reconstruction.py first.")

    coords_3d = np.column_stack([coords, z_rec])
    coords_3d = _minmax_normalize_by_slice(coords_3d, adata.obs["batch"].astype(str).values)
    adata.obsm["rec_3d"] = coords_3d

    batches_plot = adata.obs["batch"].astype(str).values
    xy_aligned_plot = _minmax_normalize_by_slice(adata.obsm["transform_fine"], batches_plot)
    xy_input_plot = _minmax_normalize_by_slice(adata.obsm["spatial"], batches_plot)

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

    adata.write(processed_file)

    color_key = "Domain" if "Domain" in adata.obs else "batch"
    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Simulation reconstructed 3D",
    )
    plt.savefig(os.path.join(result_dir, "reconstruction_3d_rec.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="gt_3d_order_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Simulation reference 3D (x,y,z_rec)",
    )
    plt.savefig(os.path.join(result_dir, "reconstruction_3d_reference.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if "spatial" in adata.obsm:
        n_slice = len(order_use)
        n_col = min(4, max(1, n_slice))
        n_row = int(np.ceil(n_slice / n_col))
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
                    batch_name = order_use[idx]
                    adata_tmp = adata[adata.obs["batch"].astype(str) == batch_name].copy()
                    show_legend = (idx == n_slice - 1)
                    sc.pl.embedding(
                        adata_tmp,
                        basis="spatial",
                        color=color_key,
                        title=str(batch_name),
                        frameon=False,
                        legend_loc="right margin" if show_legend else None,
                        s=100,
                        show=False,
                        ax=ax,
                    )
                idx += 1

        plt.savefig(os.path.join(result_dir, "spatial_domains_2d.png"), dpi=300, bbox_inches="tight")
        plt.close()

    chamfer_adjacent = _adjacent_chamfer_table(coords_3d, adata.obs["batch"].astype(str).values, order_use)
    chamfer_adjacent.to_csv(os.path.join(result_dir, "chamfer_adjacent_slices.csv"), index=False)

    pair_df, summary_df, _ = _cross_slice_expression_corr_table(adata)
    pair_df.to_csv(os.path.join(result_dir, "cross_slice_nearest_expr_corr_pairs.csv"), index=False)
    summary_df.to_csv(os.path.join(result_dir, "cross_slice_nearest_expr_corr_summary.csv"), index=False)

    plt.figure(figsize=(5.5, 4.5))
    sc.pl.embedding(
        adata,
        basis="rec_3d",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        show=False,
    )
    plt.savefig(os.path.join(export_dir, "rec_3d_batch_domain.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Updated processed data: {processed_file}")
    print(f"Saved exports to: {export_dir}")


if __name__ == "__main__":
    main()
