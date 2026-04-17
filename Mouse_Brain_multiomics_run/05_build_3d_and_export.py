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
    gene_names = adata.var_names[hvg_mask].astype(str)
    n_hvg = int(hvg_mask.sum())
    x_hvg = adata[:, hvg_mask].X

    pearson_rna = _pairwise_pearson_for_pairs(x_hvg, pairs)

    spearman_rna = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
    for i, (idx_a, idx_b) in enumerate(pairs):
        vec_a = _row_as_dense(x_hvg, int(idx_a))
        vec_b = _row_as_dense(x_hvg, int(idx_b))
        spearman_rna[i] = _spearman_corr(vec_a, vec_b)

    if "ATAC" in adata.obsm:
        atac_mat = adata.obsm["ATAC"]
        pearson_atac = _pairwise_pearson_for_pairs(atac_mat, pairs)
        n_atac_features = int(np.asarray(atac_mat).shape[1]) if not sparse.issparse(atac_mat) else int(atac_mat.shape[1])
    else:
        pearson_atac = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
        n_atac_features = 0

    if "STAIR" in adata.obsm:
        stair_mat = adata.obsm["STAIR"]
        pearson_stair = _pairwise_pearson_for_pairs(stair_mat, pairs)
        n_stair_features = int(np.asarray(stair_mat).shape[1]) if not sparse.issparse(stair_mat) else int(stair_mat.shape[1])
    else:
        pearson_stair = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
        n_stair_features = 0

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
            "pearson_atac": pearson_atac,
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

    pcc_dimension_df = pd.DataFrame(
        [
            {
                "evaluation_dimension": "RNA continuity (RNA PCC)",
                "feature_source": "adata.X (HVG normalized expression)",
                "core_purpose": "Verify alignment preserves spatial continuity of transcriptomic signal",
                "expected_behavior": "Usually high due to smoother RNA continuity",
                "metric_column": "pearson_hvg",
                "pcc_mean": float(pair_df["pearson_hvg"].mean()) if not pair_df.empty else np.nan,
                "pcc_median": float(pair_df["pearson_hvg"].median()) if not pair_df.empty else np.nan,
                "n_pairs": int(pair_df.shape[0]),
                "n_features": n_hvg,
            },
            {
                "evaluation_dimension": "Epigenomic continuity (ATAC PCC)",
                "feature_source": "adata.obsm['ATAC']",
                "core_purpose": "Verify robustness to sparse/noisy ATAC while retaining spatial structure",
                "expected_behavior": "Typically lower than RNA PCC; better methods should improve it",
                "metric_column": "pearson_atac",
                "pcc_mean": float(pair_df["pearson_atac"].mean()) if not pair_df.empty else np.nan,
                "pcc_median": float(pair_df["pearson_atac"].median()) if not pair_df.empty else np.nan,
                "n_pairs": int(pair_df.shape[0]),
                "n_features": n_atac_features,
            },
            {
                "evaluation_dimension": "Latent manifold coherence (Latent PCC)",
                "feature_source": "adata.obsm['STAIR']",
                "core_purpose": "Verify HGAT fuses modalities/slices into a coherent manifold",
                "expected_behavior": "Should be the highest and reflect core algorithmic value",
                "metric_column": "pearson_stair",
                "pcc_mean": float(pair_df["pearson_stair"].mean()) if not pair_df.empty else np.nan,
                "pcc_median": float(pair_df["pearson_stair"].median()) if not pair_df.empty else np.nan,
                "n_pairs": int(pair_df.shape[0]),
                "n_features": n_stair_features,
            },
        ]
    )

    # Save selected HVGs for traceability.
    adata.uns["hvg_for_cross_slice_corr"] = list(gene_names)
    return pair_df, summary_df, pcc_dimension_df


def _plot_cross_slice_pcc_by_pair(pair_df: pd.DataFrame, result_dir: str) -> tuple[str, str]:
    if pair_df.empty:
        return "", ""

    df = pair_df.copy()
    df["slice_pair"] = df.apply(
        lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])),
        axis=1,
    )

    metrics = [
        ("pearson_hvg", "RNA PCC"),
        ("pearson_atac", "ATAC PCC"),
        ("pearson_stair", "Latent PCC"),
    ]

    pair_order = (
        df.groupby("slice_pair")["distance_3d"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    n_pair = len(pair_order)
    n_col = len(metrics)
    fig, axs = plt.subplots(1, n_col, figsize=(4.8 * n_col, max(3.8, 0.7 * n_pair)))
    axs = np.array(axs).reshape(-1)

    for i, (metric_col, metric_name) in enumerate(metrics):
        ax = axs[i]
        data = [df.loc[df["slice_pair"] == p, metric_col].dropna().to_numpy(dtype=float) for p in pair_order]
        valid_pos = [idx + 1 for idx, arr in enumerate(data) if arr.size > 0]
        valid_data = [arr for arr in data if arr.size > 0]

        if valid_data:
            ax.boxplot(
                valid_data,
                vert=False,
                positions=valid_pos,
                widths=0.6,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="#9ecae1", alpha=0.7),
                medianprops=dict(color="#08519c", linewidth=1.4),
            )

        ax.set_yticks(np.arange(1, n_pair + 1))
        ax.set_yticklabels(pair_order, fontsize=8)
        ax.set_xlabel(metric_name)
        ax.set_title(f"{metric_name} by slice pair")
        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    boxplot_file = os.path.join(result_dir, "cross_slice_pcc_by_pair_boxplot.png")
    plt.savefig(boxplot_file, dpi=300, bbox_inches="tight")
    plt.close()

    mean_rows = []
    for p in pair_order:
        part = df[df["slice_pair"] == p]
        mean_rows.append(
            {
                "slice_pair": p,
                "RNA PCC": float(part["pearson_hvg"].mean()),
                "ATAC PCC": float(part["pearson_atac"].mean()),
                "Latent PCC": float(part["pearson_stair"].mean()),
            }
        )
    mean_df = pd.DataFrame(mean_rows)

    x = np.arange(mean_df.shape[0])
    width = 0.24
    plt.figure(figsize=(max(7.0, 1.6 * mean_df.shape[0]), 4.6))
    plt.bar(x - width, mean_df["RNA PCC"], width=width, label="RNA PCC", color="#3182bd")
    plt.bar(x, mean_df["ATAC PCC"], width=width, label="ATAC PCC", color="#e6550d")
    plt.bar(x + width, mean_df["Latent PCC"], width=width, label="Latent PCC", color="#31a354")
    plt.xticks(x, mean_df["slice_pair"], rotation=25, ha="right")
    plt.ylabel("Mean PCC")
    plt.title("Mean PCC by slice pair")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    meanplot_file = os.path.join(result_dir, "cross_slice_pcc_by_pair_means.png")
    plt.savefig(meanplot_file, dpi=300, bbox_inches="tight")
    plt.close()

    return boxplot_file, meanplot_file


def _export_pcc_chamfer_side_by_side(
    corr_pair_before_df: pd.DataFrame,
    corr_pair_after_df: pd.DataFrame,
    baseline_norm_df: pd.DataFrame,
    result_dir: str,
) -> tuple[str, str]:
    if corr_pair_before_df.empty or corr_pair_after_df.empty or baseline_norm_df.empty:
        return "", ""

    cp_before = corr_pair_before_df.copy()
    cp_before["slice_pair"] = cp_before.apply(
        lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])),
        axis=1,
    )
    cp_after = corr_pair_after_df.copy()
    cp_after["slice_pair"] = cp_after.apply(
        lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])),
        axis=1,
    )

    pcc_before = (
        cp_before.groupby("slice_pair", as_index=False)[["pearson_hvg", "pearson_atac", "pearson_stair"]]
        .mean()
        .rename(
            columns={
                "pearson_hvg": "RNA PCC before",
                "pearson_atac": "ATAC PCC before",
                "pearson_stair": "Latent PCC before",
            }
        )
    )
    pcc_after = (
        cp_after.groupby("slice_pair", as_index=False)[["pearson_hvg", "pearson_atac", "pearson_stair"]]
        .mean()
        .rename(
            columns={
                "pearson_hvg": "RNA PCC after",
                "pearson_atac": "ATAC PCC after",
                "pearson_stair": "Latent PCC after",
            }
        )
    )
    pcc_by_pair = pcc_before.merge(pcc_after, on="slice_pair", how="inner")

    bn = baseline_norm_df[baseline_norm_df["slice_a"] != "MEAN"].copy()
    bn["slice_pair"] = bn.apply(
        lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])),
        axis=1,
    )

    side_df = pcc_by_pair.merge(
        bn[["slice_pair", "neg_log10_cd_sq_before", "neg_log10_cd_sq_after"]],
        on="slice_pair",
        how="inner",
    )

    if side_df.empty:
        return "", ""

    side_df["delta_after_minus_before"] = (
        side_df["neg_log10_cd_sq_after"] - side_df["neg_log10_cd_sq_before"]
    )

    pair_order = side_df["slice_pair"].tolist()

    pcc_rows = []
    for _, r in side_df.iterrows():
        pcc_rows.append(
            {
                "panel": "PCC",
                "slice_pair": str(r["slice_pair"]),
                "RNA PCC before": float(r["RNA PCC before"]),
                "RNA PCC after": float(r["RNA PCC after"]),
                "ATAC PCC before": float(r["ATAC PCC before"]),
                "ATAC PCC after": float(r["ATAC PCC after"]),
                "Latent PCC before": float(r["Latent PCC before"]),
                "Latent PCC after": float(r["Latent PCC after"]),
            }
        )

    chamfer_rows = []
    for _, r in side_df.iterrows():
        chamfer_rows.append(
            {
                "panel": "Chamfer (-log10 CD_sq)",
                "slice_pair": str(r["slice_pair"]),
                "neg_log10_cd_sq_before": float(r["neg_log10_cd_sq_before"]),
                "neg_log10_cd_sq_after": float(r["neg_log10_cd_sq_after"]),
                "delta_after_minus_before": float(r["delta_after_minus_before"]),
            }
        )

    side_table_file = os.path.join(result_dir, "pcc_chamfer_side_by_side.csv")
    side_table = pd.concat(
        [
            pd.DataFrame(pcc_rows),
            pd.DataFrame(chamfer_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    side_table.to_csv(side_table_file, index=False)

    if side_df.empty:
        return side_table_file, ""

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax0 = axs[0]
    x0 = np.arange(len(pair_order))
    w0 = 0.12
    ax0.bar(x0 - 2.5 * w0, side_df["RNA PCC before"], width=w0, label="RNA before", color="#9ecae1")
    ax0.bar(x0 - 1.5 * w0, side_df["RNA PCC after"], width=w0, label="RNA after", color="#3182bd")
    ax0.bar(x0 - 0.5 * w0, side_df["ATAC PCC before"], width=w0, label="ATAC before", color="#fdae6b")
    ax0.bar(x0 + 0.5 * w0, side_df["ATAC PCC after"], width=w0, label="ATAC after", color="#e6550d")
    ax0.bar(x0 + 1.5 * w0, side_df["Latent PCC before"], width=w0, label="Latent before", color="#a1d99b")
    ax0.bar(x0 + 2.5 * w0, side_df["Latent PCC after"], width=w0, label="Latent after", color="#31a354")
    ax0.set_xticks(x0)
    ax0.set_xticklabels(pair_order, rotation=20, ha="right")
    ax0.set_ylabel("Mean PCC")
    ymax = float(
        np.nanmax(
            side_df[
                [
                    "RNA PCC before",
                    "RNA PCC after",
                    "ATAC PCC before",
                    "ATAC PCC after",
                    "Latent PCC before",
                    "Latent PCC after",
                ]
            ].to_numpy()
        )
    )
    ax0.set_ylim(0.0, min(1.0, max(0.8, ymax + 0.08)))
    ax0.set_title("PCC by Slice Pair (Before vs After)")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(frameon=False, fontsize=9)

    ax1 = axs[1]
    x = np.arange(side_df.shape[0])
    w = 0.34
    ax1.bar(x - w / 2, side_df["neg_log10_cd_sq_before"], width=w, label="Before", color="#9ecae1")
    ax1.bar(x + w / 2, side_df["neg_log10_cd_sq_after"], width=w, label="After", color="#08519c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_order, rotation=20, ha="right")
    ax1.set_ylabel("-log10(CD_sq)")
    ax1.set_title("Chamfer by Slice Pair (Normalized)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False)

    plt.tight_layout()
    side_plot_file = os.path.join(result_dir, "pcc_chamfer_side_by_side.png")
    plt.savefig(side_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    return side_table_file, side_plot_file


def main():
    fixed_order = [
        "Mouse_Brain_ATAC",
        "Mouse_Brain_H3K27ac",
        "Mouse_Brain_H3K4me3",
        "Mouse_Brain_H3K27me3",
    ]

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Mouse_Brain_multiomics_result")

    processed_file = os.path.join(result_dir, "multiomics_processed.h5ad")
    final_file = os.path.join(result_dir, "adata.h5ad")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("multiomics_processed.h5ad not found. Run previous scripts first.")

    adata = sc.read_h5ad(processed_file)

    if "transform_fine" not in adata.obsm:
        raise KeyError("transform_fine not found. Run 04_location_alignment.py first.")
    if "z_rec" not in adata.obs:
        raise KeyError("z_rec not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata.obs["x_aligned"] = adata.obsm["transform_fine"][:, 0]
    adata.obs["y_aligned"] = adata.obsm["transform_fine"][:, 1]

    adata.obsm["rec_3d"] = adata.obs[["x_aligned", "y_aligned", "z_rec"]].values
    adata.obsm["pre_3d"] = np.column_stack([
        adata.obsm["spatial"][:, 0],
        adata.obsm["spatial"][:, 1],
        adata.obs["z_rec"].values,
    ])
    adata.obsm["rec_3d_plot"] = np.column_stack([
        adata.obs["x_aligned"].values,
        adata.obs["y_aligned"].values,
        -adata.obs["z_rec"].values,
    ])

    ref_xy = adata.obsm["spatial"]
    adata.obsm["gt_3d_order"] = np.column_stack([ref_xy[:, 0], ref_xy[:, 1], adata.obs["slice_order"].values])
    adata.obsm["gt_3d_order_plot"] = np.column_stack([ref_xy[:, 0], ref_xy[:, 1], -adata.obs["slice_order"].values])

    color_key = "Domain" if "Domain" in adata.obs else "batch"

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Mouse brain reconstructed 3D",
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
        title="Mouse brain reference 3D (x,y,slice_order)",
    )
    plt.savefig(os.path.join(result_dir, "reconstruction_3d_reference.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Chamfer distance on adjacent aligned slices (transform_fine coordinates).
    batch_values = adata.obs["batch"].astype(str).values
    xy_aligned = adata.obsm["transform_fine"]
    xy_input = adata.obsm["spatial"]

    present = set(batch_values)
    order_use = [b for b in fixed_order if b in present]
    chamfer_df = _adjacent_chamfer_table(xy_aligned, batch_values, order_use)
    if not chamfer_df.empty:
        chamfer_df = pd.concat(
            [
                chamfer_df,
                pd.DataFrame(
                    [
                        {
                            "slice_a": "MEAN",
                            "slice_b": "MEAN",
                            "n_a": int(chamfer_df["n_a"].mean()),
                            "n_b": int(chamfer_df["n_b"].mean()),
                            "cd_sq_a_to_b": float(chamfer_df["cd_sq_a_to_b"].mean()),
                            "cd_sq_b_to_a": float(chamfer_df["cd_sq_b_to_a"].mean()),
                            "cd_sq_symmetric": float(chamfer_df["cd_sq_symmetric"].mean()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    chamfer_file = os.path.join(result_dir, "chamfer_adjacent_slices.csv")
    chamfer_df.to_csv(chamfer_file, index=False)

    # Benchmark-friendly CD on slice-wise normalized coordinates in [0,1].
    xy_aligned_norm = _minmax_normalize_by_slice(xy_aligned, batch_values)
    xy_input_norm = _minmax_normalize_by_slice(xy_input, batch_values)

    chamfer_norm_df = _adjacent_chamfer_table(xy_aligned_norm, batch_values, order_use)
    if not chamfer_norm_df.empty:
        chamfer_norm_df = pd.concat(
            [
                chamfer_norm_df,
                pd.DataFrame(
                    [
                        {
                            "slice_a": "MEAN",
                            "slice_b": "MEAN",
                            "n_a": int(chamfer_norm_df["n_a"].mean()),
                            "n_b": int(chamfer_norm_df["n_b"].mean()),
                            "cd_sq_a_to_b": float(chamfer_norm_df["cd_sq_a_to_b"].mean()),
                            "cd_sq_b_to_a": float(chamfer_norm_df["cd_sq_b_to_a"].mean()),
                            "cd_sq_symmetric": float(chamfer_norm_df["cd_sq_symmetric"].mean()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        chamfer_norm_df["neg_log10_cd_sq_symmetric"] = _neg_log10_safe(chamfer_norm_df["cd_sq_symmetric"])

    chamfer_norm_file = os.path.join(result_dir, "chamfer_adjacent_slices_norm.csv")
    chamfer_norm_df.to_csv(chamfer_norm_file, index=False)

    # Baseline comparison: before (spatial) vs after (transform_fine).
    chamfer_before = _adjacent_chamfer_table(xy_input, batch_values, order_use)
    chamfer_after = _adjacent_chamfer_table(xy_aligned, batch_values, order_use)
    baseline_df = chamfer_before[["slice_a", "slice_b", "cd_sq_symmetric"]].rename(
        columns={"cd_sq_symmetric": "cd_sq_before"}
    ).merge(
        chamfer_after[["slice_a", "slice_b", "cd_sq_symmetric"]].rename(columns={"cd_sq_symmetric": "cd_sq_after"}),
        on=["slice_a", "slice_b"],
        how="inner",
    )
    baseline_df["improvement_sq"] = baseline_df["cd_sq_before"] - baseline_df["cd_sq_after"]
    baseline_df["improvement_sq_pct"] = np.where(
        baseline_df["cd_sq_before"] > 0,
        100.0 * baseline_df["improvement_sq"] / baseline_df["cd_sq_before"],
        np.nan,
    )

    if not baseline_df.empty:
        mean_row = pd.DataFrame(
            [
                {
                    "slice_a": "MEAN",
                    "slice_b": "MEAN",
                    "cd_sq_before": float(baseline_df["cd_sq_before"].mean()),
                    "cd_sq_after": float(baseline_df["cd_sq_after"].mean()),
                    "improvement_sq": float(baseline_df["improvement_sq"].mean()),
                    "improvement_sq_pct": float(baseline_df["improvement_sq_pct"].mean()),
                }
            ]
        )
        baseline_df = pd.concat([baseline_df, mean_row], ignore_index=True)

    baseline_file = os.path.join(result_dir, "chamfer_baseline.csv")
    baseline_df.to_csv(baseline_file, index=False)

    baseline_norm_df = _adjacent_chamfer_table(xy_input_norm, batch_values, order_use)[
        ["slice_a", "slice_b", "cd_sq_symmetric"]
    ].rename(columns={"cd_sq_symmetric": "cd_sq_before"}).merge(
        _adjacent_chamfer_table(xy_aligned_norm, batch_values, order_use)[["slice_a", "slice_b", "cd_sq_symmetric"]]
        .rename(columns={"cd_sq_symmetric": "cd_sq_after"}),
        on=["slice_a", "slice_b"],
        how="inner",
    )
    baseline_norm_df["improvement_sq"] = baseline_norm_df["cd_sq_before"] - baseline_norm_df["cd_sq_after"]
    baseline_norm_df["improvement_sq_pct"] = np.where(
        baseline_norm_df["cd_sq_before"] > 0,
        100.0 * baseline_norm_df["improvement_sq"] / baseline_norm_df["cd_sq_before"],
        np.nan,
    )
    baseline_norm_df["neg_log10_cd_sq_before"] = _neg_log10_safe(baseline_norm_df["cd_sq_before"])
    baseline_norm_df["neg_log10_cd_sq_after"] = _neg_log10_safe(baseline_norm_df["cd_sq_after"])

    if not baseline_norm_df.empty:
        mean_row = pd.DataFrame(
            [
                {
                    "slice_a": "MEAN",
                    "slice_b": "MEAN",
                    "cd_sq_before": float(baseline_norm_df["cd_sq_before"].mean()),
                    "cd_sq_after": float(baseline_norm_df["cd_sq_after"].mean()),
                    "improvement_sq": float(baseline_norm_df["improvement_sq"].mean()),
                    "improvement_sq_pct": float(baseline_norm_df["improvement_sq_pct"].mean()),
                    "neg_log10_cd_sq_before": float(baseline_norm_df["neg_log10_cd_sq_before"].mean()),
                    "neg_log10_cd_sq_after": float(baseline_norm_df["neg_log10_cd_sq_after"].mean()),
                }
            ]
        )
        baseline_norm_df = pd.concat([baseline_norm_df, mean_row], ignore_index=True)

    baseline_norm_file = os.path.join(result_dir, "chamfer_baseline_norm.csv")
    baseline_norm_df.to_csv(baseline_norm_file, index=False)

    corr_pair_df, corr_summary_df, pcc_dimension_df = _cross_slice_expression_corr_table(
        adata,
        coords_3d_key="rec_3d",
        batch_key="batch",
        hvg_fallback_top=2000,
    )
    corr_pair_before_df, _, _ = _cross_slice_expression_corr_table(
        adata,
        coords_3d_key="pre_3d",
        batch_key="batch",
        hvg_fallback_top=2000,
    )
    corr_pair_file = os.path.join(result_dir, "cross_slice_nearest_expr_corr_pairs.csv")
    corr_summary_file = os.path.join(result_dir, "cross_slice_nearest_expr_corr_summary.csv")
    pcc_dim_file = os.path.join(result_dir, "cross_slice_pcc_dimensions.csv")
    corr_pair_df.to_csv(corr_pair_file, index=False)
    corr_summary_df.to_csv(corr_summary_file, index=False)
    pcc_dimension_df.to_csv(pcc_dim_file, index=False)
    pcc_boxplot_file, pcc_meanplot_file = _plot_cross_slice_pcc_by_pair(corr_pair_df, result_dir)
    side_table_file, side_plot_file = _export_pcc_chamfer_side_by_side(
        corr_pair_before_df,
        corr_pair_df,
        baseline_norm_df,
        result_dir,
    )

    adata.write(final_file)
    print(f"Saved adjacent-slice Chamfer scores to: {chamfer_file}")
    print(f"Saved Chamfer baseline comparison to: {baseline_file}")
    print(f"Saved normalized adjacent-slice Chamfer scores to: {chamfer_norm_file}")
    print(f"Saved normalized Chamfer baseline comparison to: {baseline_norm_file}")
    print(f"Saved cross-slice nearest expression correlations to: {corr_pair_file}")
    print(f"Saved cross-slice correlation summary to: {corr_summary_file}")
    print(f"Saved PCC dimension summary to: {pcc_dim_file}")
    if pcc_boxplot_file:
        print(f"Saved PCC-by-pair boxplot to: {pcc_boxplot_file}")
    if pcc_meanplot_file:
        print(f"Saved PCC-by-pair mean bars to: {pcc_meanplot_file}")
    if side_table_file:
        print(f"Saved side-by-side PCC/Chamfer table to: {side_table_file}")
    if side_plot_file:
        print(f"Saved side-by-side PCC/Chamfer plot to: {side_plot_file}")
    print(f"Saved final AnnData to: {final_file}")


if __name__ == "__main__":
    main()
