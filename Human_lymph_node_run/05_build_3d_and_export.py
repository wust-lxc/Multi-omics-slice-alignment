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


def _adjacent_chamfer(points: np.ndarray, batches: np.ndarray, order_use):
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
        p = points[idx]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        span = p_max - p_min
        span[span <= 1e-12] = 1.0
        out[idx] = (p - p_min) / span
    return out


def _pick_hvg_mask(adata, top_fallback=2000):
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

    var = np.nan_to_num(x2_mean - x_mean * x_mean, nan=0.0, posinf=0.0, neginf=0.0)
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


def _pairwise_pearson_for_pairs(matrix, pairs: np.ndarray) -> np.ndarray:
    out = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
    for i, (idx_a, idx_b) in enumerate(pairs):
        vec_a = _row_as_dense(matrix, int(idx_a))
        vec_b = _row_as_dense(matrix, int(idx_b))
        out[i] = _pearson_corr(vec_a, vec_b)
    return out


def _cross_slice_corr_pairs(adata, coords_3d_key: str, batch_key: str = "batch", hvg_top: int = 2000) -> pd.DataFrame:
    coords = np.asarray(adata.obsm[coords_3d_key], dtype=np.float64)
    batches = adata.obs[batch_key].astype(str).values
    pairs = _collect_cross_slice_nearest_pairs(coords_3d=coords, batches=batches)

    hvg_mask = _pick_hvg_mask(adata, top_fallback=hvg_top)
    x_hvg = adata[:, hvg_mask].X
    pearson_rna = _pairwise_pearson_for_pairs(x_hvg, pairs)

    if "ADT" in adata.obsm:
        pearson_adt = _pairwise_pearson_for_pairs(adata.obsm["ADT"], pairs)
    else:
        pearson_adt = np.full((pairs.shape[0],), np.nan, dtype=np.float64)

    if "STAIR" in adata.obsm:
        pearson_stair = _pairwise_pearson_for_pairs(adata.obsm["STAIR"], pairs)
    else:
        pearson_stair = np.full((pairs.shape[0],), np.nan, dtype=np.float64)

    return pd.DataFrame(
        {
            "slice_a": [str(batches[int(i)]) for i in pairs[:, 0]] if pairs.size else [],
            "slice_b": [str(batches[int(i)]) for i in pairs[:, 1]] if pairs.size else [],
            "pearson_rna": pearson_rna,
            "pearson_adt": pearson_adt,
            "pearson_stair": pearson_stair,
        }
    )


def _export_pcc_chamfer_side_by_side(
    corr_before: pd.DataFrame,
    corr_after: pd.DataFrame,
    baseline_norm_df: pd.DataFrame,
    result_dir: str,
) -> tuple[str, str]:
    if corr_before.empty or corr_after.empty or baseline_norm_df.empty:
        return "", ""

    b = corr_before.copy()
    a = corr_after.copy()
    b["slice_pair"] = b.apply(lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])), axis=1)
    a["slice_pair"] = a.apply(lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])), axis=1)

    b_mean = b.groupby("slice_pair", as_index=False)[["pearson_rna", "pearson_adt", "pearson_stair"]].mean()
    b_mean = b_mean.rename(
        columns={
            "pearson_rna": "RNA PCC before",
            "pearson_adt": "ADT PCC before",
            "pearson_stair": "Latent PCC before",
        }
    )
    a_mean = a.groupby("slice_pair", as_index=False)[["pearson_rna", "pearson_adt", "pearson_stair"]].mean()
    a_mean = a_mean.rename(
        columns={
            "pearson_rna": "RNA PCC after",
            "pearson_adt": "ADT PCC after",
            "pearson_stair": "Latent PCC after",
        }
    )
    pcc = b_mean.merge(a_mean, on="slice_pair", how="inner")

    bn = baseline_norm_df.copy()
    bn = bn[bn["slice_a"] != "MEAN"]
    bn["slice_pair"] = bn.apply(lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])), axis=1)

    side = pcc.merge(
        bn[["slice_pair", "cd_sq_before_norm", "cd_sq_after_norm"]],
        on="slice_pair",
        how="inner",
    )
    if side.empty:
        return "", ""

    side["neg_log10_cd_sq_before_norm"] = -np.log10(np.clip(side["cd_sq_before_norm"].to_numpy(dtype=float), 1e-12, None))
    side["neg_log10_cd_sq_after_norm"] = -np.log10(np.clip(side["cd_sq_after_norm"].to_numpy(dtype=float), 1e-12, None))

    side_csv = os.path.join(result_dir, "pcc_chamfer_side_by_side.csv")
    side.to_csv(side_csv, index=False)

    x = np.arange(side.shape[0])
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))

    w = 0.12
    axs[0].bar(x - 2.5 * w, side["RNA PCC before"], width=w, label="RNA before", color="#9ecae1")
    axs[0].bar(x - 1.5 * w, side["RNA PCC after"], width=w, label="RNA after", color="#3182bd")
    axs[0].bar(x - 0.5 * w, side["ADT PCC before"], width=w, label="ADT before", color="#fdae6b")
    axs[0].bar(x + 0.5 * w, side["ADT PCC after"], width=w, label="ADT after", color="#e6550d")
    axs[0].bar(x + 1.5 * w, side["Latent PCC before"], width=w, label="Latent before", color="#a1d99b")
    axs[0].bar(x + 2.5 * w, side["Latent PCC after"], width=w, label="Latent after", color="#31a354")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(side["slice_pair"].tolist(), rotation=20, ha="right")
    axs[0].set_ylabel("Mean PCC")
    axs[0].set_title("PCC by Slice Pair (Before vs After)")
    axs[0].grid(axis="y", alpha=0.25)
    axs[0].legend(frameon=False, fontsize=8)

    w2 = 0.32
    axs[1].bar(x - w2 / 2, side["neg_log10_cd_sq_before_norm"], width=w2, label="Before", color="#9ecae1")
    axs[1].bar(x + w2 / 2, side["neg_log10_cd_sq_after_norm"], width=w2, label="After", color="#08519c")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(side["slice_pair"].tolist(), rotation=20, ha="right")
    axs[1].set_ylabel("-log10(CD_sq_norm)")
    axs[1].set_title("Chamfer by Slice Pair (Before vs After)")
    axs[1].grid(axis="y", alpha=0.25)
    axs[1].legend(frameon=False)

    plt.tight_layout()
    side_png = os.path.join(result_dir, "pcc_chamfer_side_by_side.png")
    plt.savefig(side_png, dpi=300, bbox_inches="tight")
    plt.close()
    return side_csv, side_png


def _moran_i_knn(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
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


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Human_lymph_node_result")

    processed_file = os.path.join(result_dir, "human_lymph_node_processed.h5ad")
    final_file = os.path.join(result_dir, "adata.h5ad")
    metric_file = os.path.join(result_dir, "metrics_summary.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("human_lymph_node_processed.h5ad not found. Run previous scripts first.")

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

    batches_plot = adata.obs["batch"].astype(str).values
    xy_aligned_plot = _minmax_normalize_by_slice(adata.obsm["transform_fine"], batches_plot)
    xy_input_plot = _minmax_normalize_by_slice(adata.obsm["spatial"], batches_plot)

    adata.obsm["rec_3d_plot"] = np.column_stack([
        xy_aligned_plot[:, 0],
        xy_aligned_plot[:, 1],
        -adata.obs["z_rec"].values,
    ])
    adata.obsm["gt_3d_order_plot"] = np.column_stack([
        xy_input_plot[:, 0],
        xy_input_plot[:, 1],
        -adata.obs["z_rec"].values,
    ])

    color_key = "Domain" if "Domain" in adata.obs else "batch"
    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Human lymph node reconstructed 3D",
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
        title="Human lymph node reference 3D (x,y,z_rec)",
    )
    plt.savefig(os.path.join(result_dir, "reconstruction_3d_reference.png"), dpi=300, bbox_inches="tight")
    plt.close()

    batches = adata.obs["batch"].astype(str).values
    order_use = (
        adata.obs[["batch", "z_rec"]]
        .drop_duplicates()
        .sort_values("z_rec")["batch"]
        .astype(str)
        .tolist()
    )

    xy_aligned = adata.obsm["transform_fine"]
    xy_input = adata.obsm["spatial"]

    # Raw and normalized adjacent-slice Chamfer before/after.
    chamfer_before = _adjacent_chamfer(xy_input, batches, order_use)
    chamfer_after = _adjacent_chamfer(xy_aligned, batches, order_use)

    xy_aligned_norm = _minmax_normalize_by_slice(xy_aligned, batches)
    xy_input_norm = _minmax_normalize_by_slice(xy_input, batches)
    chamfer_before_norm = _adjacent_chamfer(xy_input_norm, batches, order_use)
    chamfer_after_norm = _adjacent_chamfer(xy_aligned_norm, batches, order_use)

    coords_3d = np.asarray(adata.obsm["rec_3d"], dtype=np.float64)
    pairs = _collect_cross_slice_nearest_pairs(coords_3d=coords_3d, batches=batches)

    hvg_mask = _pick_hvg_mask(adata, top_fallback=2000)
    x_hvg = adata[:, hvg_mask].X
    pearson_rna = _pairwise_pearson_for_pairs(x_hvg, pairs)

    if "ADT" in adata.obsm:
        pearson_adt = _pairwise_pearson_for_pairs(adata.obsm["ADT"], pairs)
        n_adt = int(np.asarray(adata.obsm["ADT"]).shape[1])
    else:
        pearson_adt = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
        n_adt = 0

    if "STAIR" in adata.obsm:
        pearson_stair = _pairwise_pearson_for_pairs(adata.obsm["STAIR"], pairs)
        n_stair = int(np.asarray(adata.obsm["STAIR"]).shape[1])
    else:
        pearson_stair = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
        n_stair = 0

    rows = []

    rows.append(
        {
            "metric_group": "basic",
            "metric_name": "n_cells",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(adata.n_obs),
            "extra": "",
        }
    )
    rows.append(
        {
            "metric_group": "basic",
            "metric_name": "n_genes",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(adata.n_vars),
            "extra": "",
        }
    )

    for s in order_use:
        n_s = int(np.sum(batches == s))
        rows.append(
            {
                "metric_group": "basic",
                "metric_name": "n_cells_per_slice",
                "slice": s,
                "slice_pair": "",
                "value": float(n_s),
                "extra": "",
            }
        )

    if "alignment_rms_init" in adata.uns:
        rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_init_vs_input",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(adata.uns["alignment_rms_init"]),
                "extra": "",
            }
        )
    if "alignment_rms_fine" in adata.uns:
        rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_fine_vs_input",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(adata.uns["alignment_rms_fine"]),
                "extra": "",
            }
        )

    if not chamfer_before.empty and not chamfer_after.empty:
        merged = chamfer_before[["slice_a", "slice_b", "cd_sq_symmetric"]].rename(
            columns={"cd_sq_symmetric": "cd_sq_before"}
        ).merge(
            chamfer_after[["slice_a", "slice_b", "cd_sq_symmetric"]].rename(columns={"cd_sq_symmetric": "cd_sq_after"}),
            on=["slice_a", "slice_b"],
            how="inner",
        )
        merged["improvement_pct"] = np.where(
            merged["cd_sq_before"] > 0,
            100.0 * (merged["cd_sq_before"] - merged["cd_sq_after"]) / merged["cd_sq_before"],
            np.nan,
        )

        for _, r in merged.iterrows():
            sp = f"{r['slice_a']}|{r['slice_b']}"
            rows.append({"metric_group": "chamfer_raw", "metric_name": "cd_sq_before", "slice": "", "slice_pair": sp, "value": float(r["cd_sq_before"]), "extra": ""})
            rows.append({"metric_group": "chamfer_raw", "metric_name": "cd_sq_after", "slice": "", "slice_pair": sp, "value": float(r["cd_sq_after"]), "extra": ""})
            rows.append({"metric_group": "chamfer_raw", "metric_name": "improvement_pct", "slice": "", "slice_pair": sp, "value": float(r["improvement_pct"]), "extra": ""})

        rows.append({"metric_group": "chamfer_raw", "metric_name": "cd_sq_before_mean", "slice": "ALL", "slice_pair": "MEAN", "value": float(merged["cd_sq_before"].mean()), "extra": ""})
        rows.append({"metric_group": "chamfer_raw", "metric_name": "cd_sq_after_mean", "slice": "ALL", "slice_pair": "MEAN", "value": float(merged["cd_sq_after"].mean()), "extra": ""})
        rows.append({"metric_group": "chamfer_raw", "metric_name": "improvement_pct_mean", "slice": "ALL", "slice_pair": "MEAN", "value": float(merged["improvement_pct"].mean()), "extra": ""})

    if not chamfer_before_norm.empty and not chamfer_after_norm.empty:
        merged_n = chamfer_before_norm[["slice_a", "slice_b", "cd_sq_symmetric"]].rename(
            columns={"cd_sq_symmetric": "cd_sq_before_norm"}
        ).merge(
            chamfer_after_norm[["slice_a", "slice_b", "cd_sq_symmetric"]].rename(columns={"cd_sq_symmetric": "cd_sq_after_norm"}),
            on=["slice_a", "slice_b"],
            how="inner",
        )
        merged_n["improvement_pct_norm"] = np.where(
            merged_n["cd_sq_before_norm"] > 0,
            100.0 * (merged_n["cd_sq_before_norm"] - merged_n["cd_sq_after_norm"]) / merged_n["cd_sq_before_norm"],
            np.nan,
        )

        for _, r in merged_n.iterrows():
            sp = f"{r['slice_a']}|{r['slice_b']}"
            rows.append({"metric_group": "chamfer_norm", "metric_name": "cd_sq_before_norm", "slice": "", "slice_pair": sp, "value": float(r["cd_sq_before_norm"]), "extra": ""})
            rows.append({"metric_group": "chamfer_norm", "metric_name": "cd_sq_after_norm", "slice": "", "slice_pair": sp, "value": float(r["cd_sq_after_norm"]), "extra": ""})
            rows.append({"metric_group": "chamfer_norm", "metric_name": "improvement_pct_norm", "slice": "", "slice_pair": sp, "value": float(r["improvement_pct_norm"]), "extra": ""})

        rows.append({"metric_group": "chamfer_norm", "metric_name": "cd_sq_before_norm_mean", "slice": "ALL", "slice_pair": "MEAN", "value": float(merged_n["cd_sq_before_norm"].mean()), "extra": ""})
        rows.append({"metric_group": "chamfer_norm", "metric_name": "cd_sq_after_norm_mean", "slice": "ALL", "slice_pair": "MEAN", "value": float(merged_n["cd_sq_after_norm"].mean()), "extra": ""})
        rows.append({"metric_group": "chamfer_norm", "metric_name": "improvement_pct_norm_mean", "slice": "ALL", "slice_pair": "MEAN", "value": float(merged_n["improvement_pct_norm"].mean()), "extra": ""})
    else:
        merged_n = pd.DataFrame()

    rows.append(
        {
            "metric_group": "correlation",
            "metric_name": "cross_slice_pair_count",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(pairs.shape[0]),
            "extra": "",
        }
    )
    rows.append(
        {
            "metric_group": "correlation",
            "metric_name": "pearson_rna_mean",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(np.nanmean(pearson_rna)) if pearson_rna.size else np.nan,
            "extra": f"n_hvg={int(hvg_mask.sum())}",
        }
    )
    rows.append(
        {
            "metric_group": "correlation",
            "metric_name": "pearson_adt_mean",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(np.nanmean(pearson_adt)) if pearson_adt.size else np.nan,
            "extra": f"n_adt={n_adt}",
        }
    )
    rows.append(
        {
            "metric_group": "correlation",
            "metric_name": "pearson_stair_mean",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(np.nanmean(pearson_stair)) if pearson_stair.size else np.nan,
            "extra": f"n_stair={n_stair}",
        }
    )

    for s in order_use:
        idx = adata.obs["batch"].astype(str).values == s
        coords_s = np.asarray(adata.obsm["spatial"])[idx]
        codes_s = pd.Categorical(adata.obs.loc[idx, "Domain"].astype(str)).codes.astype(np.float64)
        rows.append(
            {
                "metric_group": "moran",
                "metric_name": "moran_i_domain_knn",
                "slice": str(s),
                "slice_pair": "",
                "value": float(_moran_i_knn(codes_s, coords_s, k=6)),
                "extra": f"k=6;n_cells={int(idx.sum())}",
            }
        )

    cluster_method = "unknown"
    if "cluster_method" in adata.obs.columns:
        vals = adata.obs["cluster_method"].astype(str).unique().tolist()
        if len(vals) > 0:
            cluster_method = vals[0]
    elif "cluster_method" in adata.uns:
        cluster_method = str(adata.uns.get("cluster_method", "unknown"))
    rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "cluster_method",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": np.nan,
            "extra": cluster_method,
        }
    )
    rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "n_domain_clusters",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(adata.obs["Domain"].astype(str).nunique()) if "Domain" in adata.obs else np.nan,
            "extra": "",
        }
    )

    # Spatial domains visualization by slice.
    if "spatial" in adata.obsm:
        n_slice = len(order_use)
        n_col = min(3, max(1, n_slice))
        n_row = int(np.ceil(n_slice / n_col))
        fig, axs = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 4.0 * n_row), constrained_layout=True)
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
                    show_legend = idx == n_slice - 1
                    sc.pl.embedding(
                        adata_tmp,
                        basis="spatial",
                        color=color_key,
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

    corr_after = _cross_slice_corr_pairs(adata, coords_3d_key="rec_3d", batch_key="batch", hvg_top=2000)
    corr_before = _cross_slice_corr_pairs(adata, coords_3d_key="pre_3d", batch_key="batch", hvg_top=2000)
    side_csv, side_png = _export_pcc_chamfer_side_by_side(
        corr_before=corr_before,
        corr_after=corr_after,
        baseline_norm_df=merged_n,
        result_dir=result_dir,
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metric_file, index=False)

    adata.write(final_file)

    print(f"Saved all metrics to one file: {metric_file}")
    if side_csv:
        print(f"Saved side-by-side PCC/Chamfer table to: {side_csv}")
    if side_png:
        print(f"Saved side-by-side PCC/Chamfer plot to: {side_png}")
    print(f"Saved final AnnData to: {final_file}")


if __name__ == "__main__":
    main()
