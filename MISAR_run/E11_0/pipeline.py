import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from STAIR.data_paths import resolve_data_root
from STAIR.loc_alignment import Loc_Align
from STAIR.multi_emb_alignment import Multi_Emb_Align
from STAIR.utils import cluster_func, set_seed


TIMEPOINT = "E11_0"
SLICE_NAMES = ["E11_0-S1", "E11_0-S2"]
LSI_COMPONENTS = 50
PEAK_BIN_SIZE = 10000

HVG_TOP = 3000
AE_EPOCH = 120
AE_BATCH_SIZE = 256
LOSS_WEIGHT_RNA = 1.0
LOSS_WEIGHT_EPI = 5.0

HGAT_EPOCH = 120
HGAT_GAMMA = 0.85
HGAT_DROPOUT_HOM = 0.25
HGAT_DROPOUT_HET = 0.25
SIM_THRESHOLD = 0.25
C_NEIGH_HET = 0.35
N_NEIGH_HOM = 10
N_NEIGH_HET = 30

# Global mclust G for the merged two-slice embedding.
# Use the larger per-slice Combined_Clusters count to keep shared domains across slices.
CLUSTER_NUM = 13
MCLUST_SOURCE_KEY = "STAIR_bc"
MCLUST_REP_KEY = "STAIR_mclust"
MCLUST_PCA_COMPONENTS = 10
MCLUST_MODEL_NAME = "EEV"
SPATIAL_2D_DOT_SIZE = 140
SPATIAL_2D_FIG_WIDTH = 10.5
SPATIAL_2D_FIG_HEIGHT_PER_SLICE = 4.6


def configure_runtime_threads() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def result_dir() -> Path:
    return repo_root() / "MISAR_result" / TIMEPOINT


def merged_file() -> Path:
    return result_dir() / f"misar_{TIMEPOINT}_merged.h5ad"


def processed_file() -> Path:
    return result_dir() / f"misar_{TIMEPOINT}_processed.h5ad"


def _to_positive_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.nanmin(x, axis=0, keepdims=True)
    scale = np.nanpercentile(x, 99, axis=0, keepdims=True)
    scale[scale <= 0] = 1.0
    x = x / scale
    x = np.clip(x, 0.0, None)
    x = x + 1e-4
    return x.astype(np.float32)


def _tfidf_lsi_matrix(x, n_components: int = 50) -> np.ndarray:
    if not sparse.issparse(x):
        x = sparse.csr_matrix(np.asarray(x, dtype=np.float32))
    else:
        x = x.tocsr().astype(np.float32)
    x.data = np.nan_to_num(x.data, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = np.asarray(x.sum(axis=1)).ravel()
    row_scale = np.divide(1.0, row_sum, out=np.zeros_like(row_sum, dtype=np.float32), where=row_sum > 0)
    tf = sparse.diags(row_scale).dot(x)

    df = np.asarray((x > 0).sum(axis=0)).ravel()
    idf = np.log1p(x.shape[0] / (1.0 + df)).astype(np.float32)
    tfidf = tf.multiply(idf)

    n_components = int(min(n_components, x.shape[0] - 1, x.shape[1] - 1))
    if n_components < 2:
        raise ValueError("Not enough cells/features to compute LSI.")
    lsi = TruncatedSVD(n_components=n_components, random_state=2022).fit_transform(tfidf)
    lsi = StandardScaler().fit_transform(lsi)
    return _to_positive_features(lsi)


def _parse_peak_interval(peak_name: str) -> tuple[str, int, int] | None:
    parts = str(peak_name).split(".")
    if len(parts) < 3:
        return None
    chrom = parts[0]
    start_part = parts[1]
    end_part = parts[2]
    if not start_part.isdigit():
        return None
    end_digits = []
    for ch in end_part:
        if ch.isdigit():
            end_digits.append(ch)
        else:
            break
    if not end_digits:
        return None
    start = int(start_part)
    end = int("".join(end_digits))
    if end <= start:
        return None
    return chrom, start, end


def _bin_sort_key(label: str) -> tuple:
    chrom, span = label.split(":", 1)
    start = int(span.split("-", 1)[0])
    chrom_body = chrom[3:] if chrom.startswith("chr") else chrom
    if chrom_body.isdigit():
        chrom_key = (0, int(chrom_body))
    else:
        chrom_key = (1, chrom_body)
    return chrom_key, start


def _peak_to_bin_matrix(adata_peak, bin_size: int) -> tuple[sparse.csr_matrix, list[str], int]:
    x = adata_peak.X
    if not sparse.issparse(x):
        x = sparse.csr_matrix(np.asarray(x, dtype=np.float32))
    else:
        x = x.tocsr().astype(np.float32)
    x.data = np.nan_to_num(x.data, nan=0.0, posinf=0.0, neginf=0.0)

    label_to_col: dict[str, int] = {}
    labels: list[str] = []
    kept_peak_indices: list[int] = []
    peak_to_bin_col: list[int] = []
    for peak_idx, peak_name in enumerate(adata_peak.var_names.astype(str)):
        parsed = _parse_peak_interval(peak_name)
        if parsed is None:
            continue
        chrom, start, end = parsed
        midpoint = (start + end) // 2
        bin_start = (midpoint // bin_size) * bin_size
        label = f"{chrom}:{bin_start}-{bin_start + bin_size}"
        if label not in label_to_col:
            label_to_col[label] = len(labels)
            labels.append(label)
        kept_peak_indices.append(peak_idx)
        peak_to_bin_col.append(label_to_col[label])

    if not kept_peak_indices:
        raise ValueError("No parseable genomic peak names found.")

    x = x[:, kept_peak_indices]
    peak_to_bin_col_arr = np.asarray(peak_to_bin_col, dtype=np.int64)
    coo = x.tocoo()
    binned = sparse.coo_matrix(
        (coo.data, (coo.row, peak_to_bin_col_arr[coo.col])),
        shape=(x.shape[0], len(labels)),
        dtype=np.float32,
    ).tocsr()
    binned.sum_duplicates()
    return binned, labels, len(kept_peak_indices)


def _joint_peak_bin_lsi(
    bin_matrices: list[sparse.csr_matrix],
    bin_labels_by_slice: list[list[str]],
    n_components: int,
) -> tuple[list[np.ndarray], str, int]:
    global_labels = sorted(set().union(*[set(labels) for labels in bin_labels_by_slice]), key=_bin_sort_key)
    label_to_global = {label: idx for idx, label in enumerate(global_labels)}
    global_mats = []
    for x, labels in zip(bin_matrices, bin_labels_by_slice):
        local_to_global = np.asarray([label_to_global[label] for label in labels], dtype=np.int64)
        coo = x.tocoo()
        global_x = sparse.coo_matrix(
            (coo.data, (coo.row, local_to_global[coo.col])),
            shape=(x.shape[0], len(global_labels)),
            dtype=np.float32,
        ).tocsr()
        global_x.sum_duplicates()
        global_mats.append(global_x)

    x_all = sparse.vstack(global_mats, format="csr")
    lsi_all = _tfidf_lsi_matrix(x_all, n_components=n_components)
    split_points = np.cumsum([x.shape[0] for x in global_mats])[:-1]
    epi_by_slice = [arr.astype(np.float32) for arr in np.split(lsi_all, split_points, axis=0)]
    source = f"peak_midpoint_bin_{PEAK_BIN_SIZE}_joint_tfidf_lsi_{lsi_all.shape[1]}"
    return epi_by_slice, source, len(global_labels)


def _common_genes() -> list[str]:
    gene_sets = []
    data_root = resolve_data_root(repo_root()) / "MISAR"
    for slice_name in SLICE_NAMES:
        rna_file = data_root / slice_name / "adata_RNA.h5ad"
        if not rna_file.exists():
            raise FileNotFoundError(f"Missing RNA file: {rna_file}")
        adata_rna = sc.read_h5ad(rna_file)
        adata_rna.var_names_make_unique()
        gene_sets.append(set(map(str, adata_rna.var_names)))
    common = sorted(set.intersection(*gene_sets))
    if len(common) == 0:
        raise ValueError(f"No common genes found for {TIMEPOINT}.")
    return common


def prepare_data() -> None:
    configure_runtime_threads()
    set_seed(42)

    data_root = resolve_data_root(repo_root()) / "MISAR"
    out_dir = result_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    common_genes = _common_genes()
    raw_slices = []
    bin_matrices = []
    bin_labels_by_slice = []
    peak_summary = []

    for order, slice_name in enumerate(SLICE_NAMES):
        slice_dir = data_root / slice_name
        rna_file = slice_dir / "adata_RNA.h5ad"
        peak_file = slice_dir / "adata_Peak.h5ad"
        if not peak_file.exists():
            raise FileNotFoundError(f"Missing Peak file: {peak_file}")

        adata_rna = sc.read_h5ad(rna_file)
        adata_peak = sc.read_h5ad(peak_file)
        adata_rna.var_names_make_unique()
        adata_peak.var_names_make_unique()

        common_obs = adata_rna.obs_names.intersection(adata_peak.obs_names)
        if len(common_obs) == 0:
            raise ValueError(f"No overlapping barcodes for {slice_name}.")

        adata_rna = adata_rna[common_obs, common_genes].copy()
        adata_peak = adata_peak[common_obs].copy()

        if "spatial" not in adata_rna.obsm:
            if "spatial" not in adata_peak.obsm:
                raise KeyError(f"No spatial coordinates found for {slice_name}.")
            adata_rna.obsm["spatial"] = np.asarray(adata_peak.obsm["spatial"])

        adata_rna.obs["original_barcode"] = adata_rna.obs_names.astype(str)
        adata_rna.obs_names = [f"{slice_name}_{bc}" for bc in adata_rna.obs_names.astype(str)]
        adata_rna.obs_names_make_unique()
        adata_rna.var_names_make_unique()

        adata_rna.obs["batch"] = slice_name
        adata_rna.obs["slice_order"] = int(order)
        adata_rna.obs["epigenomic_assay"] = "Peak"

        if "Combined_Clusters" in adata_rna.obs:
            adata_rna.obs["truth"] = adata_rna.obs["Combined_Clusters"].astype(str)
            adata_rna.obs["truth_prefixed"] = slice_name + ":" + adata_rna.obs["truth"].astype(str)
            adata_rna.obs["Domain_input"] = adata_rna.obs["truth"].astype(str)
        if "Combined_Clusters_annotation" in adata_rna.obs:
            adata_rna.obs["truth_annotation"] = adata_rna.obs["Combined_Clusters_annotation"].astype(str)

        bin_x, bin_labels, n_parseable_peaks = _peak_to_bin_matrix(adata_peak, bin_size=PEAK_BIN_SIZE)
        raw_slices.append(adata_rna)
        bin_matrices.append(bin_x)
        bin_labels_by_slice.append(bin_labels)
        peak_summary.append(
            {
                "slice": slice_name,
                "n_peaks": int(adata_peak.n_vars),
                "n_parseable_peaks": int(n_parseable_peaks),
                "n_bins_local": int(len(bin_labels)),
            }
        )

    epi_by_slice, epi_source, n_bins_global = _joint_peak_bin_lsi(
        bin_matrices,
        bin_labels_by_slice,
        n_components=LSI_COMPONENTS,
    )

    adata_list = []
    summary_rows = []
    for adata_rna, epi_feat, peak_info in zip(raw_slices, epi_by_slice, peak_summary):
        adata_rna.obsm["EPI"] = epi_feat
        adata_list.append(adata_rna)
        summary_rows.append(
            {
                "slice": str(adata_rna.obs["batch"].iloc[0]),
                "n_cells": int(adata_rna.n_obs),
                "n_genes_common": int(adata_rna.n_vars),
                "epi_dim": int(adata_rna.obsm["EPI"].shape[1]),
                "epi_source": epi_source,
                "n_peaks": peak_info["n_peaks"],
                "n_parseable_peaks": peak_info["n_parseable_peaks"],
                "n_bins_local": peak_info["n_bins_local"],
                "n_bins_global": int(n_bins_global),
                "truth_key": "Combined_Clusters" if "Combined_Clusters" in adata_rna.obs else "",
            }
        )

    merged = ad.concat(adata_list, join="inner", merge="same")
    merged.obs["batch"] = merged.obs["batch"].astype("category")
    merged.obs["batch"] = merged.obs["batch"].cat.set_categories(SLICE_NAMES)
    merged.uns["timepoint"] = TIMEPOINT
    merged.uns["slice_order"] = SLICE_NAMES
    merged.uns["EPI_transform"] = epi_source
    merged.uns["peak_bin_size"] = int(PEAK_BIN_SIZE)
    merged.uns["peak_n_bins_global"] = int(n_bins_global)

    merged.write(merged_file())
    merged.write(processed_file())
    pd.DataFrame(summary_rows).to_csv(out_dir / "input_summary.csv", index=False)

    print(f"Prepared MISAR {TIMEPOINT}: cells={merged.n_obs}, genes={merged.n_vars}")
    print(f"Saved merged data to: {merged_file()}")
    print(f"Initialized processed data to: {processed_file()}")


def _add_scaled_pca_rep(adata, source_key: str, target_key: str, n_components: int, random_state: int = 2022):
    if source_key not in adata.obsm:
        raise KeyError(f"{source_key!r} not found in adata.obsm.")
    x = np.asarray(adata.obsm[source_key], dtype=np.float64)
    x = StandardScaler().fit_transform(x)
    n_components = int(min(n_components, x.shape[1], x.shape[0] - 1))
    if n_components < 1:
        raise ValueError("n_components must be at least 1 after shape adjustment.")
    adata.obsm[target_key] = PCA(n_components=n_components, random_state=random_state).fit_transform(x)
    adata.uns[f"{target_key}_source"] = source_key
    adata.uns[f"{target_key}_n_components"] = int(n_components)
    return adata


def embedding_alignment() -> None:
    configure_runtime_threads()
    set_seed(42)

    out_dir = result_dir()
    embedding_dir = out_dir / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    if not merged_file().exists():
        raise FileNotFoundError(f"{merged_file()} not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(merged_file())
    adata.obs_names_make_unique()
    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )
    print("Detected slice order:", batch_order)

    if CLUSTER_NUM is None:
        cluster_num = int(adata.obs["truth"].dropna().astype(str).nunique()) if "truth" in adata.obs else 18
    else:
        cluster_num = int(CLUSTER_NUM)

    emb_align = Multi_Emb_Align(
        adata,
        batch_key="batch",
        hvg=min(HVG_TOP, adata.n_vars),
        n_hidden=128,
        n_latent=32,
        likelihood="nb",
        num_workers=0,
        result_path=str(out_dir),
        atac_key="EPI",
        encode_batch=False,
        decode_batch=True,
    )

    emb_align.prepare(count_key=None, lib_size="explog", normalize=True, scale=False)
    emb_align.preprocess(
        epoch_ae=AE_EPOCH,
        batch_size=AE_BATCH_SIZE,
        loss_weight_rna=LOSS_WEIGHT_RNA,
        loss_weight_atac=LOSS_WEIGHT_EPI,
        atac_loss="mse",
    )
    emb_align.latent()
    emb_align.batch_center_obsm(source_key="latent", target_key="latent_bc", batch_key="batch")
    emb_align.prepare_hgat(
        spatial_key="spatial",
        feat_key="latent_bc",
        slice_order=batch_order,
        n_neigh_hom=N_NEIGH_HOM,
        n_neigh_het=N_NEIGH_HET,
        c_neigh_het=C_NEIGH_HET,
        sim_threshold=SIM_THRESHOLD,
    )

    set_seed(42)
    emb_align.train_hgat(
        gamma=HGAT_GAMMA,
        mini_batch=False,
        epoch_hgat=HGAT_EPOCH,
        batches=6,
        dropout_hom=HGAT_DROPOUT_HOM,
        dropout_het=HGAT_DROPOUT_HET,
    )

    adata, attention = emb_align.predict_hgat(mini_batch=False, batches=6)
    attention.to_csv(embedding_dir / "attention.csv")
    emb_align.batch_center_obsm(source_key="STAIR", target_key="STAIR_bc", batch_key="batch")
    adata = emb_align.adata
    adata = _add_scaled_pca_rep(
        adata,
        source_key=MCLUST_SOURCE_KEY,
        target_key=MCLUST_REP_KEY,
        n_components=MCLUST_PCA_COMPONENTS,
        random_state=2022,
    )
    adata = cluster_func(
        adata,
        clustering="mclust",
        use_rep=MCLUST_REP_KEY,
        cluster_num=cluster_num,
        modelNames=MCLUST_MODEL_NAME,
        key_add="STAIR",
    )
    adata.obs["Domain_mclust_global"] = adata.obs["STAIR"].astype(str)
    adata.obs["Domain"] = adata.obs["Domain_mclust_global"].astype(str)
    adata.obs["cluster_method"] = "mclust"
    adata.uns["cluster_method"] = "mclust"
    adata.uns["mclust_rep_key"] = MCLUST_REP_KEY
    adata.uns["mclust_source_key"] = MCLUST_SOURCE_KEY
    adata.uns["mclust_pca_components"] = int(MCLUST_PCA_COMPONENTS)
    adata.uns["mclust_model_name"] = MCLUST_MODEL_NAME
    adata.uns["mclust_cluster_num"] = int(cluster_num)
    adata.write(processed_file())

    print("Clustering method: mclust")
    print(f"mclust clusters: G={cluster_num}")
    print(f"Updated processed data: {processed_file()}")


def z_reconstruction() -> None:
    if not processed_file().exists():
        raise FileNotFoundError(f"{processed_file()} not found. Run 02_embedding_alignment.py first.")
    adata = sc.read_h5ad(processed_file())
    batches_present = set(adata.obs["batch"].astype(str).unique())
    final_order = [b for b in SLICE_NAMES if b in batches_present] + sorted(list(batches_present - set(SLICE_NAMES)))
    if len(final_order) == 0:
        raise ValueError("No slices found in adata.obs['batch'].")

    z_map = dict(zip(final_order, np.linspace(0.0, float(len(final_order) - 1), len(final_order))))
    score_map = dict(zip(final_order, np.arange(len(final_order), dtype=float)))
    adata.obs["z_rec_raw"] = adata.obs["batch"].astype(str).map(score_map).astype(float)
    adata.obs["z_rec"] = adata.obs["batch"].astype(str).map(z_map).astype(float)
    adata.write(processed_file())

    print(f"Using fixed order: {final_order}")
    print(f"Updated processed data: {processed_file()}")


def location_alignment() -> None:
    configure_runtime_threads()
    out_dir = result_dir()
    location_dir = out_dir / "location"
    location_dir.mkdir(parents=True, exist_ok=True)
    if not processed_file().exists():
        raise FileNotFoundError(f"{processed_file()} not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata = sc.read_h5ad(processed_file())
    emb_key = "STAIR_bc" if "STAIR_bc" in adata.obsm else "STAIR"
    if emb_key not in adata.obsm:
        raise KeyError("STAIR embedding not found. Run 02_embedding_alignment.py first.")
    if "Domain" not in adata.obs:
        adata.obs["Domain"] = adata.obs["batch"].astype(str)

    uns_keep = {
        key: adata.uns[key]
        for key in [
            "cluster_method",
            "mclust_rep_key",
            "mclust_source_key",
            "mclust_pca_components",
            "mclust_model_name",
            "mclust_cluster_num",
        ]
        if key in adata.uns
    }
    keys_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )

    loc_align = Loc_Align(adata, batch_key="batch", batch_order=keys_order, result_path=str(out_dir))
    loc_align.init_align(emb_key=emb_key, spatial_key="spatial", num_mnn=10, use_scale=False)
    loc_align.detect_fine_points(domain_key="Domain", slice_boundary=True, domain_boundary=False, alpha=45, return_result=False)
    loc_align.plot_edge(spatial_key="transform_init", figsize=(6, 6), s=1.5)
    adata = loc_align.fine_align(max_iterations=160, tolerance=1e-10)
    adata.uns.update(uns_keep)

    for basis, filename in [("transform_init", "alignment_init.png"), ("transform_fine", "alignment_fine.png")]:
        plt.figure(figsize=(7.0, 3.8))
        sc.pl.embedding(adata, basis=basis, color=["batch", "Domain"], frameon=False, ncols=2, s=8, show=False)
        plt.savefig(location_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    orig = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    init_xy = np.asarray(adata.obsm["transform_init"], dtype=np.float64)
    fine_xy = np.asarray(adata.obsm["transform_fine"], dtype=np.float64)
    adata.uns["alignment_rms_init"] = float(np.sqrt(np.mean(np.sum((init_xy - orig) ** 2, axis=1))))
    adata.uns["alignment_rms_fine"] = float(np.sqrt(np.mean(np.sum((fine_xy - orig) ** 2, axis=1))))
    adata.write(processed_file())

    print(f"Embedding key used for location alignment: {emb_key}")
    print(f"Updated processed data: {processed_file()}")


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
    return float(np.sum(z[:, None] * z[neigh_idx] * (1.0 / float(k_use))) / denom)


def _add_ari_row(metrics_rows: list[dict], adata, slice_name: str, truth_key: str = "truth", pred_key: str = "Domain") -> float | None:
    if truth_key not in adata.obs or pred_key not in adata.obs:
        return None
    if slice_name == "ALL":
        idx = pd.Series(True, index=adata.obs_names)
        truth_use = "truth_prefixed" if "truth_prefixed" in adata.obs else truth_key
    else:
        idx = adata.obs["batch"].astype(str) == slice_name
        truth_use = truth_key
    valid = idx & ~adata.obs[truth_use].isna() & ~adata.obs[pred_key].isna()
    if int(valid.sum()) == 0:
        return None
    y_true = adata.obs.loc[valid, truth_use].astype(str)
    y_pred = adata.obs.loc[valid, pred_key].astype(str)
    value = float(adjusted_rand_score(y_true, y_pred))
    metrics_rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "ARI",
            "slice": slice_name,
            "slice_pair": "|".join(SLICE_NAMES),
            "value": value,
            "extra": (
                f"truth_key={truth_use};pred_key={pred_key};n_valid={int(valid.sum())};"
                f"n_truth={int(y_true.nunique())};n_pred={int(y_pred.nunique())}"
            ),
        }
    )
    return value


def _add_moran_row(metrics_rows: list[dict], adata, slice_name: str, coord_key: str, pred_key: str = "Domain", k: int = 6) -> float | None:
    if pred_key not in adata.obs or coord_key not in adata.obsm:
        return None
    if slice_name == "ALL":
        idx = np.ones(adata.n_obs, dtype=bool)
    else:
        idx = adata.obs["batch"].astype(str).values == slice_name
    if int(idx.sum()) < 3:
        return None
    labels = adata.obs.loc[idx, pred_key].astype(str)
    codes = pd.Categorical(labels).codes.astype(np.float64)
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)[idx]
    value = float(_moran_i_knn(codes, coords, k=k))
    metrics_rows.append(
        {
            "metric_group": "moran",
            "metric_name": "Moran's I",
            "slice": slice_name,
            "slice_pair": "|".join(SLICE_NAMES),
            "value": value,
            "extra": f"source=moran_i_domain_knn;coords={coord_key};k={k};n_cells={int(idx.sum())}",
        }
    )
    return value


def _plot_spatial_comparison(adata, out_dir: Path, truth_key: str = "truth", pred_key: str = "Domain") -> Path | None:
    if truth_key not in adata.obs or pred_key not in adata.obs or "spatial" not in adata.obsm:
        return None
    adata_plot = adata.copy()
    adata_plot.obs[truth_key] = adata_plot.obs[truth_key].astype("category")
    adata_plot.obs[pred_key] = adata_plot.obs[pred_key].astype("category")
    slices = (
        adata_plot.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )
    fig, axes = plt.subplots(
        nrows=len(slices),
        ncols=2,
        figsize=(SPATIAL_2D_FIG_WIDTH, SPATIAL_2D_FIG_HEIGHT_PER_SLICE * len(slices)),
    )
    axes = np.array(axes).reshape(len(slices), 2)
    for i, slice_id in enumerate(slices):
        adata_sub = adata_plot[adata_plot.obs["batch"].astype(str) == slice_id].copy()
        sc.pl.embedding(
            adata_sub,
            basis="spatial",
            color=truth_key,
            ax=axes[i, 0],
            show=False,
            title=f"Slice: {slice_id} | Ground Truth",
            frameon=False,
            size=SPATIAL_2D_DOT_SIZE,
            legend_fontsize=10,
        )
        sc.pl.embedding(
            adata_sub,
            basis="spatial",
            color=pred_key,
            ax=axes[i, 1],
            show=False,
            title=f"Slice: {slice_id} | STAIR Domain",
            frameon=False,
            size=SPATIAL_2D_DOT_SIZE,
            legend_fontsize=10,
        )
    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.8)
    output_path = out_dir / "spatial_comparison_2D.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def export_results() -> None:
    configure_runtime_threads()
    out_dir = result_dir()
    final_file = out_dir / "adata.h5ad"
    chamfer_file = out_dir / "chamfer_distance.csv"
    metric_file = out_dir / "metrics_summary.csv"
    if not processed_file().exists():
        raise FileNotFoundError(f"{processed_file()} not found. Run previous steps first.")

    adata = sc.read_h5ad(processed_file())
    if "transform_init" not in adata.obsm or "transform_fine" not in adata.obsm:
        raise KeyError("transform_init/transform_fine missing. Run 04_location_alignment.py first.")
    batches = adata.obs["batch"].astype(str).values
    slice_a, slice_b = SLICE_NAMES[0], SLICE_NAMES[1]

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

    z_rec = adata.obs["z_rec"].astype(float).values if "z_rec" in adata.obs else np.zeros(adata.n_obs)
    adata.obsm["rec_3d"] = np.column_stack([np.asarray(adata.obsm["transform_fine"], dtype=np.float64), z_rec])
    xy_aligned_plot = _minmax_normalize_by_slice(adata.obsm["transform_fine"], batches)
    xy_input_plot = _minmax_normalize_by_slice(adata.obsm["spatial"], batches)
    adata.obsm["rec_3d_plot"] = np.column_stack([xy_aligned_plot[:, 0], xy_aligned_plot[:, 1], -z_rec])
    adata.obsm["gt_3d_order_plot"] = np.column_stack([xy_input_plot[:, 0], xy_input_plot[:, 1], -z_rec])

    color_key = "Domain" if "Domain" in adata.obs else "batch"
    spatial_comparison_file = _plot_spatial_comparison(adata, out_dir)

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=4,
        show=False,
        title=f"MISAR {TIMEPOINT} reconstructed 3D",
    )
    plt.savefig(out_dir / "reconstruction_3d_rec.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="gt_3d_order_plot",
        projection="3d",
        color=color_key,
        s=4,
        show=False,
        title=f"MISAR {TIMEPOINT} reference 3D",
    )
    plt.savefig(out_dir / "reconstruction_3d_reference.png", dpi=300, bbox_inches="tight")
    plt.close()

    slice_pair = "|".join(SLICE_NAMES)
    metrics_rows = [
        {"metric_group": "basic", "metric_name": "n_cells", "slice": "ALL", "slice_pair": slice_pair, "value": float(adata.n_obs), "extra": ""},
        {"metric_group": "basic", "metric_name": "n_genes", "slice": "ALL", "slice_pair": slice_pair, "value": float(adata.n_vars), "extra": ""},
    ]
    for _, row in chamfer_df.iterrows():
        metrics_rows.append(
            {
                "metric_group": "chamfer",
                "metric_name": str(row["stage"]),
                "slice": "",
                "slice_pair": slice_pair,
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
                "slice_pair": slice_pair,
                "value": float(adata.obs["Domain"].astype(str).nunique()),
                "extra": "",
            }
        )
        _add_ari_row(metrics_rows, adata, slice_name="ALL", truth_key="truth", pred_key="Domain")
        ari_values = []
        for slice_name in SLICE_NAMES:
            ari_value = _add_ari_row(metrics_rows, adata, slice_name=slice_name, truth_key="truth", pred_key="Domain")
            if ari_value is not None and np.isfinite(ari_value):
                ari_values.append(ari_value)
        if ari_values:
            metrics_rows.append(
                {
                    "metric_group": "clustering",
                    "metric_name": "ARI_mean_by_slice",
                    "slice": "MEAN",
                    "slice_pair": slice_pair,
                    "value": float(np.mean(ari_values)),
                    "extra": f"aggregation=unweighted_slice_mean;n_slices={len(ari_values)}",
                }
            )
        _add_moran_row(metrics_rows, adata, slice_name="ALL", coord_key="rec_3d", pred_key="Domain", k=6)
        moran_values = []
        for slice_name in SLICE_NAMES:
            moran_value = _add_moran_row(metrics_rows, adata, slice_name=slice_name, coord_key="spatial", pred_key="Domain", k=6)
            if moran_value is not None and np.isfinite(moran_value):
                moran_values.append(moran_value)
        if moran_values:
            metrics_rows.append(
                {
                    "metric_group": "moran",
                    "metric_name": "Moran's I_mean_by_slice",
                    "slice": "MEAN",
                    "slice_pair": slice_pair,
                    "value": float(np.mean(moran_values)),
                    "extra": f"aggregation=unweighted_slice_mean;n_slices={len(moran_values)}",
                }
            )
    metrics_rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "cluster_method",
            "slice": "ALL",
            "slice_pair": slice_pair,
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
                "slice_pair": slice_pair,
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
                "slice_pair": slice_pair,
                "value": float(adata.uns["alignment_rms_fine"]),
                "extra": "",
            }
        )

    pd.DataFrame(metrics_rows).to_csv(metric_file, index=False)
    adata.write(final_file)

    print(f"Saved Chamfer distances to: {chamfer_file}")
    print(f"Saved metrics summary to: {metric_file}")
    print(f"Saved final AnnData to: {final_file}")
    if spatial_comparison_file is not None:
        print(f"Saved 2D spatial comparison plot to: {spatial_comparison_file}")


def run_all(run_dir: Path) -> int:
    configure_runtime_threads()
    steps = [
        run_dir / "01_prepare_data.py",
        run_dir / "02_embedding_alignment.py",
        run_dir / "03_slice_order_and_z_reconstruction.py",
        run_dir / "04_location_alignment.py",
        run_dir / "05_export_results.py",
    ]
    for step in steps:
        if not step.exists():
            raise FileNotFoundError(f"Missing pipeline step: {step}")
    try:
        for step in steps:
            print(f"[RUN] {step}")
            subprocess.run([sys.executable, str(step)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline failed at: {exc.cmd}")
        return exc.returncode
    print(f"MISAR {TIMEPOINT} pipeline finished.")
    return 0
