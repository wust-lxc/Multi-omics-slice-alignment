import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from STAIR.embedding.dataset_hgat import calcu_adaptive_hyperedge
from STAIR.embedding.module_hgat import HyperGAT_pyg
from STAIR.multi_emb_alignment import Multi_Emb_Align
from STAIR.utils import cluster_func, set_seed


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


def _train_single_slice_hypergat(
    adata,
    source_key: str = "latent",
    target_key: str = "STAIR_spatial",
    spatial_key: str = "spatial",
    n_neigh: int = 12,
    sim_threshold: float = 0.20,
    epoch: int = 180,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    dropout: float = 0.15,
    residual_weight: float = 0.65,
    device: str | None = None,
) -> None:
    if source_key not in adata.obsm:
        raise KeyError(f"{source_key!r} not found in adata.obsm.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} not found in adata.obsm.")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    x_np = np.asarray(adata.obsm[source_key], dtype=np.float32)
    coord_np = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    hyperedge_index = calcu_adaptive_hyperedge(
        coord_np,
        x_np,
        n_neigh=n_neigh,
        sim_threshold=sim_threshold,
    ).to(device)

    x = torch.from_numpy(x_np).float().to(device)
    model = HyperGAT_pyg(latent_dim=x_np.shape[1], dropout_gat=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []
    for _ in tqdm(range(epoch), desc="single-slice HyperGAT"):
        model.train()
        xbar, _ = model(x, hyperedge_index)
        loss = F.mse_loss(xbar, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        xbar, _ = model(x, hyperedge_index)
        out = residual_weight * xbar + (1.0 - residual_weight) * x

    adata.obsm[target_key] = out.detach().cpu().numpy().astype(np.float32)
    adata.uns[f"{target_key}_source"] = source_key
    adata.uns[f"{target_key}_hyperedge_n_neigh"] = int(n_neigh)
    adata.uns[f"{target_key}_hyperedge_sim_threshold"] = float(sim_threshold)
    adata.uns[f"{target_key}_epoch"] = int(epoch)
    adata.uns[f"{target_key}_residual_weight"] = float(residual_weight)
    adata.uns[f"{target_key}_loss_final"] = float(loss_history[-1]) if loss_history else np.nan


def main():
    set_seed(42)

    hvg_top = int(os.environ.get("STAIR_H3K27AC_HVG_TOP", "3000"))
    ae_epoch = 120
    ae_batch_size = 256
    loss_weight_rna = 1.0
    loss_weight_epi = 5.0
    epi_loss = "mse"
    hypergat_epoch = 180
    hypergat_n_neigh = 12
    hypergat_sim_threshold = 0.70
    hypergat_residual_weight = 0.65

    cluster_num = 18
    mclust_source_key = "STAIR_spatial"
    mclust_rep_key = "STAIR_mclust"
    mclust_pca_components = 8
    mclust_model_name = "EVE"

    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac"
    processed_file = result_dir / "h3k27ac_processed.h5ad"
    if not processed_file.exists():
        raise FileNotFoundError("h3k27ac_processed.h5ad not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(processed_file)
    adata.obs_names_make_unique()
    hvg_top = min(hvg_top, adata.n_vars)

    emb_align = Multi_Emb_Align(
        adata,
        batch_key=None,
        hvg=hvg_top,
        n_hidden=128,
        n_latent=32,
        likelihood="nb",
        num_workers=0,
        result_path=str(result_dir),
        atac_key="EPI",
        encode_batch=False,
        decode_batch=False,
    )

    emb_align.prepare(count_key=None, lib_size="explog", normalize=True, scale=False)
    emb_align.preprocess(
        epoch_ae=ae_epoch,
        batch_size=ae_batch_size,
        loss_weight_rna=loss_weight_rna,
        loss_weight_atac=loss_weight_epi,
        atac_loss=epi_loss,
    )
    adata = emb_align.latent(return_data=True)
    adata.obsm["STAIR_ae"] = adata.obsm["latent"].copy()
    _train_single_slice_hypergat(
        adata,
        source_key="latent",
        target_key="STAIR_spatial",
        spatial_key="spatial",
        n_neigh=hypergat_n_neigh,
        sim_threshold=hypergat_sim_threshold,
        epoch=hypergat_epoch,
        residual_weight=hypergat_residual_weight,
    )
    adata.obsm["STAIR"] = adata.obsm["STAIR_spatial"].copy()

    adata = _add_scaled_pca_rep(
        adata,
        source_key=mclust_source_key,
        target_key=mclust_rep_key,
        n_components=mclust_pca_components,
        random_state=2022,
    )
    adata = cluster_func(
        adata,
        clustering="mclust",
        use_rep=mclust_rep_key,
        cluster_num=cluster_num,
        modelNames=mclust_model_name,
        key_add="STAIR",
    )
    adata.obs["Domain_mclust_global"] = adata.obs["STAIR"].astype(str)
    adata.obs["Domain"] = adata.obs["Domain_mclust_global"].astype(str)
    domain_refinement = f"none_pure_global_mclust_g{cluster_num}_{mclust_rep_key}"
    adata.obs["cluster_method"] = "mclust"
    adata.uns["cluster_method"] = "mclust"
    adata.uns["domain_refinement"] = domain_refinement
    adata.obs["domain_refinement"] = domain_refinement
    adata.uns["mclust_cluster_num"] = int(cluster_num)
    adata.uns["mclust_rep_key"] = mclust_rep_key
    adata.uns["mclust_source_key"] = mclust_source_key
    adata.uns["mclust_pca_components"] = int(mclust_pca_components)
    adata.uns["mclust_model_name"] = mclust_model_name
    adata.uns["single_slice_spatial_graph"] = "adaptive_hypergraph"
    adata.uns["h3k27ac_loss_weight"] = float(loss_weight_epi)
    adata.uns["h3k27ac_feature_transform"] = str(adata.uns.get("EPI_transform", "positive_99pct"))

    sc.pp.neighbors(adata, use_rep="STAIR")
    sc.tl.umap(adata, min_dist=0.2)

    adata.write(processed_file)

    print("Clustering method: mclust")
    print(f"mclust clusters: G={cluster_num}")
    print(f"Domain refinement: {domain_refinement}")
    print(f"Single-slice spatial graph: adaptive hypergraph, n_neigh={hypergat_n_neigh}, sim_threshold={hypergat_sim_threshold}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
