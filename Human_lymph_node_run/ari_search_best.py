import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

from STAIR.multi_emb_alignment import Multi_Emb_Align
from STAIR.utils import cluster_func, set_seed


@dataclass
class SearchConfig:
    name: str
    hvg_top: int = 4000
    ae_epoch: int = 200
    ae_batch_size: int = 256
    loss_weight_rna: float = 1.0
    loss_weight_atac: float = 10.0
    atac_loss: str = "mse"
    hgat_epoch: int = 200
    hgat_batches: int = 6
    sim_threshold: float = 0.3
    c_neigh_het: float = 0.35
    n_neigh_hom: int = 10
    gamma: float = 0.8
    mini_batch: bool = False
    n_hidden: int = 128
    n_latent: int = 32


def _eval_cluster_grid(adata: sc.AnnData, y_true: pd.Series) -> tuple[float, str, int, pd.DataFrame]:
    rows = []
    best_ari = -1.0
    best_method = ""
    best_k = -1

    for method in ("mclust", "kmeans"):
        for k in (8, 9, 10):
            adata_tmp = adata.copy()
            try:
                adata_tmp = cluster_func(
                    adata_tmp,
                    clustering=method,
                    use_rep="STAIR",
                    cluster_num=k,
                    key_add="TMP",
                )
                y_pred = adata_tmp.obs.loc[y_true.index, "TMP"].astype(str)
                ari = float(adjusted_rand_score(y_true, y_pred))
                rows.append({"method": method, "cluster_num": k, "ari": ari})
                if ari > best_ari:
                    best_ari = ari
                    best_method = method
                    best_k = k
            except Exception as exc:
                rows.append({"method": method, "cluster_num": k, "ari": np.nan, "error": str(exc)})

    return best_ari, best_method, best_k, pd.DataFrame(rows)


def _run_one_config(adata_raw: sc.AnnData, cfg: SearchConfig, result_dir: str) -> dict:
    adata = adata_raw.copy()

    hvg_top = min(cfg.hvg_top, adata.n_vars)
    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )

    emb_align = Multi_Emb_Align(
        adata,
        batch_key="batch",
        hvg=hvg_top,
        n_hidden=cfg.n_hidden,
        n_latent=cfg.n_latent,
        likelihood="nb",
        num_workers=0,
        result_path=result_dir,
        atac_key="ADT",
    )

    emb_align.prepare(count_key=None, lib_size="explog", normalize=True, scale=False)
    emb_align.preprocess(
        epoch_ae=cfg.ae_epoch,
        batch_size=cfg.ae_batch_size,
        loss_weight_rna=cfg.loss_weight_rna,
        loss_weight_atac=cfg.loss_weight_atac,
        atac_loss=cfg.atac_loss,
    )
    emb_align.latent()

    emb_align.prepare_hgat(
        spatial_key="spatial",
        slice_order=batch_order,
        n_neigh_hom=cfg.n_neigh_hom,
        c_neigh_het=cfg.c_neigh_het,
        sim_threshold=cfg.sim_threshold,
    )
    emb_align.train_hgat(
        gamma=cfg.gamma,
        mini_batch=cfg.mini_batch,
        epoch_hgat=cfg.hgat_epoch,
        batches=cfg.hgat_batches,
    )

    adata, attention = emb_align.predict_hgat(
        mini_batch=cfg.mini_batch,
        batches=cfg.hgat_batches,
    )

    valid = (~adata.obs["final_annot"].isna()).copy()
    y_true = adata.obs.loc[valid, "final_annot"].astype(str)

    best_ari, best_method, best_k, grid_df = _eval_cluster_grid(adata, y_true=y_true)

    # Set the best Domain labels back to adata for possible export.
    adata_best = cluster_func(
        adata.copy(),
        clustering=best_method,
        use_rep="STAIR",
        cluster_num=best_k,
        key_add="STAIR",
    )
    adata_best.obs["Domain"] = adata_best.obs["STAIR"].astype(str)

    return {
        "config": asdict(cfg),
        "best_ari": best_ari,
        "best_method": best_method,
        "best_cluster_num": best_k,
        "grid": grid_df,
        "adata_best": adata_best,
        "attention": attention,
    }


def main() -> None:
    set_seed(42)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Human_lymph_node_result")
    tuning_dir = os.path.join(result_dir, "ari_tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    merged_file = os.path.join(result_dir, "human_lymph_node_merged.h5ad")
    processed_file = os.path.join(result_dir, "human_lymph_node_processed.h5ad")
    best_h5ad = os.path.join(tuning_dir, "best_from_search.h5ad")
    summary_file = os.path.join(tuning_dir, "search_summary.csv")

    if not os.path.exists(merged_file):
        raise FileNotFoundError("human_lymph_node_merged.h5ad not found. Run 01_prepare_data.py first.")

    adata_raw = sc.read_h5ad(merged_file)
    adata_raw.obs_names_make_unique()
    if "final_annot" in adata_raw.obs.columns:
        adata_raw.obs["final_annot"] = (
            adata_raw.obs["final_annot"].astype(str).str.lower().str.replace("vessels", "vessel")
        )

    configs = [
        SearchConfig(name="baseline_repro", loss_weight_atac=10.0, hgat_epoch=200, sim_threshold=0.3, c_neigh_het=0.35, gamma=0.8),
        SearchConfig(name="balance_loss", loss_weight_atac=5.0, hgat_epoch=200, sim_threshold=0.3, c_neigh_het=0.35, gamma=0.8),
        SearchConfig(name="mild_crossslice", loss_weight_atac=7.0, hgat_epoch=220, sim_threshold=0.25, c_neigh_het=0.30, gamma=0.75),
        SearchConfig(name="soft_crossslice", loss_weight_atac=5.0, hgat_epoch=220, sim_threshold=0.25, c_neigh_het=0.30, gamma=0.8),
    ]

    summary_rows = []
    best_run = None

    for cfg in configs:
        print(f"[RUN] {cfg.name}")
        out = _run_one_config(adata_raw, cfg, result_dir=result_dir)

        grid_path = os.path.join(tuning_dir, f"grid_{cfg.name}.csv")
        out["grid"].to_csv(grid_path, index=False)

        row = {
            "name": cfg.name,
            "best_ari": out["best_ari"],
            "best_method": out["best_method"],
            "best_cluster_num": out["best_cluster_num"],
            "grid_path": grid_path,
            "config_json": json.dumps(out["config"], ensure_ascii=True),
        }
        summary_rows.append(row)

        if (best_run is None) or (out["best_ari"] > best_run["best_ari"]):
            best_run = {
                "name": cfg.name,
                "best_ari": out["best_ari"],
                "best_method": out["best_method"],
                "best_cluster_num": out["best_cluster_num"],
                "config": out["config"],
                "adata_best": out["adata_best"],
                "attention": out["attention"],
            }

    summary_df = pd.DataFrame(summary_rows).sort_values("best_ari", ascending=False)
    summary_df.to_csv(summary_file, index=False)

    if best_run is None:
        raise RuntimeError("No successful run found during ARI search.")

    # Export best artifacts and also overwrite processed file for downstream scripts.
    best_run["adata_best"].write(best_h5ad)
    best_run["adata_best"].write(processed_file)

    attention_file = os.path.join(result_dir, "embedding", "attention.csv")
    best_run["attention"].to_csv(attention_file)

    print("\n[RESULT] Best run summary")
    print(f"  name: {best_run['name']}")
    print(f"  best_ari: {best_run['best_ari']:.6f}")
    print(f"  best_method: {best_run['best_method']}")
    print(f"  best_cluster_num: {best_run['best_cluster_num']}")
    print(f"  summary_file: {summary_file}")
    print(f"  best_h5ad: {best_h5ad}")
    print(f"  processed_file(updated): {processed_file}")


if __name__ == "__main__":
    main()
