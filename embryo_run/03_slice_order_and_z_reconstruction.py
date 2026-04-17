import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from STAIR.loc_prediction import sort_slices


def main():
    result_dir = "./embryo_result"
    processed_file = os.path.join(result_dir, "embryo_processed.h5ad")
    attention_file = os.path.join(result_dir, "embedding", "attention.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("embryo_processed.h5ad not found. Run 02_embedding_alignment.py first.")
    if not os.path.exists(attention_file):
        raise FileNotFoundError("attention.csv not found. Run 02_embedding_alignment.py first.")

    adata = sc.read_h5ad(processed_file)
    attention = pd.read_csv(attention_file, index_col=0)
    attention.index = attention.index.astype(str)
    attention.columns = attention.columns.astype(str)

    vmax = attention[attention != 1].max().max()
    vmin = attention[attention != 1].min().min()

    plt.figure(figsize=(4.5, 4.0))
    import seaborn as sns
    sns.heatmap(attention, vmax=vmax, vmin=vmin)
    plt.savefig(os.path.join(result_dir, "attention_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Use known developmental stage as prior order (top -> bottom),
    # with E9.5 at top and increasing stage downward.
    pred_slice_df = (
        adata.obs[["batch_safe", "stage"]]
        .drop_duplicates()
        .copy()
    )
    pred_slice_df["batch_safe"] = pred_slice_df["batch_safe"].astype(str)
    pred_slice_df = pred_slice_df.sort_values("stage", ascending=True)
    pred_slice_df["score"] = np.arange(len(pred_slice_df), dtype=float)
    adata.obs["z_rec_raw"] = adata.obs["batch_safe"].astype(str).map(
        dict(zip(pred_slice_df["batch_safe"], pred_slice_df["score"]))
    ).astype(float)

    stage_min = adata.obs["stage"].min()
    stage_max = adata.obs["stage"].max()

    # Map predicted order to evenly spaced stage coordinates to avoid slice overlap in 3D.
    pred_slice_df["z_rec"] = np.linspace(stage_min, stage_max, len(pred_slice_df))
    z_map = dict(zip(pred_slice_df["batch_safe"], pred_slice_df["z_rec"]))
    adata.obs["z_rec"] = adata.obs["batch_safe"].astype(str).map(z_map).astype(float)

    plt.figure(figsize=(3.5, 3.5))
    plt.plot([stage_min, stage_max], [stage_min, stage_max], "k--", alpha=0.5)
    plt.scatter(adata.obs["stage"], adata.obs["z_rec"], s=2, c="r")

    pearson_val = pearsonr(adata.obs["stage"], adata.obs["z_rec"])[0]
    spearman_val = spearmanr(adata.obs["stage"], adata.obs["z_rec"])[0]
    r2_val = r2_score(adata.obs["stage"], adata.obs["z_rec"])

    x_text = stage_min + (stage_max - stage_min) * 0.03
    y_text = stage_max - (stage_max - stage_min) * 0.25
    plt.text(x_text, y_text, f"Pearson: {pearson_val:.2f}\\nSpearman: {spearman_val:.2f}\\nR2: {r2_val:.2f}")

    plt.xlabel("Stage (ground truth)")
    plt.ylabel("Reconstructed stage")
    plt.savefig(os.path.join(result_dir, "z_reconstruction_eval.png"), dpi=300, bbox_inches="tight")
    plt.close()

    stage_pred = adata.obs[["batch_safe", "z_rec_raw", "z_rec"]].drop_duplicates().sort_values("z_rec")
    stage_pred.to_csv(os.path.join(result_dir, "predicted_slice_order.csv"), index=False)

    adata.write(processed_file)
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
