import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


def main():
    # Fixed top-to-bottom order provided by user (no prediction).
    fixed_order = [
        "Mouse_Brain_ATAC",
        "Mouse_Brain_H3K27ac",
        "Mouse_Brain_H3K4me3",
        "Mouse_Brain_H3K27me3",
    ]

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Mouse_Brain_multiomics_result")

    processed_file = os.path.join(result_dir, "multiomics_processed.h5ad")
    attention_file = os.path.join(result_dir, "embedding", "attention.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("multiomics_processed.h5ad not found. Run 02_embedding_alignment.py first.")

    adata = sc.read_h5ad(processed_file)

    # Keep attention heatmap export if attention exists, but do not use it for ordering.
    if os.path.exists(attention_file):
        attention = pd.read_csv(attention_file, index_col=0)
        attention.index = attention.index.astype(str)
        attention.columns = attention.columns.astype(str)

        vmax = attention[attention != 1].max().max()
        vmin = attention[attention != 1].min().min()

        plt.figure(figsize=(4.8, 4.2))
        sns.heatmap(attention, vmax=vmax, vmin=vmin)
        plt.savefig(os.path.join(result_dir, "attention_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

    batches_present = set(adata.obs["batch"].astype(str).unique())
    fixed_present = [b for b in fixed_order if b in batches_present]
    missing = [b for b in fixed_order if b not in batches_present]
    extras = sorted(list(batches_present - set(fixed_order)))

    if len(fixed_present) == 0:
        raise ValueError("None of the fixed-order slice names were found in adata.obs['batch'].")

    # If unexpected slices exist, append them after user-defined order for robustness.
    final_order = fixed_present + extras

    order_df = pd.DataFrame(
        {
            "batch": final_order,
            "score": np.arange(len(final_order), dtype=float),
        }
    )

    if order_df.shape[0] > 1:
        order_df["z_rec"] = np.linspace(0.0, float(order_df.shape[0] - 1), order_df.shape[0])
    else:
        order_df["z_rec"] = 0.0

    adata.obs["z_rec_raw"] = adata.obs["batch"].astype(str).map(dict(zip(order_df["batch"], order_df["score"]))).astype(float)
    adata.obs["z_rec"] = adata.obs["batch"].astype(str).map(dict(zip(order_df["batch"], order_df["z_rec"]))).astype(float)

    plt.figure(figsize=(4.8, 3.8))
    plt.scatter(order_df["score"], order_df["z_rec"], s=40, c="#1f77b4")
    for _, row in order_df.iterrows():
        plt.text(row["score"], row["z_rec"], str(row["batch"]), fontsize=8)
    plt.xlabel("Fixed order index")
    plt.ylabel("z_rec")
    plt.title("Fixed slice order")
    plt.savefig(os.path.join(result_dir, "z_reconstruction_eval.png"), dpi=300, bbox_inches="tight")
    plt.close()

    order_df.to_csv(os.path.join(result_dir, "predicted_slice_order.csv"), index=False)

    adata.write(processed_file)
    if missing:
        print(f"Warning: missing fixed slices in data: {missing}")
    if extras:
        print(f"Warning: extra slices appended after fixed order: {extras}")
    print(f"Using fixed order: {final_order}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
