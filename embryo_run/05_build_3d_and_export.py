import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def main():
    result_dir = "./embryo_result"
    processed_file = os.path.join(result_dir, "embryo_processed.h5ad")
    final_file = os.path.join(result_dir, "adata.h5ad")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("embryo_processed.h5ad not found. Run previous scripts first.")

    adata = sc.read_h5ad(processed_file)

    if "transform_fine" not in adata.obsm:
        raise KeyError("transform_fine not found. Run 04_location_alignment.py first.")
    if "z_rec" not in adata.obs:
        raise KeyError("z_rec not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata.obs["x_aligned"] = adata.obsm["transform_fine"][:, 0]
    adata.obs["y_aligned"] = adata.obsm["transform_fine"][:, 1]

    adata.obsm["rec_3d"] = adata.obs[["x_aligned", "y_aligned", "z_rec"]].values
    # Plot-only axis orientation: flip z so earlier stage (E9.5) is shown on top.
    adata.obsm["rec_3d_plot"] = np.column_stack([
        adata.obs["x_aligned"].values,
        adata.obs["y_aligned"].values,
        -adata.obs["z_rec"].values,
    ])

    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        ref_xy = adata.obs[["x", "y"]].values
    else:
        ref_xy = adata.obsm["spatial"]

    ref_xyz = ref_xy.copy()
    if ref_xyz.shape[1] > 2:
        ref_xyz = ref_xyz[:, :2]
    adata.obsm["gt_3d_stage"] = np.column_stack([ref_xyz, adata.obs["stage"].values])
    adata.obsm["gt_3d_stage_plot"] = np.column_stack([ref_xyz, -adata.obs["stage"].values])

    color_key = "Domain" if "Domain" in adata.obs else "batch_safe"

    plt.figure(figsize=(5.5, 5.0))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Embryo reconstructed 3D",
    )
    plt.savefig(os.path.join(result_dir, "reconstruction_3d_rec.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.5, 5.0))
    sc.pl.embedding(
        adata,
        basis="gt_3d_stage_plot",
        projection="3d",
        color=color_key,
        s=2,
        show=False,
        title="Embryo reference 3D (x,y,stage)",
    )
    plt.savefig(os.path.join(result_dir, "reconstruction_3d_reference.png"), dpi=300, bbox_inches="tight")
    plt.close()

    adata.write(final_file)
    print(f"Saved final AnnData to: {final_file}")


if __name__ == "__main__":
    main()
