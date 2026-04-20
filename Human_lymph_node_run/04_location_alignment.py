import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from STAIR.loc_alignment import Loc_Align


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Human_lymph_node_result")
    location_dir = os.path.join(result_dir, "location")
    os.makedirs(location_dir, exist_ok=True)

    processed_file = os.path.join(result_dir, "human_lymph_node_processed.h5ad")
    order_file = os.path.join(result_dir, "predicted_slice_order.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("human_lymph_node_processed.h5ad not found. Run 03_slice_order_and_z_reconstruction.py first.")
    if not os.path.exists(order_file):
        raise FileNotFoundError("predicted_slice_order.csv not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata = sc.read_h5ad(processed_file)
    if "STAIR" not in adata.obsm:
        raise KeyError("STAIR embedding not found. Run 02_embedding_alignment.py first.")

    if "Domain" not in adata.obs:
        adata.obs["Domain"] = adata.obs["batch"].astype(str)

    order_df = pd.read_csv(order_file)
    keys_order = order_df.sort_values("z_rec")["batch"].astype(str).tolist()

    loc_align = Loc_Align(
        adata,
        batch_key="batch",
        batch_order=keys_order,
        result_path=result_dir,
    )

    loc_align.init_align(
        emb_key="STAIR",
        spatial_key="spatial",
        num_mnn=2,
        use_scale=True,
    )

    try:
        loc_align.detect_fine_points(
            domain_key="Domain",
            slice_boundary=True,
            domain_boundary=True,
            num_domains=2,
            alpha=80,
            return_result=False,
        )
    except Exception as exc:
        print("Domain boundary detection failed, fallback to slice-only edges.")
        print(f"Detail: {exc}")
        loc_align.detect_fine_points(
            domain_key="Domain",
            slice_boundary=True,
            domain_boundary=False,
            alpha=80,
            return_result=False,
        )

    # Save edge visualizations under location/edge.
    loc_align.plot_edge(spatial_key="transform_init", figsize=(6, 6), s=1.5)

    adata = loc_align.fine_align(max_iterations=160, tolerance=1e-10)

    plt.figure(figsize=(6.8, 5.6))
    sc.pl.embedding(
        adata,
        basis="transform_init",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        s=6,
        show=False,
    )
    plt.savefig(os.path.join(location_dir, "alignment_init.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.8, 5.6))
    sc.pl.embedding(
        adata,
        basis="transform_fine",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        s=6,
        show=False,
    )
    plt.savefig(os.path.join(location_dir, "alignment_fine.png"), dpi=300, bbox_inches="tight")
    plt.close()

    orig = adata.obsm["spatial"]
    init_xy = adata.obsm["transform_init"]
    fine_xy = adata.obsm["transform_fine"]

    adata.uns["alignment_rms_init"] = float(np.sqrt(np.mean(np.sum((init_xy - orig) ** 2, axis=1))))
    adata.uns["alignment_rms_fine"] = float(np.sqrt(np.mean(np.sum((fine_xy - orig) ** 2, axis=1))))

    pd.DataFrame(
        {
            "metric": ["rms_displacement_vs_input"],
            "init": [float(adata.uns["alignment_rms_init"])],
            "fine": [float(adata.uns["alignment_rms_fine"])],
        }
    ).to_csv(os.path.join(location_dir, "alignment_displacement.csv"), index=False)

    adata.write(processed_file)
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
