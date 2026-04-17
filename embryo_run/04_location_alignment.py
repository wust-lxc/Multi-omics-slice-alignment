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
    result_dir = "./embryo_result"
    location_dir = os.path.join(result_dir, "location")
    os.makedirs(location_dir, exist_ok=True)

    processed_file = os.path.join(result_dir, "embryo_processed.h5ad")
    if not os.path.exists(processed_file):
        raise FileNotFoundError("embryo_processed.h5ad not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata = sc.read_h5ad(processed_file)

    if "STAIR" not in adata.obsm:
        raise KeyError("STAIR embedding not found. Run 02_embedding_alignment.py first.")

    if "Domain" not in adata.obs:
        adata.obs["Domain"] = adata.obs["batch_safe"].astype(str)

    # Fixed prior order: align slices strictly by stage from top to bottom,
    # with E9.5 first.
    keys_order = (
        adata.obs[["batch_safe", "stage"]]
        .drop_duplicates()
        .sort_values("stage", ascending=True)["batch_safe"]
        .astype(str)
        .tolist()
    )

    loc_align = Loc_Align(
        adata,
        batch_key="batch_safe",
        batch_order=keys_order,
        result_path=result_dir,
    )

    loc_align.init_align(
        emb_key="STAIR",
        spatial_key="spatial",
        num_mnn=2,
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

    loc_align.plot_edge(spatial_key="transform_init", figsize=(6, 6), s=1.5)

    adata = loc_align.fine_align(max_iterations=160, tolerance=1e-10)

    plt.figure(figsize=(6.5, 5.5))
    sc.pl.embedding(
        adata,
        basis="transform_init",
        color=["batch_safe", "Domain"],
        frameon=False,
        ncols=2,
        s=6,
        show=False,
    )
    plt.savefig(os.path.join(location_dir, "alignment_init.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 5.5))
    sc.pl.embedding(
        adata,
        basis="transform_fine",
        color=["batch_safe", "Domain"],
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

    rms_init = float(np.sqrt(np.mean(np.sum((init_xy - orig) ** 2, axis=1))))
    rms_fine = float(np.sqrt(np.mean(np.sum((fine_xy - orig) ** 2, axis=1))))

    pd.DataFrame(
        {
            "metric": ["rms_displacement_vs_input"],
            "init": [rms_init],
            "fine": [rms_fine],
        }
    ).to_csv(os.path.join(location_dir, "alignment_displacement.csv"), index=False)

    adata.write(processed_file)
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
