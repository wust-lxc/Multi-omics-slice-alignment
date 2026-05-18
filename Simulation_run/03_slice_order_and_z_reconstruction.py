import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc


def main():
    fixed_order = [f"Simulation{i}" for i in range(1, 6)]

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Simulation_result")

    processed_file = os.path.join(result_dir, "simulation_processed.h5ad")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("simulation_processed.h5ad not found. Run 02_embedding_alignment.py first.")

    adata = sc.read_h5ad(processed_file)

    batches_present = set(adata.obs["batch"].astype(str).unique())
    fixed_present = [b for b in fixed_order if b in batches_present]
    missing = [b for b in fixed_order if b not in batches_present]
    extras = sorted(list(batches_present - set(fixed_order)))

    if len(fixed_present) == 0:
        raise ValueError("None of the fixed-order slice names were found in adata.obs['batch'].")

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

    adata.obs["z_rec_raw"] = adata.obs["batch"].astype(str).map(
        dict(zip(order_df["batch"], order_df["score"]))
    ).astype(float)
    adata.obs["z_rec"] = adata.obs["batch"].astype(str).map(
        dict(zip(order_df["batch"], order_df["z_rec"]))
    ).astype(float)

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
