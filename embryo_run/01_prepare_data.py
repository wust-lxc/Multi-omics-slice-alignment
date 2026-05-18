import warnings
warnings.filterwarnings("ignore")

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

from STAIR.utils import set_seed


def main():
    set_seed(42)

    data_dir = "./data/Stereo_seq_mouse_embryo"
    result_dir = "./embryo_result"
    os.makedirs(result_dir, exist_ok=True)

    embryo_pattern = re.compile(r"^(E\d+(?:\.\d+)?)_.*\.h5ad$")
    detected = []
    for file_name in os.listdir(data_dir):
        match = embryo_pattern.match(file_name)
        if match is None:
            continue
        stage_raw = match.group(1)
        stage_value = float(stage_raw.replace("E", ""))
        detected.append((stage_value, stage_raw, file_name))

    detected.sort(key=lambda x: x[0])
    input_files = [(stage_raw, file_name) for _, stage_raw, file_name in detected]

    if len(input_files) == 0:
        raise FileNotFoundError("No embryo slice files found in ./data (expected names like E9.5_*.h5ad).")

    print("Detected embryo slices:", [name for _, name in input_files])

    adatas = []
    for stage_raw, file_name in input_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required input file: {file_path}")

        adata = sc.read_h5ad(file_path)
        if "spatial" not in adata.obsm:
            raise KeyError(f"'spatial' not found in {file_name}")

        stage_safe = stage_raw.replace(".", "p")
        stage_value = float(stage_raw.replace("E", ""))

        adata.obs["batch_raw"] = stage_raw
        adata.obs["batch_safe"] = stage_safe
        adata.obs["batch"] = stage_safe
        adata.obs["stage"] = stage_value

        adatas.append(adata)

    merged = ad.concat(adatas, join="inner", merge="same")

    batch_order = [stage.replace(".", "p") for stage, _ in input_files]
    merged.obs["batch_safe"] = merged.obs["batch_safe"].astype("category")
    merged.obs["batch_safe"] = merged.obs["batch_safe"].cat.set_categories(batch_order)

    merged.obs["batch"] = merged.obs["batch"].astype("category")
    merged.obs["batch"] = merged.obs["batch"].cat.set_categories(batch_order)

    merged_file = os.path.join(result_dir, "embryo_merged.h5ad")
    processed_file = os.path.join(result_dir, "embryo_processed.h5ad")

    print(f"Merged shape: cells={merged.n_obs}, genes={merged.n_vars}")
    merged.write(merged_file)
    # Avoid writing two full-size files in step 1 to reduce disk pressure.
    # Step 2 will refresh embryo_processed.h5ad after embedding/alignment outputs are created.
    if os.path.exists(processed_file):
        os.remove(processed_file)

    plt.figure(figsize=(6.5, 5))
    sc.pl.embedding(
        merged,
        basis="spatial",
        color="batch_safe",
        frameon=False,
        s=2,
        show=False,
        title="Embryo input spatial coordinates",
    )
    plt.savefig(os.path.join(result_dir, "input_spatial.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved merged data to: {merged_file}")
    print(f"Initialized merged data only (processed file will be created in step 2): {merged_file}")


if __name__ == "__main__":
    main()
