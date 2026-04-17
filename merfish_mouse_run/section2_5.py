import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
# 服务器端必须设置 Agg 后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import scanpy as sc
import pandas as pd
import numpy as np

# ==============================================================================
# 0. 环境与数据准备
# ==============================================================================

result_path = './merfish_mouse_result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# 读取上一步保存的累积文件
processed_file_path = os.path.join(result_path, 'hypothalamic_preoptic_processed.h5ad')

if os.path.exists(processed_file_path):
    print(f"Loading processed data from: {processed_file_path}")
    adata = sc.read(processed_file_path)
else:
    raise FileNotFoundError(f"Processed file not found at {processed_file_path}. Please run previous sections first.")

# ==============================================================================
# 1. 构建 3D 重构坐标 (rec_3d)
# ==============================================================================
print("Constructing 3D coordinates...")

# 检查必要信息
if 'transform_fine' not in adata.obsm:
    raise KeyError("'transform_fine' (aligned coordinates) not found. Please check Section 2.4")
if 'z_rec' not in adata.obs:
    raise KeyError("'z_rec' (reconstructed Z) not found. Please check Section 2.3")

# 将对齐后的 X, Y 提取到 obs 中，以便按照您的代码逻辑使用
# transform_fine 是 [N_cells, 2] 的矩阵
adata.obs['x_aligned'] = adata.obsm['transform_fine'][:, 0]
adata.obs['y_aligned'] = adata.obsm['transform_fine'][:, 1]

# 构建 rec_3d (X_aligned, Y_aligned, Z_rec)
adata.obsm['rec_3d'] = adata.obs[['x_aligned', 'y_aligned', 'z_rec']].values

# ==============================================================================
# 2. 3D 可视化 (De novo vs Ground Truth)
# ==============================================================================
print("Plotting 3D Reconstructions...")

# 设置颜色 (如果之前步骤已经定义了 Domain_colors，这里会自动沿用)
# 如果 Domain 列不存在，做一个简单的容错
if 'Domain' not in adata.obs:
    adata.obs['Domain'] = 'stHY' # Dummy

# 图 1: De novo reconstructed 3D coordinates
save_path_rec3d = os.path.join(result_path, 'reconstruction_3d_denovo.png')
with rc_context({'figure.figsize': (5,5)}):
    # 注意：projection='3d' 在 scanpy 的某些版本/backend 下可能表现不同
    # 这里我们尝试标准调用
    ax = sc.pl.embedding(
        adata,
        basis="rec_3d",
        na_color=(1, 1, 1, 0),
        projection="3d",
        color="Domain",
        title='De novo reconstructed 3D coordinates',
        show=False
    )
    plt.savefig(save_path_rec3d, dpi=300, bbox_inches='tight')
    plt.close()
print(f"Saved De novo 3D plot to: {save_path_rec3d}")

# 图 2: Ground truth 3D coordinates
# 检查是否存在 spatial_3d_μm
gt_key = "spatial_3d_μm"
if gt_key in adata.obsm:
    save_path_gt3d = os.path.join(result_path, 'reconstruction_3d_groundtruth.png')
    with rc_context({'figure.figsize': (5,5)}):
        sc.pl.embedding(
            adata,
            basis=gt_key,
            na_color=(1, 1, 1, 0),
            projection="3d",
            color="Domain",
            title='Ground truth 3D coordinates',
            show=False
        )
        plt.savefig(save_path_gt3d, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Saved Ground Truth 3D plot to: {save_path_gt3d}")
else:
    print(f"Warning: '{gt_key}' not found. Skipping GT 3D plot.")

# ==============================================================================
# 3. 保存最终结果
# ==============================================================================
final_file_path = os.path.join(result_path, 'adata.h5ad')
print(f"Saving final AnnData object to: {final_file_path}")
adata.write(final_file_path)