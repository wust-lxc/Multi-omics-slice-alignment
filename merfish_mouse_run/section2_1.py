import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
# 设置后端为 Agg，专门用于服务器端无 GUI 环境生成图片
matplotlib.use('Agg') 

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

from STAIR.emb_alignment import Emb_Align
from STAIR.utils import *

# ==============================================================================
# 0. 准备输出目录
# ==============================================================================
output_dir = './merfish_mouse_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# ==============================================================================
# 1. 加载数据
# ==============================================================================
print("Loading data...")
adata = sc.read('./data/hypothalamic_preoptic.h5ad')

# ==============================================================================
# 2. 保存 3D Ground Truth 图片
# ==============================================================================
print("Plotting and saving 3D Ground Truth...")
from matplotlib.pyplot import rc_context
with rc_context({'figure.figsize': (5,5)}):
    # 检查是否存在 spatial_3d_μm
    basis_key = "spatial_3d_μm" if "spatial_3d_μm" in adata.obsm else "spatial"
    
    # 注意：scanpy 的 show=False 允许后续保存图片
    sc.pl.embedding(
        adata,
        basis=basis_key,
        na_color=(1, 1, 1, 0),
        projection="3d",
        s=0.1,
        color="cell_type" if "cell_type" in adata.obs else "batch",
        title='Ground truth 3D coordinates',
        show=False 
    )
    # 保存图片
    save_path_3d = os.path.join(output_dir, 'ground_truth_3d.png')
    plt.savefig(save_path_3d, dpi=300, bbox_inches='tight')
    plt.close() # 关闭当前图形，释放内存
    print(f"Saved: {save_path_3d}")

# ==============================================================================
# 3. 保存 2D 对比图片 (Ground truth vs Rotated)
# ==============================================================================
print("Plotting and saving 2D comparison...")
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)

[axs[i].axis('off') for i in range(2)]

# 确定旋转后的 key
if 'spatial_2d_μm_rotate' not in adata.obsm:
    print("提示: 未找到 'spatial_2d_μm_rotate'，仅做演示，使用 'spatial' 代替。")
    rotate_key = 'spatial' 
else:
    rotate_key = 'spatial_2d_μm_rotate'

# 设置视野范围 (可选)
if 'spatial' in adata.obsm:
    axs[0].set_xlim([adata.obsm['spatial'].min(0)[0], adata.obsm['spatial'].max(0)[0]])
    axs[0].set_ylim([adata.obsm['spatial'].min(0)[1], adata.obsm['spatial'].max(0)[1]])

if rotate_key in adata.obsm:
    axs[1].set_xlim([adata.obsm[rotate_key].min(0)[0], adata.obsm[rotate_key].max(0)[0]])
    axs[1].set_ylim([adata.obsm[rotate_key].min(0)[1], adata.obsm[rotate_key].max(0)[1]])

# 绘制子图
sc.pl.embedding(adata, basis='spatial', color='batch', title='Ground truth 2D coordinates', 
                frameon=False, s=20, show=False, ax=axs[0])
sc.pl.embedding(adata, basis=rotate_key, color='batch', title='Rotated 2D coordinates', 
                frameon=False, s=20, show=False, ax=axs[1])

# 保存图片
save_path_2d = os.path.join(output_dir, 'input_comparison_2d.png')
plt.savefig(save_path_2d, dpi=300)
plt.close()
print(f"Saved: {save_path_2d}")

print("Done.")