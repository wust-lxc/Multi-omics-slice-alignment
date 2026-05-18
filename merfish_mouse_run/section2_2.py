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
from STAIR.emb_alignment import Emb_Align
from STAIR.utils import *
set_seed(42)

# ========================================S======================================
# 0. 环境与数据准备
# ==============================================================================

result_path = './merfish_mouse_result'
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print(f"Created directory: {result_path}")

embedding_path = os.path.join(result_path, 'embedding')
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

# 定义最终保存的文件路径
processed_file_path = os.path.join(result_path, 'hypothalamic_preoptic_processed.h5ad')

# 加载原始数据
print("Loading original data...")
adata = sc.read('./data/hypothalamic_preoptic.h5ad')

# 定义切片顺序
keys_use = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11']

# 初始化模型
emb_align = Emb_Align(adata, batch_key='batch', result_path=result_path)

# 重新运行 AE 提取 Latent (确保 adata.obsm['latent'] 存在)
print("Ensuring latent features exist...")
emb_align.prepare(lib_size='explog', normalize=True, scale=False)
emb_align.preprocess(epoch_ae=100, batch_size=128) 
emb_align.latent()

# ==============================================================================
# 1. 构建异构图 (Construct the heterogeneous graph)
# ==============================================================================
print("Constructing heterogeneous graph...")
# 使用旋转后的坐标作为空间输入
emb_align.prepare_hgat(spatial_key='spatial_2d_μm_rotate',
                       slice_order=keys_use,sim_threshold=0.3)

# ==============================================================================
# 2. 学习与对齐 (Learning & Integrating)
# ==============================================================================
print("Training HGAT (Full Batch Mode)...")
# 训练 (使用 Full Batch 保证超图结构完整)
emb_align.train_hgat(mini_batch=False)

print("Predicting aligned embeddings...")
# 预测
adata, atte = emb_align.predict_hgat(mini_batch=False)

# 保存 Attention 结果
atte_csv_path = os.path.join(embedding_path, 'attention.csv')
atte.to_csv(atte_csv_path)
print(f"Attention weights saved to: {atte_csv_path}")

# ==============================================================================
# 3. 聚类 (Clustering of spatial embedding)
# ==============================================================================
print("Clustering...")
# 使用 mclust (需 R 环境支持)
adata = cluster_func(adata, clustering='mclust', use_rep='STAIR', cluster_num=13, key_add='STAIR')
print("Mclust clustering completed.")

# ==============================================================================
# 4. 可视化 (Visualization) - UMAP
# ==============================================================================
print("Generating UMAP...")
sc.pp.neighbors(adata, use_rep='STAIR')
sc.tl.umap(adata, min_dist=0.2)

# Plot 1: Batch & Cluster UMAP
save_path_umap = os.path.join(result_path, 'umap_batch_cluster.png')
with rc_context({'figure.figsize': (4,4)}):
    sc.pl.umap(adata, color=['batch', 'STAIR'], frameon=False, ncols=3, show=False)
    plt.savefig(save_path_umap, dpi=300, bbox_inches='tight')
    plt.close()
print(f"Saved UMAP to {save_path_umap}")

# ==============================================================================
# 5. 注释 (Annotation)
# ==============================================================================
print("Annotating domains...")
mapping = { 
    '1.0': 'stHY', '10.0': 'VLPO', '11.0': '3V', '12.0': 'BAC', '13.0': 'PVN', '2.0': 'MnPO',
    '3.0': 'Hypo', '4.0': 'Pe', '5.0': 'MPA', '6.0': 'opn', '7.0': 'MPN', '8.0': 'BNST', '9.0': 'ACA_Fx'
}

# 仅替换存在的类别，防止报错
current_cats = adata.obs['STAIR'].astype(str).unique()
safe_mapping = {k: v for k, v in mapping.items() if k in current_cats}

adata.obs['Domain'] = adata.obs['STAIR'].astype(str).replace(safe_mapping)

# 设置类别顺序
target_categories = ['stHY', 'MnPO', 'Hypo', 'Pe', 'MPA', 'opn', 'MPN', 'BNST', 'ACA_Fx', 'VLPO', '3V', 'BAC', 'PVN']
valid_target_cats = [c for c in target_categories if c in adata.obs['Domain'].unique()]
adata.obs['Domain'] = adata.obs['Domain'].astype('category')
try:
    adata.obs['Domain'] = adata.obs['Domain'].cat.set_categories(valid_target_cats)
except:
    pass

# 设置颜色
hex_colors = [matplotlib.colors.rgb2hex(color) for color in plt.cm.tab20.colors]
if 'Domain' in adata.obs:
    n_domains = len(set(adata.obs['Domain'].dropna()))
    adata.uns['Domain_colors'] = hex_colors[:n_domains]

# Plot 2: Domain UMAP
save_path_domain = os.path.join(result_path, 'umap_domain.png')
with rc_context({'figure.figsize': (4,4)}):
    sc.pl.umap(adata, color=['Domain'], frameon=False, ncols=3, show=False)
    plt.savefig(save_path_domain, dpi=300, bbox_inches='tight')
    plt.close()
print(f"Saved Domain UMAP to {save_path_domain}")

# ==============================================================================
# 6. 空间域 2D 可视化 (2D visualization of spatial domain)
# ==============================================================================
print("Generating Spatial Domain Plots...")
n_row = 3
n_col = 4
fig, axs = plt.subplots(n_row, n_col, figsize=(9.5,7.8), constrained_layout=True)

index = 0
spatial_key = 'spatial' # 使用原始空间坐标展示 Domain 分布

# 获取全局坐标范围
x_min, x_max = adata.obsm[spatial_key][:, 0].min(), adata.obsm[spatial_key][:, 0].max()
y_min, y_max = adata.obsm[spatial_key][:, 1].min(), adata.obsm[spatial_key][:, 1].max()

for i in range(n_row):
    for j in range(n_col):
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
        axs[i,j].axis('off')
        axs[i,j].set_xlim([x_min, x_max])
        axs[i,j].set_ylim([y_min, y_max])
        
        if index < len(keys_use):
            key = keys_use[index]
            # 提取该切片的数据
            adata_tmp = adata[adata.obs['batch']==key].copy()

            # 仅在每一行的最后一个子图显示图例
            show_legend = (j == n_col - 1)
            sc.pl.embedding(
                adata_tmp,
                basis=spatial_key,
                color='Domain',
                title=key,
                frameon=False,
                legend_loc='right' if show_legend else None,
                s=30,
                show=False,
                ax=axs[i,j]
            )

        index += 1

# 保存最终大图
save_path_spatial = os.path.join(result_path, 'spatial_domains_2d.png')
plt.savefig(save_path_spatial, dpi=300)
plt.close()
print(f"Saved Spatial Domains plot to {save_path_spatial}")

# ==============================================================================
# [关键步骤] 保存处理后的数据
# ==============================================================================
print(f"Saving processed AnnData to: {processed_file_path}")
adata.write(processed_file_path)