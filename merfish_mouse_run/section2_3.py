import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
# 服务器端设置 Agg 后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

# 导入 STAIR 工具
from STAIR.loc_prediction import sort_slices

# ==============================================================================
# 0. 环境与数据加载
# ==============================================================================

result_path = './merfish_mouse_result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# 定义文件路径
processed_file_path = os.path.join(result_path, 'hypothalamic_preoptic_processed.h5ad')
original_file_path = './data/hypothalamic_preoptic.h5ad'
atte_path = os.path.join(result_path, 'embedding', 'attention.csv')

# 1. 优先加载处理过的数据 (来自 section 2.2)
if os.path.exists(processed_file_path):
    print(f"Loading processed data from {processed_file_path}...")
    adata = sc.read(processed_file_path)
else:
    print(f"Warning: Processed file not found. Falling back to original {original_file_path}...")
    print("Note: If you haven't run section2_2.py, 'STAIR' embeddings will be missing.")
    adata = sc.read(original_file_path)

# 2. 加载 Attention 矩阵
if os.path.exists(atte_path):
    print(f"Loading attention weights from {atte_path}...")
    atte = pd.read_csv(atte_path, index_col=0)
    # 确保索引和列名一致为字符串
    atte.index = atte.index.astype(str)
    atte.columns = atte.columns.astype(str)
else:
    raise FileNotFoundError(f"Attention file not found at {atte_path}. Please run section2_2.py first.")

# 定义切片列表
keys_use = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11']

# 兼容性检查
if 'Bregma' not in adata.obs:
    print("Warning: 'Bregma' not found, utilizing 'z' column.")
    adata.obs['Bregma'] = adata.obs['z']

# ==============================================================================
# 1. Attention Heatmap 可视化
# ==============================================================================
print("Plotting Attention Heatmap...")
vmax = atte[atte!=1].max().max()
vmin = atte[atte!=1].min().min()

plt.figure(figsize=(4.4,4))
sns.heatmap(atte, vmax=vmax, vmin=vmin)

save_path_heatmap = os.path.join(result_path, 'attention_heatmap.png')
plt.savefig(save_path_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved heatmap to {save_path_heatmap}")

# ==============================================================================
# 2. Attention-Spatial Consistency (Spearman Correlation)
# ==============================================================================
print("Analyzing Attention-Spatial Consistency...")

use = adata.obs[['batch', 'Bregma']].drop_duplicates()
use.index = use['batch']
use = use['Bregma']

attes = []
dists_list = []
for i in keys_use:
    for j in keys_use:
        if (i != j) & (i < j):
            try:
                # 提取双向注意力的均值
                atte_tmp = (atte.loc[i,j] + atte.loc[j,i]) / 2
                dist_tmp = abs(use[i] - use[j])
                attes.append(atte_tmp)
                dists_list.append(dist_tmp)
            except KeyError:
                pass

plt.figure(figsize=(3,3))
plt.scatter(dists_list, attes, s=5)
plt.xlabel('Bregma distance', fontsize=13)
plt.ylabel('Attention score', fontsize=13)

if len(attes) > 1:
    spearman_corr = round(spearmanr(attes, dists_list)[0], 2)
    plt.text(0.28, 0.115, 'Spearman: ' + str(spearman_corr), transform=plt.gca().transAxes)

save_path_spearman = os.path.join(result_path, 'attention_bregma_correlation.png')
plt.savefig(save_path_spearman, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved correlation plot to {save_path_spearman}")

# ==============================================================================
# 3. Z-axis Reconstruction (Z 轴重构)
# ==============================================================================
print("Reconstructing Z-coordinates...")

# 使用 STAIR 提供的工具根据 Attention 排序切片
dists_pred = sort_slices(atte, start='A11')

# 映射得到初始相对位置
adata.obs['z_rec'] = adata.obs['batch'].replace(dists_pred).astype(float)

# 归一化 z_rec 到 [0, 1]
z_rec_min = adata.obs['z_rec'].min()
z_rec_max = adata.obs['z_rec'].max()
adata.obs['z_rec'] = (adata.obs['z_rec'] - z_rec_min) / (z_rec_max - z_rec_min)

# 将归一化后的 z_rec 映射回原始物理尺度 (利用 Ground Truth 的范围)
z_gt_max = adata.obs['z'].max()
z_gt_min = adata.obs['z'].min()
adata.obs['z_rec'] = adata.obs['z_rec'] * (z_gt_max - z_gt_min) + z_gt_min

print("Sample reconstructed Z values:")
print(adata.obs[['batch', 'z', 'z_rec']].drop_duplicates().head())

# ==============================================================================
# 4. Reconstruction Evaluation Plot
# ==============================================================================
print("Plotting Reconstruction Evaluation...")

# 确定绘图范围
all_z_values = adata.obs['z_rec'].tolist() + adata.obs['z'].tolist()
vmin_plot = np.min(all_z_values)
vmax_plot = np.max(all_z_values)

plt.figure(figsize=(3,3))
# 绘制 y=x 参考线
plt.plot([vmin_plot, vmax_plot], [vmin_plot, vmax_plot], 'k--', alpha=0.5)
# 绘制散点
plt.scatter(adata.obs['z'], adata.obs['z_rec'], s=1, c='r')

# 计算相关性指标 (使用内置 round 函数)
corr_val = round(pearsonr(adata.obs['z'], adata.obs['z_rec'])[0], 2)
r2_val = round(r2_score(adata.obs['z'], adata.obs['z_rec']), 2)

# 动态文本位置
text_x = vmin_plot + (vmax_plot - vmin_plot) * 0.05
text_y = vmax_plot - (vmax_plot - vmin_plot) * 0.2

plt.text(text_x, text_y, 
         'Corr: ' + str(corr_val) + '\n' + "R^2: " + str(r2_val))

plt.xlim([vmin_plot, vmax_plot])
plt.ylim([vmin_plot, vmax_plot])
plt.xlabel('Bregma (Ground Truth)')
plt.ylabel('Reconstructed Bregma')

save_path_rec = os.path.join(result_path, 'z_reconstruction_correlation.png')
plt.savefig(save_path_rec, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved reconstruction plot to {save_path_rec}")

# ==============================================================================
# [关键步骤] 保存包含 z_rec 的数据到磁盘
# ==============================================================================
print(f"Updating processed data at: {processed_file_path}")
adata.write(processed_file_path)