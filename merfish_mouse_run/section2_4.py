import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib
# 服务器端必须设置 Agg 后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
import math

# 导入 STAIR 模块
from STAIR.emb_alignment import Emb_Align
from STAIR.loc_alignment import Loc_Align
from STAIR.location.transformation import best_fit_transform

# ==============================================================================
# 0. 环境与数据加载
# ==============================================================================
result_path = './merfish_mouse_result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

location_path = os.path.join(result_path, 'location')
if not os.path.exists(location_path):
    os.makedirs(location_path)

# [关键点] 读取上一步更新后的处理文件
processed_file_path = os.path.join(result_path, 'hypothalamic_preoptic_processed.h5ad')

if os.path.exists(processed_file_path):
    print(f"Loading processed data from: {processed_file_path}")
    adata = sc.read(processed_file_path)
else:
    raise FileNotFoundError(f"Processed file not found at {processed_file_path}. Please run section2_2.py and section2_3.py first.")

# 检查必要信息是否存在
if 'STAIR' not in adata.obsm:
    raise KeyError("'STAIR' embeddings not found in data. Please check section2_2.py execution.")
if 'z_rec' not in adata.obs:
    print("Warning: 'z_rec' not found. Alignment order might be suboptimal.")

# 定义切片列表 (如果 z_rec 存在，基于它排序；否则使用默认)
keys_use = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11']

if 'z_rec' in adata.obs:
    # 基于重构的 z 坐标确定对齐顺序
    keys_order = adata.obs[['batch', 'z_rec']].drop_duplicates().sort_values('z_rec', ascending=False)['batch'].tolist()
    print(f"Alignment order based on z_rec: {keys_order}")
else:
    keys_order = keys_use
    print(f"Using default alignment order: {keys_order}")

# ==============================================================================
# 1. 初始化对齐模块 (Loc_Align Initialization)
# ==============================================================================
print("Initializing Loc_Align...")

loc_align = Loc_Align(adata, batch_key='batch', batch_order=keys_order, result_path=result_path)

# ==============================================================================
# 2. 初始对齐 (Initial Alignment)
# ==============================================================================
print("Performing Initial Alignment...")

# 使用旋转后的坐标（模拟未知/错位状态）作为输入
# 如果数据中没有这个 key，自动降级为 'spatial'
input_spatial_key = 'spatial_2d_μm_rotate' if 'spatial_2d_μm_rotate' in adata.obsm else 'spatial'
print(f"Using '{input_spatial_key}' as input spatial coordinates.")

loc_align.init_align(
    emb_key='STAIR', 
    spatial_key=input_spatial_key,
    num_mnn=1
)

# ==============================================================================
# 3. 精细对齐点检测 (Detect Fine Points)
# ==============================================================================
print("Detecting Fine Alignment Points...")

# 确保 Domain 列存在 (通常由 section2_2 生成)
if 'Domain' not in adata.obs:
    print("Warning: 'Domain' column missing. Using dummy domain for compatibility.")
    adata.obs['Domain'] = 'stHY'

loc_align.detect_fine_points(
    domain_key='Domain',
    slice_boundary=True,
    domain_boundary=True,
    num_domains=3,
    alpha=70,
    return_result=False
)

# 保存边界检测图
print("Plotting Edge Detection...")
save_path_edge = os.path.join(location_path, 'edge_detection.png')
# plot_edge 可能会生成图像，我们将其保存
loc_align.plot_edge(spatial_key='transform_init', figsize=(6,6), s=2)
plt.savefig(save_path_edge, dpi=300)
plt.close()
print(f"Saved edge detection plot to: {save_path_edge}")

# ==============================================================================
# 4. 执行精细对齐 (Fine Alignment)
# ==============================================================================
print("Performing Fine Alignment...")
adata = loc_align.fine_align(
    max_iterations=200,    # <--- 修改点：从默认 20 增加到 200
    tolerance=1e-10        # 保持高精度阈值
)

# 此时 adata.obsm['transform_fine'] 已生成

# ==============================================================================
# 5. 可视化对齐结果 (Visualization)
# ==============================================================================
print("Visualizing Alignment Results...")

# 准备颜色映射
hex_colors = [matplotlib.colors.rgb2hex(color) for color in plt.cm.tab20.colors]
categories = ['stHY', 'MnPO', 'Hypo', 'Pe', 'MPA', 'opn', 'MPN', 'BNST', 'ACA_Fx', 'VLPO', '3V', 'BAC', 'PVN']

# 仅保留存在的类别
current_domains = adata.obs['Domain'].astype(str).unique()
valid_cats = [c for c in categories if c in current_domains]

adata.obs['Domain'] = adata.obs['Domain'].astype(str).astype('category')
try:
    adata.obs['Domain'] = adata.obs['Domain'].cat.set_categories(valid_cats)
except:
    pass

if 'Domain_colors' not in adata.uns or len(adata.uns['Domain_colors']) < len(valid_cats):
    adata.uns['Domain_colors'] = hex_colors[:len(valid_cats)]

# 5.1 Initial Alignment Plot
save_path_init = os.path.join(location_path, 'alignment_init.png')
with rc_context({'figure.figsize': (6, 6)}):
    sc.pl.embedding(adata, basis='transform_init', color=['batch', 'Domain'],
                    frameon=False, ncols=2, show=False, s=15)
    plt.savefig(save_path_init, dpi=300, bbox_inches='tight')
    plt.close()
print(f"Saved initial alignment plot to: {save_path_init}")

# 5.2 Fine Alignment Plot
save_path_fine = os.path.join(location_path, 'alignment_fine.png')
with rc_context({'figure.figsize': (6, 6)}):
    sc.pl.embedding(adata, basis='transform_fine', color=['batch', 'Domain'],
                    frameon=False, ncols=2, show=False, s=15)
    plt.savefig(save_path_fine, dpi=300, bbox_inches='tight')
    plt.close()
print(f"Saved fine alignment plot to: {save_path_fine}")

# ==============================================================================
# 6. 误差计算 (Error Calculation) - 使用原始论文逻辑
# ==============================================================================
print("Calculating Alignment Errors (Using Original Paper Logic)...")

# 1. 替换为原始论文提供的角度计算函数
def cacul_angle(cos_val, sin_val):
    # 原文逻辑：利用三角恒等式求解正切值
    # 注意：原文这里没有处理除零风险，保留原样以复刻逻辑
    tan_val = math.sqrt(1 - cos_val**2) / cos_val  
    if cos_val > 0:
        if sin_val >= 0:
            theta = math.atan(tan_val)  # 计算角度
        else:
            theta = 2*math.pi - math.atan(tan_val)
    else:
        if sin_val >= 0:
            theta = math.pi - math.atan(tan_val)
        else:
            theta = math.pi + math.atan(tan_val)
    return theta

def square(x):
    return (x ** 2)

angle_inits = []
angle_fines = []
trans_inits = []
trans_fines = []

gt_key = 'spatial_2d_μm'

if gt_key in adata.obsm:
    # 使用 keys_use[1:] 进行评估 (跳过第一个参考切片)
    eval_slices = keys_use[1:] 
    
    for key in eval_slices:
        idx = adata.obs['batch'] == key
        if not np.any(idx):
            continue
            
        subset_init = adata.obsm['transform_init'][idx]
        subset_fine = adata.obsm['transform_fine'][idx]
        subset_gt = adata.obsm[gt_key][idx]
        
        # 计算 Init -> GT
        # best_fit_transform 返回 (T, R, t)
        init_res = best_fit_transform(subset_init, subset_gt)
        init_R, init_t = init_res[1], init_res[2]
        
        # 计算 Fine -> GT
        fine_res = best_fit_transform(subset_fine, subset_gt)
        fine_R, fine_t = fine_res[1], fine_res[2]

        # [关键修改点] 复刻原始论文的调用逻辑：
        # 原文代码：cacul_angle(cos_val=init[0][0,0], sin_val=init[0][0,0])
        # 这里 init_R[0,0] 是 cos, init_R[1,0] 是 sin
        # 但原文逻辑是将 R[0,0] (cos) 同时传给了 sin_val 参数
        theta_init = cacul_angle(cos_val=init_R[0,0], sin_val=init_R[0,0])
        theta_fine = cacul_angle(cos_val=fine_R[0,0], sin_val=fine_R[0,0])
        
        angle_inits.append(theta_init)
        angle_fines.append(theta_fine)
        
        trans_inits.append(init_t)
        trans_fines.append(fine_t)

    if len(eval_slices) > 0:
        # 构建 DataFrame
        error_init = pd.DataFrame(trans_inits, index=eval_slices, columns=['trans_x', 'trans_y'])
        error_fine = pd.DataFrame(trans_fines, index=eval_slices, columns=['trans_x', 'trans_y'])
        
        error_init['Angle'] = angle_inits
        error_init['Method'] = 'STAIR_init'
        error_fine['Angle'] = angle_fines
        error_fine['Method'] = 'STAIR_fine'
        
        error = pd.concat([error_init, error_fine])
        
        # [关键修改点] 复刻原始论文的平移误差计算逻辑
        # error['Trans'] = np.sqrt(error['trans_x'].map(square) + error['trans_y'].map(square))/1000
        error['Trans'] = np.sqrt(error['trans_x'].apply(square) + error['trans_y'].apply(square)) / 1000
        
        # 整理用于绘图的数据，去掉坐标列，保留 Angle, Method, Trans
        error_plot_data = error[['Angle', 'Method', 'Trans']]
        
        print("Plotting Error Boxplot (Original Paper Style)...")
        plt.figure(figsize=(3, 3))
        
        # Melt 数据以适应 seaborn 的 hue 绘图逻辑
        # id_vars='Method', value_vars=['Angle', 'Trans']
        # 注意：原文代码这里 error = error.iloc[:,2:] 然后 melt，意味着它只画 Angle 和 Trans
        plot_df = error_plot_data.melt(id_vars=['Method'])
        
        sns.boxplot(x='Method', y='value', hue='variable', data=plot_df)
        
        save_path_error = os.path.join(location_path, 'alignment_error_boxplot_original_logic.png')
        plt.savefig(save_path_error, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error plot to: {save_path_error}")
        
        # 打印一下数值以便查看
        print("\n=== Error Metrics (Original Logic) ===")
        print(error_plot_data.groupby('Method').median())
        
    else:
        print("Not enough slices to calculate error.")
else:
    print(f"Ground Truth key '{gt_key}' not found. Skipping error calculation.")
# ==============================================================================
# 7. 保存最终结果 (可选)
# ==============================================================================
# 此时 adata.obsm['transform_fine'] 包含了最终对齐坐标
print(f"Saving final alignment data to: {processed_file_path}")
adata.write(processed_file_path)