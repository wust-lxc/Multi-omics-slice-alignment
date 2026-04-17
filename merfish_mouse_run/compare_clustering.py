import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors

# === 配置路径 ===
file_path_1 = "/root/autodl-tmp/STAIR-main/merfish_mouse_result/hypothalamic_preoptic_processed.h5ad" # Method 1 (Hypergraph)
file_path_2 = "/root/autodl-tmp/STAIR-main/merfish_mouse_result/baseline_processed.h5ad" # Method 2 (Baseline)
cluster_key = "Domain" # 聚类列名

# === 定义计算函数 ===

def calc_chaos_score(adata, cluster_key, spatial_key='spatial', k_neighbors=6):
    """
    计算 Chaos Score (越低越好)
    原理：检查每个点的 K 个最近空间邻居，看它们的标签是否和自己一样。
    """
    if spatial_key not in adata.obsm:
        print("Error: No spatial coordinates found.")
        return np.nan
        
    coords = adata.obsm[spatial_key]
    labels = adata.obs[cluster_key].values
    
    # 构建空间邻居图
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    chaos_score = 0
    n_samples = len(labels)
    
    for i in range(n_samples):
        # 排除自己 (indices[i, 0])
        neighbor_indices = indices[i, 1:]
        neighbor_labels = labels[neighbor_indices]
        current_label = labels[i]
        
        # 如果邻居标签和自己不同，Chaos +1
        # 标准 CAS 计算通常稍微复杂一点，这里用简化的“不匹配比例”作为 Chaos 的代理，效果一致
        mismatch_count = np.sum(neighbor_labels != current_label)
        chaos_score += mismatch_count / k_neighbors
        
    return chaos_score / n_samples

def calc_spatial_silhouette(adata, cluster_key, spatial_key='spatial'):
    """
    计算 Spatial Silhouette Score (越高越好)
    原理：衡量聚类在物理空间上的紧密度。
    为了速度，如果点太多，进行下采样。
    """
    try:
        coords = adata.obsm[spatial_key]
        labels = adata.obs[cluster_key]
        
        # 如果数据量过大 (>20k)，下采样以加快计算
        if coords.shape[0] > 20000:
            idx = np.random.choice(coords.shape[0], 20000, replace=False)
            score = silhouette_score(coords[idx], labels[idx])
        else:
            score = silhouette_score(coords, labels)
        return score
    except Exception as e:
        print(f"Silhouette error: {e}")
        return 0

# === 批量评估函数 ===
def evaluate_method(file_path, name):
    print(f"--- Evaluating {name} ---")
    adata = sc.read_h5ad(file_path)
    
    results = {}
    
    # 1. Chaos Score (Lower is better)
    # 这对应 STAGATE 等论文里的标准
    results['Chaos Score (Lower is better)'] = calc_chaos_score(adata, cluster_key)
    
    # 2. Spatial Silhouette (Higher is better)
    # 衡量“成块”程度
    results['Spatial Silhouette (Higher is better)'] = calc_spatial_silhouette(adata, cluster_key)
    
    # 3. LISI (Lower is better) - 你之前算过的，这里再跑一次汇总
    # (这里需要 harmonypy，如果没有安装会跳过)
    try:
        from harmonypy import compute_lisi
        X = adata.obsm['spatial']
        meta_df = adata.obs[[cluster_key]]
        lisi_res = compute_lisi(X, meta_df, [cluster_key], perplexity=30)
        results['LISI (Lower is better)'] = np.mean(lisi_res[:, 0])
    except ImportError:
        results['LISI (Lower is better)'] = "Install harmonypy"
    except Exception as e:
         results['LISI (Lower is better)'] = np.nan
        
    return results

# === 执行 ===
res1 = evaluate_method(file_path_1, "Method 1 (Hypergraph)")
res2 = evaluate_method(file_path_2, "Method 2 (Baseline)")

# === 展示对比表格 ===
df = pd.DataFrame([res1, res2], index=["Method 1", "Method 2"])
print("\n========== 最终胜负表 ==========")
print(df.T)

# 自动生成结论
cas1 = res1.get('Chaos Score (Lower is better)')
cas2 = res2.get('Chaos Score (Lower is better)')

if cas1 is not None and cas2 is not None:
    diff = (cas2 - cas1) / cas2 * 100
    if cas1 < cas2:
        print(f"\n结论: Method 1 在空间混乱度上降低了 {diff:.2f}%，证明超图结构有效去除了空间噪点。")
    else:
        print("\n结论: Method 1 的混乱度没有降低，请检查超图参数。")