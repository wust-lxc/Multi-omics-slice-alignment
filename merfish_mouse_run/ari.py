import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score

# === 配置 ===
# Baseline 文件路径
baseline_path = "/root/autodl-tmp/STAIR-main/merfish_mouse_result/baseline_processed.h5ad"

# Method 1 (超图) 的成绩 (来自你刚才的运行结果)
score_method_1 = 0.3645 

print(f"=== 正在评估 Baseline (原版 STAIR) ===")
try:
    adata = sc.read_h5ad(baseline_path)
    print(f"成功读取文件: {baseline_path}")
    
    # 1. 寻找嵌入 Key
    # 优先找 'STAIR', 找不到就用第一个
    emb_key = 'STAIR' if 'STAIR' in adata.obsm.keys() else list(adata.obsm.keys())[0]
    print(f"使用嵌入 Key: {emb_key}")
    
    # 2. 重新聚类 (Re-clustering)
    # 使用与 Method 1 相同的参数 (n_neighbors=15, resolution=0.5) 保证公平
    print("正在进行 Leiden 聚类...")
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
    sc.tl.leiden(adata, key_added='new_cluster', resolution=0.5)
    
    # 3. 计算 Baseline ARI
    if 'Domain' in adata.obs.columns:
        valid_mask = ~adata.obs['Domain'].isna()
        y_true = adata.obs.loc[valid_mask, 'Domain']
        y_pred = adata.obs.loc[valid_mask, 'new_cluster']
        
        score_method_2 = adjusted_rand_score(y_true, y_pred)
        print(f"\n[Baseline 结果] ARI (vs Domain): {score_method_2:.4f}")
        
        # === 4. 最终胜负判定 ===
        print("\n" + "="*30)
        print("       最终对比结果       ")
        print("="*30)
        print(f"Method 1 (Hypergraph): {score_method_1:.4f}")
        print(f"Method 2 (Baseline)  : {score_method_2:.4f}")
        
        diff = score_method_1 - score_method_2
        improvement = (diff / score_method_2) * 100
        
        if score_method_1 > score_method_2:
            print(f"\n✅ 结论: 你的改进有效！ARI 提升了 {diff:.4f} (相对提升 {improvement:.2f}%)")
            print(f"📝 论文建议写法: 'By incorporating hypergraph structures, we improved the identification of anatomical domains, increasing the ARI from {score_method_2:.2f} (baseline) to {score_method_1:.2f}.'")
        else:
            print(f"\n⚠️ 结论: 改进不明显。Method 1 比 Baseline 低了 {-diff:.4f}。")
            print("建议: 尝试调整 resolution 或检查超图构建参数。")
            
    else:
        print("错误: Baseline 文件中没找到 'Domain' 列作为真值，无法计算 ARI。")

except Exception as e:
    print(f"运行出错: {e}")