import os
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

def main():
    # 明确读取路径
    file_path = "Human_lymph_node_result/human_lymph_node_processed.h5ad"
    print(f"正在加载数据: {file_path} ...")
    adata = sc.read_h5ad(file_path)
    
    # 过滤掉那些可能没有被注释到的无效细胞（严谨起见）
    valid_mask = ~adata.obs['final_annot'].isna() & ~adata.obs['Domain'].isna()
    
    # 提取真实标签和预测标签
    y_true_all = adata.obs['final_annot'][valid_mask].astype(str)
    y_pred_all = adata.obs['Domain'][valid_mask].astype(str)
    
    # 1. 计算全局 ARI
    global_ari = adjusted_rand_score(y_true_all, y_pred_all)
    print("\n" + "="*30)
    print(f"🌟 整体全局 ARI Score: {global_ari:.4f}")
    print("="*30)
    
    # 2. 按切片 (slice1, slice2, slice3) 分别计算 ARI
    print("\n--- 按切片拆解 ARI 指标 ---")
    batches = adata.obs['batch'].unique()
    
    for batch in sorted(batches):
        mask_batch = valid_mask & (adata.obs['batch'] == batch)
        y_true_b = adata.obs['final_annot'][mask_batch].astype(str)
        y_pred_b = adata.obs['Domain'][mask_batch].astype(str)
        
        if len(y_true_b) > 0:
            ari_b = adjusted_rand_score(y_true_b, y_pred_b)
            print(f"  📌 {batch} ARI: {ari_b:.4f} (基于 {len(y_true_b)} 个 Spot/细胞)")
        else:
            print(f"  📌 {batch} 无有效注释，跳过计算。")

if __name__ == "__main__":
    main()