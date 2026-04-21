import os
import scanpy as sc
import pandas as pd

def main():
    # 动态获取当前脚本所在目录的上一级目录（即 STAIR-main 根目录）
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_dir, "Human_lymph_node_result", "human_lymph_node_processed.h5ad")
    
    print(f"正在加载数据: {file_path}")
    adata = sc.read_h5ad(file_path)
    
    print("\n" + "="*50)
    print("🔥 诊断 1: Domain vs 切片批次 (检查批次效应是否消除)")
    print(pd.crosstab(adata.obs['Domain'], adata.obs['batch']))
    print("="*50)
    
    print("\n🔥 诊断 2: Domain vs 真实注释 (检查特征是否过平滑/糊化)")
    crosstab_gt = pd.crosstab(adata.obs['Domain'], adata.obs['final_annot'])
    print(crosstab_gt)

if __name__ == "__main__":
    main()