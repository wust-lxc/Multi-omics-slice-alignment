import scanpy as sc
import matplotlib.pyplot as plt
import os

def main():
    # === 修改后的代码 ===
    # 动态获取当前脚本所在目录的上一级目录（也就是 STAIR-main 根目录）
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_dir, "Human_lymph_node_result", "human_lymph_node_processed.h5ad")
    output_path = os.path.join(root_dir, "Human_lymph_node_result", "spatial_comparison_2D.png")
    
    print(f"正在加载数据: {file_path} ...")
    adata = sc.read_h5ad(file_path)

    # 确保分类标签是 category 类型，以便 Scanpy 分配离散颜色
    adata.obs['final_annot'] = adata.obs['final_annot'].astype('category')
    adata.obs['Domain'] = adata.obs['Domain'].astype('category')

    # 提取切片批次顺序
    slices = sorted(adata.obs['batch'].unique())
    n_slices = len(slices)

    # 2. 初始化排版：行数为切片数量，列数为 2（真实注释 vs STAIR聚类）
    # 设置大尺寸画布，保证点云清晰不拥挤
    fig, axes = plt.subplots(nrows=n_slices, ncols=2, figsize=(14, 6 * n_slices))

    print("正在绘制空间对比图...")
    for i, slice_id in enumerate(slices):
        # 提取单张切片的数据
        adata_sub = adata[adata.obs['batch'] == slice_id].copy()

        # === 左列：病理学真实注释 (Ground Truth) ===
        sc.pl.embedding(
            adata_sub,
            basis="spatial",
            color="final_annot",
            ax=axes[i, 0],
            show=False,
            title=f"Slice: {slice_id} | Ground Truth",
            frameon=False,        # 去除边框，显得更干净
            size=60,              # 根据淋巴结细胞密度，60-80 是比较好的点大小
            legend_fontsize=10    # 图例字体大小
        )

        # === 右列：STAIR 算法预测 (Predicted Domain) ===
        sc.pl.embedding(
            adata_sub,
            basis="spatial",
            color="Domain",
            ax=axes[i, 1],
            show=False,
            title=f"Slice: {slice_id} | STAIR Domain (ARI ~0.4)",
            frameon=False,
            size=60,
            legend_fontsize=10
        )

    # 3. 调整图表间距并保存高清图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 完美！2D 空间对比图已保存至: {output_path}")

if __name__ == "__main__":
    main()