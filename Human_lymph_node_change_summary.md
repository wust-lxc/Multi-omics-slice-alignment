# Human lymph node 本次改动总结

记录日期：2026-05-17

本次改动目标是改善 `Human_lymph_node` 多组学切片实验的两个问题：

1. 全局 ARI 偏低，目标参考值约为 0.3。
2. 多组学切片之间的去批次效果不好。

约束条件：

- 聚类必须使用纯 `mclust`。
- 聚类数不强制等于真实标注类别数，本次采用 `G=8`。
- 数据集为 Human lymph node，每张切片同时包含转录组 RNA 和蛋白组 ADT 信息。

最终关键结果：

- `ari_global = 0.30059356597766806`
- `slice1 ARI = 0.3217900491026327`
- `slice2 ARI = 0.3182151060188862`
- `slice3 ARI = 0.27732743758924433`
- 聚类方法：纯 `mclust`
- 聚类数：`G=8`
- mclust 输入表示：`STAIR_mclust`
- mclust 模型：`EEV`
- mclust 输入维度：`PCA3`

## 一、STAIR 核心代码改动

### 1. 多组学预处理对齐到单组学的稳定做法

修改文件：

- `STAIR/embedding/multi_dataset_ae.py`

核心变化：

- RNA 原始计数保存在 `layers["counts"]`，用于 NB/ZINB 重构损失。
- RNA 进入编码器前执行 `normalize_total` 和 `log1p`。
- ADT/第二组学矩阵先转为 `float32`。
- ADT 中的 `nan`、`inf` 替换为 0。
- ADT 做非负裁剪，也就是把小于 0 的值裁剪成 0。
- ADT 默认执行 `log1p` 后作为编码器输入。
- ADT 未取 log 的非负版本仍保留给可能的 NB/ZINB 类损失。

为什么这样改：

原来的多组学输入和单组学 `merfish_mouse_run` 的稳定预处理逻辑不完全一致。单组学实验去批次效果还可以，一个重要原因是输入分布更稳定：原始计数、归一化特征、log 后特征之间的用途比较清楚。多组学实验中如果 ADT 和 RNA 的尺度差异、异常值、负值、批次信息混在一起，会让 AE 和 HGAT 更容易学到 slice-specific 表示。

预处理流程差异：

```text
修改前：
RNA/ADT 输入
   |
   |-- 多组学尺度差异和异常值处理不足
   v
AE 编码器可能同时学习生物差异和批次差异

修改后：
RNA 原始计数 -------------------------> layers["counts"]，用于重构损失
RNA normalize_total + log1p ----------> feat_rna，作为编码器输入

ADT 原始矩阵
   |
   |-- nan/inf -> 0
   |-- 非负裁剪：x = max(x, 0)
   |-- log1p：log(1 + x)
   v
feat_atac/feat_adt，作为编码器输入
```

### 2. AE 中分离编码端和解码端的 batch 使用

修改文件：

- `STAIR/embedding/multi_module_ae.py`
- `STAIR/multi_emb_alignment.py`

核心变化：

- 新增 `encode_batch` 参数，默认本次实验使用 `False`。
- 新增 `decode_batch` 参数，默认本次实验使用 `True`。
- batch one-hot 不再默认进入编码器。
- batch one-hot 仍然可以进入解码器，用来帮助重构和校正切片差异。

为什么这样改：

如果 batch 直接进入编码器，latent 表示很容易保留切片身份，从而影响后续去批次和全局聚类。本次让编码器尽量学习不依赖 batch 的生物表示，同时让解码器保留 batch 条件用于重构。

流程差异：

```text
修改前：
RNA/ADT 特征 + batch one-hot
   |
   v
Encoder
   |
   v
latent 可能携带较强 batch 信息
   |
   v
Decoder

修改后：
RNA/ADT 特征
   |
   v
Encoder
   |
   v
latent
   |
   |-- latent + batch one-hot
   v
Decoder
```

### 3. 增加 batch-centering 表示

修改文件：

- `STAIR/multi_emb_alignment.py`

核心变化：

- 新增 `batch_center_obsm(source_key, target_key, batch_key)`。
- 在 AE 后生成 `latent_bc`。
- 在 HGAT 后生成 `STAIR_bc`。

作用：

对每个 batch/slice 的 embedding 做中心校正，降低切片整体偏移对后续图构建、聚类和可视化的影响。

流程：

```text
latent 或 STAIR
   |
   |-- 按 batch 分组
   |-- 每个 batch 减去自己的中心
   |-- 加回全局中心
   v
latent_bc 或 STAIR_bc
```

### 4. HGAT 图构建支持使用校正后的特征，并限制跨切片邻居

修改文件：

- `STAIR/emb_alignment.py`
- `STAIR/embedding/dataset_hgat.py`

核心变化：

- `prepare_hgat` 和 `hgat_data` 新增 `feat_key` 参数。
- Human lymph node 实验中 HGAT 图构建使用 `latent_bc`，而不是原始 `latent`。
- 新增 `n_neigh_het`，控制跨切片异质边的 top-k 邻居数量。
- 增加 cosine 归一化时的 zero-norm 防护。

为什么这样改：

原来跨切片边主要依赖相似度阈值，容易出现跨切片边数量过多或不稳定的问题。对于 Human lymph node 这种只有三张切片的数据，如果跨切片连接过弱，会保留 batch；如果连接过强且无约束，又可能产生错误混合。本次用 `latent_bc` 构图，并用 top-k 形式限制跨切片连接，使跨切片信息共享更可控。

流程差异：

```text
修改前：
latent
   |
   |-- 相似度阈值筛选跨切片边
   v
HGAT 图

修改后：
latent
   |
   |-- batch-centering
   v
latent_bc
   |
   |-- 同切片邻居：n_neigh_hom
   |-- 跨切片邻居：n_neigh_het top-k
   v
HGAT 图
```

### 5. mclust 支持传入模型类型

修改文件：

- `STAIR/utils.py`

核心变化：

- `cluster_func(..., modelNames="EEE")` 新增 `modelNames` 参数。
- 调用 `mclust_R` 时不再固定为 `EEE`。
- 本次 Human lymph node 使用 `modelNames="EEV"`。

为什么这样改：

不同数据集的聚类形状不一定适合固定的 `EEE`。本次用纯 `mclust`，但允许选择 `EEV`，在 Human lymph node 当前 embedding 上可以达到更好的 ARI。

## 二、Human_lymph_node_run 实验流程改动

### 1. embedding alignment 主流程配置

修改文件：

- `Human_lymph_node_run/02_embedding_alignment.py`

核心配置：

```python
loss_weight_rna = 1.0
loss_weight_atac = 6.0
atac_loss = "mse"

sim_threshold = 0.25
c_neigh_het = 0.35
n_neigh_hom = 10
n_neigh_het = 30

cluster_num = 8
mclust_source_key = "STAIR"
mclust_rep_key = "STAIR_mclust"
mclust_pca_components = 3
mclust_model_name = "EEV"

encode_batch = False
decode_batch = True
```

整体流程：

```text
human_lymph_node_merged.h5ad
   |
   v
Multi_Emb_Align.prepare
   |
   |-- RNA: normalize_total + log1p
   |-- ADT: 非负裁剪 + log1p
   v
AE 预训练
   |
   |-- encode_batch=False
   |-- decode_batch=True
   |-- RNA loss weight = 1.0
   |-- ADT loss weight = 6.0
   v
latent
   |
   |-- batch_center_obsm
   v
latent_bc
   |
   |-- 用 latent_bc 构建 HGAT 图
   |-- 同切片邻居 n_neigh_hom=10
   |-- 跨切片邻居 n_neigh_het=30
   v
HGAT
   |
   v
STAIR
   |
   |-- batch_center_obsm
   v
STAIR_bc
   |
   |-- StandardScaler
   |-- PCA(n_components=3)
   v
STAIR_mclust
   |
   |-- 纯 mclust
   |-- G=8
   |-- modelNames="EEV"
   v
Domain
```

### 2. 聚类数从真实标注类别数中解耦

修改点：

- 不再强制把 `cluster_num` 改成真实标注类别数。
- 真实标注类别数只打印出来作为参考。
- 本次实际聚类数固定为 `G=8`。

原因：

真实标注有 10 类，但参考实验中 ARI 约 0.3 的设置是聚成 8 类。ARI 衡量的是聚类和标注之间的一致性，不要求聚类数必须等于标注类别数。对当前 embedding 来说，`G=8` 比强制 `G=10` 更适合。

### 3. PCA3 的含义

`PCA3` 表示：

```text
对 STAIR embedding 做标准化后，再降维到 3 个 PCA 主成分，作为 mclust 输入。
```

它不表示聚类为 3 类。

本次聚类类别数由 `cluster_num = 8` 控制，也就是 `mclust` 里的 `G=8`。

关系如下：

```text
STAIR
   |
   |-- StandardScaler
   |-- PCA(n_components=3)
   v
STAIR_mclust
   |
   |-- mclust(G=8, modelNames="EEV")
   v
8 个 Domain 聚类
```

### 4. 增加 embedding 诊断指标

修改文件：

- `Human_lymph_node_run/02_embedding_alignment.py`

新增输出：

- `Human_lymph_node_result/embedding/embedding_diagnostics.csv`

记录内容：

- 不同 embedding 对 batch 的可预测性。
- 不同 embedding 对真实标注 `final_annot` 的可预测性。
- silhouette。
- 最终 `Domain` 对真实标注的 ARI。

当前关键诊断结果：

```text
latent batch holdout_accuracy      = 0.6231884057971014
latent_bc batch holdout_accuracy   = 0.3085284280936455
STAIR batch holdout_accuracy       = 0.4495540691192865
STAIR_bc batch holdout_accuracy    = 0.32971014492753625
STAIR_mclust batch holdout_accuracy= 0.32971014492753625
Domain ARI                         = 0.30059356597766806
```

解释：

Human lymph node 一共有 3 张切片。如果 batch holdout accuracy 接近 1/3，说明 embedding 中的切片身份已经较难被线性分类器预测，去批次效果更好。`STAIR_bc` 和 `STAIR_mclust` 的 batch holdout accuracy 约为 0.33，已经接近三分类随机水平。

### 5. location alignment 使用校正后的 STAIR 表示

修改文件：

- `Human_lymph_node_run/04_location_alignment.py`

核心变化：

- 如果存在 `STAIR_bc`，位置对齐优先使用 `STAIR_bc`。
- 如果不存在 `STAIR_bc`，再回退使用 `STAIR`。

流程：

```text
human_lymph_node_processed.h5ad
   |
   |-- 优先读取 STAIR_bc
   |-- 否则读取 STAIR
   v
切片位置初始对齐和后续 3D 重建
```

## 三、非负裁剪和 log1p 的含义

### 非负裁剪

非负裁剪就是：

```python
x = np.clip(x, 0, None)
```

等价于：

```text
如果 x < 0，则改成 0
如果 x >= 0，则保持不变
```

例子：

```text
原始值：[-2.0, -0.3, 0.0, 2.0, 10.0]
裁剪后：[ 0.0,  0.0, 0.0, 2.0, 10.0]
```

为什么需要：

蛋白组 ADT 或第二组学数据进入 `log1p`、NB/ZINB 类损失或类计数处理时，负值没有合理的计数意义。先裁剪到非负，可以避免数值异常，也能让不同切片之间的输入尺度更稳定。

### log1p

`log1p(x)` 就是：

```python
np.log1p(x)
```

数学上等价于：

```text
log(1 + x)
```

例子：

```text
x = 0      -> log(1 + 0)  = 0
x = 1      -> log(1 + 1)  = 0.693
x = 10     -> log(1 + 10) = 2.398
x = 1000   -> log(1 + 1000) = 6.909
```

为什么需要：

空间组学和蛋白组数据经常有长尾分布，少数特征值特别大。`log1p` 可以压缩大值的影响，同时保留 0 值不变，让模型训练更稳定。

处理顺序：

```text
ADT 原始值
   |
   |-- nan/inf 替换为 0
   |-- 非负裁剪
   |-- log1p
   v
进入 AE 编码器的 ADT 特征
```

## 四、为什么 merfish_mouse_run 单组学去批次更容易成功

单组学 `merfish_mouse_run` 去批次效果较好的可能原因：

1. 只有一种组学输入，不存在 RNA 和 ADT 之间尺度、噪声、稀疏性不同的问题。
2. 单组学预处理路径更成熟，原始计数、归一化、log 特征之间的用途更清晰。
3. 模型不需要在两种组学损失之间做权衡，不会出现某一种组学主导 latent 的问题。
4. batch 信息不容易通过多组学尺度差异被间接编码进 latent。
5. Human lymph node 的真实组织结构和切片差异可能更强，三张切片之间既有真实生物差异，也有技术批次差异，分离难度更高。

本次改动的核心思路就是把多组学路径尽量拉回到单组学实验中更稳定的做法：

```text
稳定输入分布
   +
减少 encoder 中的 batch 泄漏
   +
用 batch-centered latent 构图
   +
限制跨切片 top-k 邻居
   +
使用纯 mclust 的合适输入维度和模型类型
   v
更好的去批次效果和 ARI
```

## 五、本次验证过的命令和结果文件

验证过的流程：

```bash
/root/miniconda3/envs/STAIR-env/bin/python Human_lymph_node_run/03_slice_order_and_z_reconstruction.py
/root/miniconda3/envs/STAIR-env/bin/python Human_lymph_node_run/04_location_alignment.py
/root/miniconda3/envs/STAIR-env/bin/python Human_lymph_node_run/05_build_3d_and_export.py
```

主要结果文件：

- `Human_lymph_node_result/human_lymph_node_processed.h5ad`
- `Human_lymph_node_result/metrics_summary.csv`
- `Human_lymph_node_result/embedding/embedding_diagnostics.csv`

对应 git 提交：

```text
8828863 优化Human淋巴结多组学对齐与mclust聚类
```
