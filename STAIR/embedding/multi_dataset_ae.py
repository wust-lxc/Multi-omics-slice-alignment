import torch
import numpy as np
from torch.utils.data import Dataset

class MultiOmicsDataset(Dataset):
    def __init__(self, adata, atac_key='ATAC', count_key=None, batch_key=None):
        super(MultiOmicsDataset, self).__init__()
        
        # 1. 提取 RNA 特征
        if count_key is not None:
            self.feat_rna = adata.layers[count_key].toarray() if hasattr(adata.layers[count_key], 'toarray') else adata.layers[count_key]
        else:
            self.feat_rna = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            
        self.count_rna = self.feat_rna.copy()
        self.size_rna = np.log(self.count_rna.sum(1) + 1).reshape(-1, 1)
        
        # 2. 提取 ATAC 特征
        self.feat_atac = adata.obsm[atac_key]
        self.count_atac = self.feat_atac.copy()
        self.size_atac = np.log(self.count_atac.sum(1) + 1).reshape(-1, 1)
        
        # 3. 提取批次信息
        self.batch_key = batch_key
        if self.batch_key is not None:
            batch_codes = adata.obs[self.batch_key].astype('category').cat.codes.values
            self.batch_tensor = torch.nn.functional.one_hot(
                torch.tensor(batch_codes, dtype=torch.long), num_classes=len(adata.obs[self.batch_key].unique())
            ).float()

    def __len__(self):
        return self.feat_rna.shape[0]

    def __getitem__(self, idx):
        if self.batch_key is not None:
            return (torch.FloatTensor(self.feat_rna[idx]), torch.FloatTensor(self.count_rna[idx]), torch.FloatTensor(self.size_rna[idx]),
                    torch.FloatTensor(self.feat_atac[idx]), torch.FloatTensor(self.count_atac[idx]), torch.FloatTensor(self.size_atac[idx]),
                    self.batch_tensor[idx])
        else:
            return (torch.FloatTensor(self.feat_rna[idx]), torch.FloatTensor(self.count_rna[idx]), torch.FloatTensor(self.size_rna[idx]),
                    torch.FloatTensor(self.feat_atac[idx]), torch.FloatTensor(self.count_atac[idx]), torch.FloatTensor(self.size_atac[idx]))