import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import sparse
import pandas as pd
from typing import Dict, Tuple, Optional

class StockDataset(Dataset):
    def __init__(self, 
                 features: np.ndarray,
                 seq_len: int = 10,
                 is_norm = True):
        """股票数据集
        Args:
            features: 特征矩阵 [T, N, F]
            seq_len: 输入序列长度
        """
        self.features = features
        self.seq_len = seq_len
        self.is_norm = is_norm

    def normalize_features(self, seq_feature: np.ndarray) -> np.ndarray:
        mean = np.nanmean(seq_feature, axis=0, keepdims=True)  # [1, num_stocks, num_features]
        std = np.nanstd(seq_feature, axis=0, keepdims=True)    # [1, num_stocks, num_features]
        return (seq_feature - mean) / (std + 1e-8)
    
    def __len__(self) -> int:
        return self.features.shape[0] - self.seq_len + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本
        
        Args:
            idx: 样本索引
        Returns:
            包含以下键的字典：
            - feature: [seq_len, num_stocks, num_features]
            - mask: [num_stocks]
            - return: [num_stocks]
        """
        end_idx = min(idx + self.seq_len, self.features.shape[0] - 1)
        seq_feature = self.features[idx:end_idx, :, 1:-2]  # [seq_len, num_stocks, num_features]
        # Normalize features
        if self.is_norm:
            seq_feature = self.normalize_features(seq_feature)
        seq_feature = np.transpose(seq_feature, (1, 0, 2))  # [num_stocks, seq_len, num_features]

        # Generate random data with same shape as seq_feature
        # random_feature = torch.randn(*seq_feature.shape)  # [num_stocks, seq_len, num_features]
        future_return = self.features[end_idx, :, -1]  # [num_stocks]
        return_mask = self.features[end_idx, :, -2].astype(bool)  # [num_stocks]
        
        return {
            'feature': torch.FloatTensor(seq_feature),
            'return_mask': torch.BoolTensor(return_mask),
            'return': torch.FloatTensor(future_return)
        }
    

if __name__ == '__main__':
    # 读取特征矩阵
    features = np.load('data/feature_matrix.npy')
    
    # 创建数据集实例
    dataset = StockDataset(
        features=features,
        seq_len=10,
        is_norm=True
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=1)
    
    # 打印数据集大小
    print(f'Dataset size: {len(dataloader.dataset)}')
    
    # # 检查一个batch的数据
    # for batch in dataloader:
    #     print('\nBatch shapes:')
    #     batch['feature_s'] = batch['feature'].squeeze(0)
    #     assert((batch['feature_s'] - batch['feature'][0])[~torch.isnan(batch['feature_s'])].sum() < 1e-6)
    #     for k, v in batch.items():
    #         print(f'{k}: {v.shape}')
    #     break

    # Check all batches for masked return values
    for batch in dataloader:
        masked_sum = (batch['return'] * batch['return_mask']).sum()
        print(batch['return_mask'].sum(), masked_sum)
