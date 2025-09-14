import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class Masked_MSELoss(nn.Module):
    def __init__(self):
        super(Masked_MSELoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        squared_error = (pred - ground_truth).pow(2)
        masked_error = squared_error * mask
        loss = masked_error.sum() / (mask.sum() + 1e-8)  # 避免除以零
        
        return loss

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, pred: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)  # [num_stocks, num_stocks]
        true_diff = ground_truth.unsqueeze(0) - ground_truth.unsqueeze(1)  # [num_stocks, num_stocks]

        # 构造完整的mask矩阵
        mask_matrix = mask.unsqueeze(0) & mask.unsqueeze(1)  # [num_stocks, num_stocks]
        loss = torch.relu(-1 *pred_diff * true_diff * mask_matrix).mean()

        return loss

class RankLSTM(nn.Module):
    def __init__(self,
                hidden_size: int = 64,
                num_features: int = 5,
                alpha_rank: float = 1.0,
                batch_size: int = None):
        """RankLSTM模型
        
        Args:
            hidden_size: LSTM隐藏层大小
            num_features: 输入特征维度
            alpha: 排序损失的权重
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.alpha_rank = alpha_rank
        self.batch_size = batch_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # 预测层（带LeakyReLU激活）
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.mseloss_mask = Masked_MSELoss()
        self.rankloss = RankLoss()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播，支持完整处理或分批处理
        
        Args:
            batch: 包含以下键的字典:
                - feature: [num_stocks, seq_len, num_features] 特征序列
                - return_mask: [num_stocks] 有效性掩码
                - return: [num_stocks] 真实收益率
            batch_size: 可选，每批处理的股票数量。若不指定则一次处理所有股票。
                
        Returns:
            包含以下键的字典:
                - pred_return: [num_stocks] 预测收益率
                - loss: 标量，总损失
                - reg_loss: 标量，回归损失
                - rank_loss: 标量，排序损失
        """
        features = batch['feature']  # [num_stocks, seq_len, num_features]
        num_stocks = features.size(0)
            
        # 存储所有批次的预测结果
        all_predictions = []
        self.batch_size = num_stocks if self.batch_size is None else self.batch_size
        
        # 1. 分批处理所有股票，只进行前向传播
        for start_idx in range(0, num_stocks, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_stocks)

            batch_features = features[start_idx:end_idx]
            lstm_out, _ = self.lstm(batch_features) # 1. LSTM编码 [batch_size, seq_len, num_features] -> [batch_size, seq_len, hidden_size]
            seq_emb = lstm_out[:, -1, :]    # 2. 取最后时间步 [batch_size, hidden_size] 
            batch_predictions = self.dense(seq_emb).squeeze(-1)   # 3. 预测层 [batch_size, hidden_size] -> [batch_size]
            all_predictions.append(batch_predictions)
        
        # 2. 合并所有批次的预测结果，保持原始顺序
        predictions = torch.cat(all_predictions, dim=0)  # [num_stocks]
        output = {'pred_return': predictions}
        
        reg_loss = self.mseloss_mask(predictions, batch['return'], batch['return_mask'])
        rank_loss = self.rankloss(predictions, batch['return'], batch['return_mask'])

        total_loss = reg_loss + self.alpha_rank * rank_loss

        output.update({
            'loss': total_loss,
            'reg_loss': reg_loss,
            'rank_loss': rank_loss
            })
        
        return output

    def load_parameters(self, param_path: str, device: Optional[torch.device] = None):
        if device is None:
            state_dict = torch.load(param_path)
        else:
            state_dict = torch.load(param_path, map_location=device)
        
        self.load_state_dict(state_dict)
        print(f"模型参数已从 {param_path} 加载")

