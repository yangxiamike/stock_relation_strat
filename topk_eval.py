import os
import yaml
import torch
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from model_rlstm import create_rank_lstm
from dataloader import create_dataloader

def evaluate_predictions(predictions: torch.Tensor, 
                       ground_truth: torch.Tensor, 
                       mask: torch.Tensor,
                       top_k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """评估预测结果，计算各项指标
    
    Args:
        predictions: 预测值 [num_stocks]
        ground_truth: 真实值 [num_stocks]
        mask: 有效数据掩码 [num_stocks]
        top_k_list: 要评估的top-k列表
    
    Returns:
        包含各项指标的字典:
        - mse: 带mask的均方误差
        - mrr_topk: 各个k值的平均倒数排名(Mean Reciprocal Rank)
        - btl_topk: 各个k值的多空策略收益(Back-testing Long)
    """
    metrics = {}
    
    # 转换为numpy array便于计算
    pred_np = predictions.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    mask_np = mask.cpu().numpy().astype(float)
    
    # 1. 计算MSE (带mask)
    metrics['mse'] = np.sum((pred_np - gt_np)**2 * mask_np) / np.sum(mask_np)
    
    # 2. 计算每个k值的指标
    max_k = max(top_k_list)
    
    # 获取真实值排序
    rank_gt = np.argsort(-gt_np)  # 降序排序
    
    # 获取预测值排序
    rank_pred = np.argsort(-pred_np)  # 降序排序
    
    # 对每个k值计算指标
    for k in top_k_list:
        # 获取top-k预测和真实值的集合
        pred_top_k = set()
        gt_top_k = set()
        
        # 收集真实值top-k（考虑mask）
        for idx in rank_gt:
            if len(gt_top_k) >= k:
                break
            if mask_np[idx] > 0.5:  # 只选择有效数据
                gt_top_k.add(idx)
                
        # 收集预测值top-k（考虑mask）
        for idx in rank_pred:
            if len(pred_top_k) >= k:
                break
            if mask_np[idx] > 0.5:  # 只选择有效数据
                pred_top_k.add(idx)
        
        # 计算MRR
        mrr = 0.0
        total_valid = 0
        for pred_idx in pred_top_k:
            # 找到该预测值在真实排序中的位置
            for rank, gt_idx in enumerate(rank_gt, 1):
                if gt_idx == pred_idx and mask_np[gt_idx] > 0.5:
                    mrr += 1.0 / rank
                    total_valid += 1
                    break
        
        metrics[f'mrr_top{k}'] = mrr / total_valid if total_valid > 0 else 0.0
        
        # 计算回测收益
        bt_return = 0.0
        valid_count = 0
        for pred_idx in pred_top_k:
            if mask_np[pred_idx] > 0.5:
                bt_return += gt_np[pred_idx]
                valid_count += 1
        
        metrics[f'btl_top{k}'] = bt_return / valid_count if valid_count > 0 else 0.0
    
    return metrics

def evaluate_model(
    model_path: str,
    feature_path: str,
    relation_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32
) -> Dict[str, float]:
    """评估模型性能
    
    Args:
        model_path: 模型权重文件路径
        feature_path: 特征数据文件路径
        relation_path: 关系矩阵文件路径（可选）
        config_path: 模型配置文件路径（可选）
        device: 运行设备
        batch_size: 批处理大小
    
    Returns:
        包含评估指标的字典
    """
    device = torch.device(device)
    
    # 加载配置
    if config_path is None:
        config_path = str(Path(model_path).parent / 'config.yaml')
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    features = np.load(feature_path)
    relation_matrix = None
    if relation_path:
        relation_matrix = sparse.load_npz(relation_path)
    
    # 创建测试数据加载器
    test_loader = create_dataloader(
        features=features,
        relation_matrix=relation_matrix,
        seq_len=config['data']['sequence_length'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型并加载权重
    model = create_rank_lstm(config['model'])
    model.load_parameters(model_path, device)
    model = model.to(device)
    model.eval()
    
    # 存储所有时间点的评估结果
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            # 将数据移动到设备上
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            output = model(batch)
            
            # 计算当前批次的指标
            metrics = evaluate_predictions(
                predictions=output['pred_return'],
                ground_truth=batch['return'],
                mask=batch['return_mask']
            )
            
            # 收集每个指标
            for name, value in metrics.items():
                all_metrics[name].append(value)
    
    # 计算所有时间点的平均指标
    final_metrics = {
        name: np.mean(values) for name, values in all_metrics.items()
    }
    
    # 添加每个指标的标准差
    for name, values in all_metrics.items():
        final_metrics[f'{name}_std'] = np.std(values)
    
    return final_metrics

if __name__ == '__main__':
    # 配置路径
    model_path = 'checkpoints/best_model.pth'
    feature_path = 'data/feature_matrix.npy'
    relation_path = 'data/relation_embedding_sparse.npz'
    config_path = 'configs/rank_lstm_config.yaml'
    
    # 评估模型
    metrics = evaluate_model(
        model_path=model_path,
        feature_path=feature_path,
        relation_path=relation_path,
        config_path=config_path
    )
    
    # 打印结果
    print("\n评估结果:")
    print("-" * 50)
    print("\n均值:")
    for name, value in sorted(metrics.items()):
        if not name.endswith('_std'):
            print(f"{name:15s}: {value:.4f} ± {metrics[name+'_std']:.4f}")
