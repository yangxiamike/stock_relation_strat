import os
import yaml
import torch
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

from model_rlstm import create_rank_lstm
from dataloader import create_dataloader

def calculate_rank_metrics(predictions: np.ndarray, 
                         actual_values: np.ndarray, 
                         mask: np.ndarray,
                         top_k: int = 10) -> Dict[str, float]:
    """计算排序相关的评估指标
    
    Args:
        predictions: 预测值
        actual_values: 真实值
        mask: 有效数据掩码
        top_k: 计算top-k准确率的k值
        
    Returns:
        包含各项指标的字典
    """
    # 只考虑有效数据
    valid_pred = predictions[mask]
    valid_actual = actual_values[mask]
    
    # 计算预测值和真实值的排序
    pred_ranks = (-valid_pred).argsort()
    true_ranks = (-valid_actual).argsort()
    
    # 计算top-k准确率
    top_k_pred = set(pred_ranks[:top_k])
    top_k_true = set(true_ranks[:top_k])
    top_k_accuracy = len(top_k_pred & top_k_true) / top_k
    
    # 计算二分类指标（以实际收益率中位数为阈值）
    threshold = np.median(valid_actual)
    pred_binary = (valid_pred > np.median(valid_pred)).astype(int)
    true_binary = (valid_actual > threshold).astype(int)
    
    precision = precision_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary)
    
    # 计算相关系数
    correlation = np.corrcoef(valid_pred, valid_actual)[0, 1]
    
    return {
        'top_k_accuracy': top_k_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correlation': correlation
    }

def evaluate_model(
    model_path: str,
    feature_path: str,
    relation_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 128
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
    
    # 创建数据加载器
    test_loader = create_dataloader(
        features=features,
        relation_matrix=relation_matrix,
        seq_len=config['data']['sequence_length'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 创建并加载模型
    model = create_rank_lstm(config['model'])
    model.load_parameters(model_path, device)
    model = model.to(device)
    model.eval()
    
    predictions = []
    actual_values = []
    masks = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 将数据移动到设备上
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            output = model(batch)
            
            # 收集结果
            predictions.append(output['pred_return'].cpu().numpy())
            actual_values.append(batch['return'].cpu().numpy())
            masks.append(batch['return_mask'].cpu().numpy())
    
    # 合并批次结果
    predictions = np.concatenate(predictions)
    actual_values = np.concatenate(actual_values)
    masks = np.concatenate(masks)
    
    # 计算指标
    metrics = calculate_rank_metrics(predictions, actual_values, masks)
    
    # 计算MSE和MAE（只考虑有效数据）
    valid_pred = predictions[masks]
    valid_actual = actual_values[masks]
    
    mse = np.mean((valid_pred - valid_actual) ** 2)
    mae = np.mean(np.abs(valid_pred - valid_actual))
    
    metrics.update({
        'mse': mse,
        'mae': mae
    })
    
    return metrics

if __name__ == '__main__':
    # 示例配置
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
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")