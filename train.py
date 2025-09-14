import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model_rlstm import RankLSTM
from torch.utils.data import DataLoader, random_split
from dataloader import StockDataset


class Trainer:
    def __init__(
        self,
        config_path: str,
        checkpoints_dir: str = "checkpoints",
        tensorboard_dir: str = "runs",
    ):

        # 加载配置
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_config = self.config['data']
        self.training_config = self.config['training']
        self.model_config = self.config['model']

        # 加载数据
        feature_path = self.data_config['feature_path']
        relation_path = self.data_config['relation_path']
        use_relation = self.data_config['use_relation']
        # 读取特征数据
        features = np.load(feature_path)
        
        # 读取关系数据
        relation_matrix = None
        if use_relation:
            relation_matrix = sparse.load_npz(relation_path)
        # 创建数据集
        dataset = StockDataset(
            features=features,
            seq_len=self.data_config['sequence_length'],
            is_norm=True
        )
        
        # 计算数据集划分
        train_size = self.data_config['split_info']['num_train']
        valid_size = self.data_config['split_info']['num_valid']
        test_size = self.data_config['split_info']['num_test']

        # 创建数据集划分
        # Sequential split for time series data
        train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        valid_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + valid_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size + valid_size, train_size + valid_size + test_size))
  
        # 创建数据加载器
        self.dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=self.training_config['num_workers'],
                pin_memory=True
            ),
            'valid': DataLoader(
                valid_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.training_config['num_workers'],
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.training_config['num_workers'],
                pin_memory=True
            )
        }
        
        # 创建模型
        self.model = RankLSTM(
                    hidden_size=self.model_config['hidden_size'],
                    num_features=self.model_config['num_features'],
                    alpha_rank=self.model_config['alpha_rank']
                )
        self.model = self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            **self.training_config['optimizer_args']
        )
        
        # 创建学习率调度器
        scheduler_config = self.training_config['scheduler']
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            verbose=True
        )
        
        # 创建Tensorboard writer
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(tensorboard_dir, current_time)

        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 训练状态
        self.epoch = 0
        self.best_valid_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
        
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点
        
        Args:
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = self.checkpoints_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.checkpoints_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_valid_loss = checkpoint['best_valid_loss']

    def _run_epoch(self, data_loader: DataLoader, is_training: bool) -> Dict[str, float]:
        """运行一个epoch
        
        Args:
            data_loader: 数据加载器
            is_training: 是否为训练模式
            
        Returns:
            包含指标的字典
        """
        self.model.train(is_training)
        
        # 在验证模式下不计算梯度
        with torch.set_grad_enabled(is_training):
            total_loss = 0
            total_reg_loss = 0
            total_rank_loss = 0
            num_batches = 0
            
            for batch in data_loader:

                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
                if is_training:
                    self.optimizer.zero_grad()
                # 截面数据，去掉batch维度，batch维度实际是股票数量
                batch['feature'] = batch['feature'].squeeze(0)
                batch['return_mask'] = batch['return_mask'].squeeze(0)
                batch['return'] = batch['return'].squeeze(0)
                output = self.model(batch)
                
                if is_training:
                    # Print LSTM gradients before backward pass
                    
                    output['loss'].backward()
                    # for name, param in self.model.named_parameters():
                    #     if 'lstm' in name and param.grad is not None:
                    #         print(f'After backward - {name} grad: {param.grad.norm().item()}')
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                # 累积损失
                total_loss += output['loss'].item()
                total_reg_loss += output['reg_loss'].item()
                total_rank_loss += output['rank_loss'].item()
                num_batches += 1
                
            # 计算平均损失
            metrics = {
                'loss': total_loss / num_batches,
                'reg_loss': total_reg_loss / num_batches,
                'rank_loss': total_rank_loss / num_batches
            }
            
            return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        return self._run_epoch(train_loader, is_training=True)
    
    def validate(self, valid_loader: DataLoader) -> Dict[str, float]:
        return self._run_epoch(valid_loader, is_training=False)
    
    def train(self):
        """完整的训练流程
        """
        max_epochs = self.training_config['max_epochs']
        early_stopping_patience = self.training_config['early_stopping_patience']
        
        print(f"开始训练，设备: {self.device}")
        
        while self.epoch < max_epochs:
            self.epoch += 1
            
            train_metrics = self.train_epoch(self.dataloaders['train'])
            valid_metrics = self.validate(self.dataloaders['valid'])
            
            self.scheduler.step(valid_metrics['loss'])
            
            # 记录到tensorboard
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, self.epoch)
            for name, value in valid_metrics.items():
                self.writer.add_scalar(f'valid/{name}', value, self.epoch)
            
            # 打印训练信息
            print(f"Epoch {self.epoch}/{max_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.8f}")
            print(f"Valid Loss: {valid_metrics['loss']:.8f}")
            
            # 检查是否为最佳模型
            if valid_metrics['loss'] < self.best_valid_loss:
                self.best_valid_loss = valid_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"发现最佳模型，验证损失: {self.best_valid_loss:.4f}")
            else:
                self.patience_counter += 1
                
            # 保存最新检查点
            self.save_checkpoint()
            
            # Early stopping检查
            if self.patience_counter >= early_stopping_patience:
                print(f"验证损失 {early_stopping_patience} 个epoch没有改善，停止训练")
                break
                
        print("训练完成!")
        print(f"最佳验证损失: {self.best_valid_loss:.4f}")
        print(f"最佳模型保存在: {self.best_model_path}")
        
        # 关闭tensorboard writer
        self.writer.close()


if __name__ == "__main__":
    # 配置文件路径
    config_path = "configs/rank_lstm_config.yaml"
    
    # 创建训练器
    trainer = Trainer(config_path)
    
    # 开始训练
    trainer.train()
