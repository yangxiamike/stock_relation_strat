from datetime import datetime
import pytest
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from models.train import test_iteration

from Stock_data import Stock_Data
from models.LSTM_base import MultiLayerLSTM  # 你的 LSTM 基线

def test_test_iteration_with_stock_data():
    import os
    import torch
    import pytest
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from models.train import test_iteration

    # 构建数据集与 DataLoader
    ds = Stock_Data(root="data/Ashare100", past_window=25, future_window=1,
                    force_reload=False, train_ratio=0.6, val_ratio=0.2, is_scale=True)

    # 取少量训练样本，保证测试快速稳定
    take = min(8, len(ds.train_idx))
    assert take > 0, "No train samples in dataset."
    train_loader = DataLoader(Subset(ds, ds.train_idx[:take].tolist()),
                              batch_size=4, shuffle=False)

    # 模型与损失
    model = MultiLayerLSTM(input_size=ds.num_node_features,
                           hidden_size=16, num_layers=2,
                           output_size=1, dropout=0.3)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=f'tests/runs/stock_{datetime.now().strftime("%d_%m_%Hh%M")}')

    # 执行一轮 test_iteration（这里用训练 loader，只是为了走通流程）
    test_iteration(model=model,
                   criterion=criterion,
                   test_dataloader=train_loader,
                   epoch=1,
                   writer=writer,
                   measure_acc=False)

    writer.close()

    # 基础断言：流程应无异常
    assert True

if __name__ == "__main__":
    test_test_iteration_with_stock_data()