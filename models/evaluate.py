import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import confusion_matrix


def get_regression_error(model, dataloader):
    """
    Computes regression errors
    :param model: Model to test
    :param dataloader: Dataloader to test on
    :return: Mean squared error, rooted mean squared error, mean absolute error, mean relative error
    """
    mse = 0
    rmse = 0
    mae = 0
    mre = 0
    for data in dataloader:
        out = model(data.x, data.edge_index, data.edge_weight)
        mse += F.mse_loss(out, data.y).item()
        rmse += F.mse_loss(out, data.y).sqrt().item()
        mae += F.l1_loss(out, data.y).item()
        mre += (F.l1_loss(out, data.y) / data.y.abs().mean()).item()
    return mse / len(dataloader), rmse / len(dataloader), mae / len(dataloader), mre / len(dataloader)


def plot_regression(model, data, title=None):
    """
    Plot 4 graphs for regression
    :param model: Model to test
    :param data: Data to test on
    :param title: Title of the plot
    """
    model.eval()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)
    out = model(data.x, data.edge_index, data.edge_weight)
    stocks_idx = np.random.choice(data.x.shape[0] // (len(data.ptr) - 1), 4)

    preds = out.reshape(len(data.ptr) - 1, -1)
    target = data.y.reshape(len(data.ptr) - 1, -1)

    for idx, stock_idx in enumerate(stocks_idx):
        ax = axs[idx // 2, idx % 2]
        ax.plot(target[:, stock_idx].detach().numpy(), label="Real")
        ax.plot(preds[:, stock_idx].detach().numpy(), label="Predicted")
        ax.set_title(f"Stock {stock_idx}")
        ax.legend()

    plt.show()

def topk_return(pred, gt, k=5, base_price=None, is_pred_ret=False):
    """
    假设top-k多头策略，平均k等分购买topk资产，复利
    Compute the return of a top-k long strategy based on predicted prices
    :param base_price: Base price of the stocks [batch_size, num_stocks]
    :param pred: Predicted returns of the stocks [batch_size, num_stocks]
    :param gt: Ground truth returns of the stocks [batch_size, num_stocks]
    :param k: Number of stocks to long
    :return: Return of the strategy
    """
    batch_size = pred.shape[0]
    if not is_pred_ret:
        pred_ret = pred / base_price - 1  # 预测收益率 [batch_size, num_stocks]
        gt_ret = gt / base_price - 1      # 真实收益率 [batch_size, num_stocks]
    # Get top-k indices for each batch
    topk_indices = np.argsort(-pred_ret, axis=1)[:, :k]

    # Use advanced indexing to get the values for top-k and bottom-k indices
    batch_indices = np.arange(batch_size)[:, None]
    topk_rets = gt_ret[batch_indices, topk_indices]

    long_return = 1 + topk_rets.sum(axis=1)/k
    long_return = long_return.prod(axis=0)

    return long_return

def measure_accuracy(model, data):
    """
    Measure accuracy
    :param model: Model to test
    :param data: Data to test on
    :return: Accuracy
    """
    out = model(data.x, data.edge_index, data.edge_weight)
    if out.shape[1] == 1:  # Binary classification
        return (F.sigmoid(out).round() == data.y).sum().item() / len(data.y)
    else:  # Multi-class classification
        return (F.softmax(out, dim=-1).argmax(dim=-1) == data.y).sum().item() / len(data.y)


def get_confusion_matrix(model, data):
    """
    Get confusion matrix
    :param model: Model to test
    :param data: Data to test on
    :return: Confusion matrix
    """
    out = model(data.x, data.edge_index, data.edge_weight)
    if out.shape[1] == 1:
        y_pred = F.sigmoid(out).round().detach().numpy()
    else:
        y_pred = F.softmax(out, dim=-1).argmax(dim=-1).detach().numpy()
    y_true = data.y.detach().numpy()
    return confusion_matrix(y_true, y_pred)
