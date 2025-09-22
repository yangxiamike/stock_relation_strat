from datetime import datetime

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import trange

from models.evaluate import measure_accuracy, topk_return


def train(model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs, task_title="", measure_acc=False):
    """
    Train function for a regression / classification model
    :param model: Model to train
    :param optimizer: Optimizer to use (Adam, ...)
    :param criterion: Loss function to use (MSE, CrossEntropy, ...)
    :param train_dataloader: Train data loader
    :param test_dataloader: Test data loader
    :param num_epochs: Number of epochs to train on the train dataset
    :param task_title: Title of the tensorboard run
    :param measure_acc: Whether to measure accuracy or not (for classification tasks)
    """
    writer = SummaryWriter(f'runs/{task_title}_{datetime.now().strftime("%d_%m_%Hh%M")}_{model.__class__.__name__}')
    for epoch in (pbar := trange(num_epochs, desc="Epochs")):
        train_iteration(model, optimizer, pbar, criterion, train_dataloader, epoch, writer, measure_acc)
        test_iteration(model, criterion, test_dataloader, epoch, writer, measure_acc)


def test_iteration(model, criterion, test_dataloader, epoch, writer, measure_acc=False):
    """
    Test iteration
    :param model: Model to test
    :param criterion: Loss function to use (MSE, CrossEntropy, ...)
    :param test_dataloader: Test data loader
    :param epoch: Current epoch
    :param writer: Tensorboard writer
    :param measure_acc: Whether to measure accuracy or not (for classification tasks)
    """
    model.eval()
    ret_acc_1 = 1
    ret_acc_5 = 1

    for idx, data in enumerate(test_dataloader):
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out, data.y)
        writer.add_scalar("Loss/Test Loss", loss.item(), epoch * len(test_dataloader) + idx)

        # 计算topk收益率
        batch_size = data.num_graphs
        # mean_y = data.mean_y.view((batch_size, -1, 1))  # [B, N_stock, 1]
        # std_y = data.std_y.view((batch_size, -1, 1))    # [B, N_stock, 1]

        # future window
        fw = out.shape[-1]
        # 反标准化
        out_unscale = data.mean_y + out * (data.std_y + 1e-8)  # 反标准化
        # [B * N_stock, fw]
        out_unscale = out_unscale.detach().cpu().numpy()
        # [B, N_stock, fw]
        out_unscale = out_unscale.reshape((batch_size, -1, fw))
        # [B, N_stock]
        out_unscale = out_unscale[:, :, -1]  # Use the last time step's prediction
        # [B * N_stock, fw]
        gt = data.price_end.detach().cpu().numpy()  # Use price_end for return calculation T+future
        # [B, N_stock, fw]
        gt = gt.reshape((batch_size, -1, fw))
        gt = gt[:, :, -1]  # Use the last time step's ground truth
        # [B * N_stock, fw]
        base_price = data.price_curr.detach().cpu().numpy() # T+0
        # [B, N_stock, fw]
        base_price = base_price.reshape((batch_size, -1, fw))
        # [B, N_stock]
        base_price = base_price[:, :, -1]  # Use the last time step's base price


        ret_acc_5 *= topk_return(out_unscale, gt, base_price=base_price, is_pred_ret=False, k=5)
        ret_acc_1 *= topk_return(out_unscale, gt, base_price=base_price, is_pred_ret=False, k=1)
    
    ret_acc_1 -= 1 # Subtract initial capital
    ret_acc_5 -= 1 # Subtract initial capital
    writer.add_scalar("Return/Top1 Return", ret_acc_1, epoch)
    writer.add_scalar("Return/Top5 Return", ret_acc_5, epoch)

    if measure_acc:
        acc = measure_accuracy(model, data)
        writer.add_scalar("Accuracy/Test Accuracy", acc, epoch * len(test_dataloader) + idx)


def train_iteration(model, optimizer, pbar, criterion, train_dataloader, epoch, writer, measure_acc=False):
    """
    Train iteration
    :param model: Model to train
    :param optimizer: Optimizer to use (Adam, ...)
    :param pbar: tqdm progress bar
    :param criterion: Loss function to use (MSE, CrossEntropy, ...)
    :param train_dataloader: Train data loader
    :param epoch: Current epoch
    :param writer: Tensorboard writer
    :param measure_acc: Whether to measure accuracy or not (for classification tasks)
    """
    model.train()
    for idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"Batch": f"{(idx + 1) / len(train_dataloader) * 100:.1f}%, Loss: {loss.item():.6f}"})
        writer.add_scalar("Loss/Train Loss", loss.item(), epoch * len(train_dataloader) + idx)
    if measure_acc:
        acc = measure_accuracy(model, data)
        writer.add_scalar("Accuracy/Train Accuracy", acc, epoch * len(train_dataloader) + idx)


