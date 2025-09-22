import os
import os.path as osp
from typing import Callable, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data


class Stock_Data(Dataset):
	"""
	继承自Geometric Dataset的自定义股票数据实例。
	使用自己构建的图数据。
	"""

	def __init__(self, root: str = "data/Ashare100/", values_file_name: str = "values.csv", adj_file_name: str = "adj_stocks.npy", 
			  past_window: int = 25, future_window: int = 1, force_reload: bool = False, 
			  train_ratio: float = 0.7, val_ratio: float = 0.1, is_scale: bool = True):
		"""
		Args:
			root: 数据集根目录
			values_file_name: 包含股票历史数据的CSV文件名，[N, T, F] [N: 股票数量, T: 时间步数, F: 特征维度]
			adj_file_name: 包含邻接矩阵的NPY文件名, [N, N] [N: 股票数量]
			past_window: 输入序列长度
			future_window: 预测未来窗口长度
			force_reload: 是否强制重新处理数据
		"""
		
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		self.past_window = past_window
		self.future_window = future_window
		self.is_scale = is_scale
		assert 0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and train_ratio + val_ratio <= 1, \
		  "Train and validation ratios must be between 0 and 1, and their sum must not exceed 1"
		self.train_ratio, self.val_ratio = train_ratio, val_ratio

		# 调用父类构造函数，force_reload启动process预处理
		super().__init__(root, force_reload=force_reload)

		# 加载基础图数据
		graph = torch.load(self.processed_paths[0], weights_only=True)
		self._x = graph["x"].contiguous()          # [N,F,T]
		self._y = graph["y"].contiguous()          # [N,T]
		self._y_raw = self._y.clone()              # 备份原价
		self._edge_index = graph["edge_index"].contiguous()
		self._edge_weight = graph["edge_weight"].contiguous()
		self._T = self._x.shape[2]

		# 1) 样本起点切分（用于 DataLoader）
		S = self.len()                              # S = T - pw - fw + 1  T = T total
		train_num = int(S * self.train_ratio)
		val_num   = int(S * self.val_ratio)
		self.train_idx = torch.arange(0, train_num)
		self.val_idx   = torch.arange(train_num, train_num + val_num)
		self.test_idx  = torch.arange(train_num + val_num, S)

		# 2) 用训练段“时间位置掩码”做缩放（仅时间轴，避免泄漏）
		if self.is_scale:
			self.scale_by_train_time()

	def scale_by_train_time(self, eps: float = 1e-8, per_stock: bool = True):
		pw, fw = self.past_window, self.future_window
		train_num = len(self.train_idx)
		t = torch.arange(self._T) # T total
		# x 仅触及历史窗；y 触及历史+未来
		t_mask_x = t <= (train_num + pw - 1)
		t_mask_y = t <= (train_num + pw + fw - 1)

		self.mean_x = self._x[:, :, t_mask_x].mean(dim=2, keepdim=True)                         # [N,F,1]
		self.std_x  = self._x[:, :, t_mask_x].std(dim=2, keepdim=True, correction=0)
		self._x = (self._x - self.mean_x) / (self.std_x + eps)

		self.mean_y = self._y[:, t_mask_y].mean(dim=1, keepdim=True)                                # [N,1]
		self.std_y  = self._y[:, t_mask_y].std(dim=1, keepdim=True, correction=0)
		self._y = (self._y - self.mean_y) / (self.std_y + eps)

	@property
	def raw_file_names(self) -> List[str]:
		return [self.values_file_name, self.adj_file_name]

	@property
	def processed_file_names(self) -> List[str]:
		return ["graph.pt"]
	
	@property
	def num_node_features(self) -> int:
		return self._x.shape[1]
	
	@property
	def num_stocks(self) -> int:
		return self._x.shape[0]
	
	@property
	def num_time_steps(self) -> int:
		return self.past_window

	def process(self):
		"""
		初始化时会调用该方法来处理原始数据文件，并将处理后的数据保存到processed_dir目录中。
		"""
		x, y, edge_index, edge_weight = self.get_graph_in_pyg_format(
			values_path=os.path.join(self.root, 'raw', self.values_file_name),
			adj_path=os.path.join(self.root, 'raw', self.adj_file_name)
		)
		torch.save({"x": x, "y": y, "edge_index": edge_index, "edge_weight": edge_weight},
				   self.processed_paths[0])

	def len(self):
		return self._T - self.past_window - self.future_window + 1

	def get(self, idx: int):
		i, pw, fw = idx, self.past_window, self.future_window
		x = self._x[:, :, i:i+pw]                                 # [N,F,pw]
		y = self._y[:, i+pw:i+pw+fw]                              # [N,fw]
		y_raw = self._y_raw[:, i+pw:i+pw+fw]                      # [N,fw]
		# 价格信息，用于收益率计算
		price_curr = self._y_raw[:, i+pw-1]
		price_end  = self._y_raw[:, i+pw+fw-1]
		return Data(x=x, y=y, edge_index=self._edge_index, edge_weight=self._edge_weight,
					price_curr=price_curr, price_end=price_end, mean_y=self.mean_y, std_y=self.std_y,
					y_raw=y_raw)

	def get_graph_in_pyg_format(self, values_path: str, adj_path: str):
		values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
		adj = np.load(adj_path)
		N = adj.shape[0]
		F = values.shape[1]
		T = values.shape[0] // N
		arr = values.to_numpy(dtype=np.float32).reshape(N, T, F)
		x = torch.from_numpy(arr).transpose(1, 2).contiguous()                  # [N,F,T]
		y = torch.from_numpy(values[['Close']].to_numpy(dtype=np.float32).reshape(N, T))  # [N,T]
		nz = np.nonzero(adj)
		edge_index = torch.as_tensor(np.vstack(nz), dtype=torch.long)           # [2,E]
		edge_weight = torch.as_tensor(adj[nz].astype(np.float32))               # [E]
		return x, y, edge_index, edge_weight
	

if __name__ == "__main__":
	from torch_geometric.loader import DataLoader
	from torch.utils.data import Subset
	from torch_geometric.loader import DataLoader

	ds = Stock_Data(is_scale=True, force_reload=True)
	train_loader = DataLoader(Subset(ds, ds.train_idx.tolist()), batch_size=32, shuffle=False)
	val_loader   = DataLoader(Subset(ds, ds.val_idx.tolist()),   batch_size=32, shuffle=False)
	test_loader  = DataLoader(Subset(ds, ds.test_idx.tolist()),  batch_size=32, shuffle=False)

	for batch in train_loader:
		batch_x = batch.x.view((32, -1, batch.x.shape[-2], batch.x.shape[-1])).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3).contiguous()  # [B, N, F, T]
		data_x = ds[0].x
		import pdb; pdb.set_trace()