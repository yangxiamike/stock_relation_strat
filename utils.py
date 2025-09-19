import pandas as pd
import scipy as sp
import numpy as np

def classify_stock_exchange(stock_codes):
    """
    根据 A 股代码将股票分类为北交所或沪深交易所。

    参数:
        stock_codes (iterable of str/int or pd.Series): 股票代码列表或 Series

    返回:
        pd.DataFrame: 包含索引为 'code'，列 'exchange' 的 DataFrame，
                      exchange 值为 '北交所' 或 '沪深所'。
    """
    codes = []
    exchanges = []

    for code in stock_codes:
        # 保留前导零，统一为字符串
        code_str = str(code)
        codes.append(code_str)

        if code_str.startswith(('82', '83', '87', '88', '920', '430')):
            exch = '北交所'
        elif code_str.startswith(('300', '301')):
            exch = '创业板'
        elif code_str.startswith(('60', '68', '900')):
            exch = '上交所'
        elif code_str.startswith(('00', '002', '003', '004', '200')):
            exch = '深交所'
        else:
            exch = '未知'

        exchanges.append(exch)

    df = pd.DataFrame({'code': codes, 'exchange': exchanges})
    # 确保 code 列为字符串类型以保留前导零
    df['code'] = df['code'].astype(str)
    return df

def pre_adj_factor(df, ticker_price, ticker_factor):   
    if ticker_factor not in df.columns:
        df['adj_close'] = df[ticker_price]
    else:
    # 计算前复权收盘价
        newest_factor = df[ticker_factor].iloc[-1]
        df['adj_close'] = df[ticker_price] * df[ticker_factor] / newest_factor

    return df


def save_sparse_relation_embedding(relation_embedding, filename):
    """
    将三维关系嵌入矩阵稀疏存储为.npz文件
    """
    # 合并最后一维为列，变成2D矩阵
    n, m, d = relation_embedding.shape
    relation_2d = relation_embedding.reshape(n * m, d)
    sparse_mat = sp.csr_matrix(relation_2d)
    sp.save_npz(filename, sparse_mat)

def load_sparse_relation_embedding(filename, shape):
    """
    读取稀疏存储的关系嵌入矩阵，并还原为原始三维形状
    """
    sparse_mat = sp.load_npz(filename)
    relation_2d = sparse_mat.toarray()
    n, m, d = shape
    return relation_2d.reshape(n, m, d)


def build_relation_embedding(stock_codes, industry_dict):
    # 1. 股票索引映射
    ticker_index = {code: idx for idx, code in enumerate(stock_codes)}
    n = len(stock_codes)
    
    # 2. 只保留包含2只及以上股票的行业
    valid_industries = {k: [c for c in v if c in ticker_index] for k, v in industry_dict.items()}
    valid_industries = {k: v for k, v in valid_industries.items() if len(v) > 1}
    valid_industry_index = {ind: idx for idx, ind in enumerate(valid_industries)}
    valid_industry_count = len(valid_industry_index)
    
    # 3. one-hot行业嵌入 (行业数+1) 包括自环信息
    one_hot_industry_embedding = np.identity(valid_industry_count + 1)
    
    # 4. 初始化关系矩阵 [n, n, 行业数+1]
    ticker_relation_embedding = np.zeros([n, n, valid_industry_count + 1], dtype=np.bool_)  # 或np.uint8
    
    # 5. 填充关系矩阵
    for industry, ind_idx in valid_industry_index.items():
        stocks_in_industry = valid_industries[industry]
        for i in range(len(stocks_in_industry)):
            idx1 = ticker_index[stocks_in_industry[i]]
            # 自环
            ticker_relation_embedding[idx1][idx1] = one_hot_industry_embedding[ind_idx]
            ticker_relation_embedding[idx1][idx1][-1] = 1  # 自环标记
            for j in range(i+1, len(stocks_in_industry)):
                idx2 = ticker_index[stocks_in_industry[j]]
                ticker_relation_embedding[idx1][idx2] = one_hot_industry_embedding[ind_idx]
                ticker_relation_embedding[idx2][idx1] = one_hot_industry_embedding[ind_idx]
    return ticker_relation_embedding
