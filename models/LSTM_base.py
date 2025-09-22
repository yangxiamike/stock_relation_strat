from torch import nn


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(MultiLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, edge_index=None, edge_weight=None):
        x = x.permute(0, 2, 1).contiguous()  # x : [num_stocks, num_features, seq_len] -> [num_stocks, seq_len, num_features]
        lstm_out, _ = self.lstm(x) # lstm_out : [num_stocks, seq_len, hidden_size]
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output