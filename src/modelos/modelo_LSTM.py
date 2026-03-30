import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_TrafficModel(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2, output_dim=12, dropout=0.2):
        super(LSTM_TrafficModel, self).__init__()
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.ln_embed = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        x = F.relu(self.ln_embed(self.embedding(x)))
        out, _ = self.lstm(x)
        last_out = self.ln_lstm(out[:, -1, :])
        return self.regressor(last_out) # [Batch*207, 12]