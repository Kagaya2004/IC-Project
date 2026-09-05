import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_TrafficModel(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2, output_dim=12, dropout=0.2):
        super(LSTM_TrafficModel, self).__init__()
        self.output_dim = output_dim
        
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.ln_embed = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x recebido: [Batch, 12, 207, 5]
        batch_size, seq_len, num_nodes, num_features = x.shape

        # Reorganiza para agrupar nós no batch: [Batch, 207, 12, 5]
        x = x.permute(0, 2, 1, 3).contiguous()
        # Achata para 3D: [Batch * 207, 12, 5]
        x = x.view(batch_size * num_nodes, seq_len, num_features)

        # Camada de embedding linear + LayerNorm
        x = F.relu(self.ln_embed(self.embedding(x)))

        # LSTM temporal independente por sensor
        out, _ = self.lstm(x)

        # Pega o último timestep da sequência
        last_out = self.ln_lstm(out[:, -1, :])

        # Regressão dos 12 passos à frente: [Batch * 207, 12]
        out_reg = self.regressor(last_out)

        # Remodela para [Batch, 207, 12]
        out_reg = out_reg.view(batch_size, num_nodes, self.output_dim)

        # Permuta para [Batch, 12, 207] para casar exatamente com a dimensão do tensor y
        return out_reg.permute(0, 2, 1).contiguous()