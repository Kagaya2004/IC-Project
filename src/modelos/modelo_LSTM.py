import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, horizon=12):
        """
        Modelo LSTM Baseline para previsão de séries temporais de tráfego.
        
        Args:
            input_dim (int): Número de features por nó (ex: velocidade, fluxo, hora do dia = 5).
            hidden_dim (int): Número de neurônios nas camadas ocultas do LSTM.
            num_layers (int): Quantidade de camadas LSTM empilhadas.
            horizon (int): Quantos passos no futuro queremos prever (ex: 12 passos).
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        
        # O LSTM processará a série temporal de cada sensor
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True # O formato de entrada será (batch, seq_len, features)
        )
        
        # Camada totalmente conectada para projetar o último estado oculto no horizonte futuro
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        """
        Passo forward do modelo.
        
        Args:
            x (Tensor): Tensor de entrada com formato (batch_size, in_steps, num_nodes, num_features).
            
        Returns:
            Tensor: Previsões com formato (batch_size, horizon, num_nodes).
        """
        batch_size, in_steps, num_nodes, num_features = x.shape
        
        # 1. Remodelagem Inicial
        # Como o LSTM padrão não entende a dimensão espacial (num_nodes), 
        # nós fundimos o batch_size com o num_nodes. Assim, o modelo trata
        # cada nó de cada lote como uma série temporal independente.
        
        # Passa de (B, T, N, F) para (B, N, T, F)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # Passa para (B * N, T, F) -> Formato perfeito para o nn.LSTM
        x = x.view(batch_size * num_nodes, in_steps, num_features)
        
        # 2. Passagem pelo LSTM
        # lstm_out conterá as saídas de todos os passos de tempo: (B * N, T, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 3. Extração do Contexto Final
        # Pegamos apenas a saída do último passo de tempo (índice -1) da sequência
        last_out = lstm_out[:, -1, :] # Formato: (B * N, hidden_dim)
        
        # 4. Projeção Futura (Previsão)
        # Passamos pela camada linear para gerar os 'horizon' passos de uma vez
        out = self.fc(last_out) # Formato: (B * N, horizon)
        
        # 5. Remodelagem Final (Desfazendo a fusão do passo 1)
        # Separa novamente o batch_size e o num_nodes: passa para (B, N, horizon)
        out = out.view(batch_size, num_nodes, self.horizon)
        
        # Reordena para o formato padrão esperado pelo script de avaliação (B, horizon, N)
        out = out.permute(0, 2, 1).contiguous()
        
        return out