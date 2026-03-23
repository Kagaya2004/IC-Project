import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class LocalLSTM_TrafficDataset(Dataset):
    def __init__(self, window_size=12, horizon=12):
        base_path = r'C:\IC\Projeto_V2\data\tratados'
        file_path = os.path.join(base_path, 'METR-LA_featured.csv')
        
        print(f"⏳ Carregando base para LSTM Local: {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Erro: {file_path} não encontrado.")

        df = pd.read_csv(file_path, index_col=0)
        
        # 1. Separar Features
        self.time_features_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.speed_cols = [c for c in df.columns if c not in self.time_features_cols]
        
        v_data = df[self.speed_cols].values.astype(np.float32)
        t_data = df[self.time_features_cols].values.astype(np.float32)

        # 2. Normalização Z-Score
        self.mean = v_data.mean()
        self.std = v_data.std()
        v_norm = (v_data - self.mean) / (self.std + 1e-8)

        # 3. Construção do Tensor [Tempo, Nós, Features]
        self.num_nodes = v_norm.shape[1]
        v_norm = np.expand_dims(v_norm, axis=-1) # [T, N, 1]
        
        # Expandir features temporais para cada nó
        t_feats_expanded = np.tile(np.expand_dims(t_data, axis=1), (1, self.num_nodes, 1)) # [T, N, 4]
        
        # Concatenar: Velocidade + 4 Features Temporais = 5 Features
        self.data = np.concatenate([v_norm, t_feats_expanded], axis=-1).astype(np.float32)
        
        self.window_size = window_size
        self.horizon = horizon
        
        # Calcular quantos passos de tempo completos temos disponíveis
        self.valid_time_steps = len(self.data) - self.window_size - self.horizon

        print(f"✅ Processamento concluído. Features por nó: {self.data.shape[2]}")

    def __len__(self):
        # O pulo do gato: Total de amostras = (Passos de Tempo) x (Número de Sensores)
        return self.valid_time_steps * self.num_nodes

    def __getitem__(self, idx):
        # Mapear o índice linear (idx) para um momento no tempo e um sensor específico
        time_idx = idx // self.num_nodes
        node_idx = idx % self.num_nodes
        
        # X: [Window_Size, Features] -> Histórico de apenas UM nó (5 features)
        x = self.data[time_idx : time_idx + self.window_size, node_idx, :]
        
        # y: [Horizon] -> Alvo de velocidade no futuro para esse MESMO nó
        y = self.data[time_idx + self.window_size : time_idx + self.window_size + self.horizon, node_idx, 0]
        
        return torch.tensor(x), torch.tensor(y)

def preparar_loaders_lstm_local(batch_size=64, window_size=12, horizon=12):
    print(f"\n🚀 Preparando DataLoaders LSTM Local...")
    dataset = LocalLSTM_TrafficDataset(window_size=window_size, horizon=horizon)
    
    # Divisão sequencial (80% treino, 20% teste) preservando o eixo do tempo
    train_size = int(0.8 * len(dataset))
    
    # Como o idx avança tempo por tempo e nó por nó, um corte linear
    # garante que treinamos com o passado e testamos com o futuro.
    train_ds = torch.utils.data.Subset(dataset, range(0, train_size))
    test_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # Usando batch_sizes maiores, pois cada amostra agora é bem pequena (só 1 nó)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"✨ Concluído! Treino: {len(train_loader)} batches | Teste: {len(test_loader)} batches.\n")
    return train_loader, test_loader