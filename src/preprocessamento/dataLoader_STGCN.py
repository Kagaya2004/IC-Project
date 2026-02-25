import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class METRLADataset(Dataset):
    def __init__(self, window_size=12, horizon=1):
        path = r'C:\IC\Projeto_V2\data\tratados\METR-LA_series.csv'
        df = pd.read_csv(path)
        
        # Identificar coluna de tempo (suporta 'Timestamp' ou 'Data_Hora')
        time_col = 'Timestamp' if 'Timestamp' in df.columns else 'Data_Hora'
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 1. Engenharia de Atributos Temporais (Cíclicos)
        hour = df[time_col].dt.hour
        dow = df[time_col].dt.dayofweek
        
        h_sin = np.sin(2 * np.pi * hour / 24).values
        h_cos = np.cos(2 * np.pi * hour / 24).values
        d_sin = np.sin(2 * np.pi * dow / 7).values
        d_cos = np.cos(2 * np.pi * dow / 7).values

        # 2. Processamento das Velocidades
        df_vel = df.drop(columns=[time_col])
        v_data = df_vel.values.astype(np.float32)
        
        # Normalização Z-Score (apenas na velocidade)
        self.mean = v_data.mean()
        self.std = v_data.std()
        v_norm = (v_data - self.mean) / self.std

        # 3. Construção do Tensor 3D [Tempo, Nós, Features]
        # v_norm: [T, N] -> [T, N, 1]
        v_norm = np.expand_dims(v_norm, axis=-1)
        
        # Features temporais: [T, 4] -> Expandir para cada nó [T, N, 4]
        t_feats = np.stack([h_sin, h_cos, d_sin, d_cos], axis=-1)
        t_feats = np.tile(np.expand_dims(t_feats, axis=1), (1, v_norm.shape[1], 1))
        
        # Concatenar: Velocidade (1) + Tempo (4) = 5 Features
        self.data = np.concatenate([v_norm, t_feats], axis=-1).astype(np.float32)
        
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        # Alvo (Y) é apenas a velocidade (índice 0) do passo futuro
        y = self.data[idx + self.window_size + self.horizon - 1, :, 0]
        return torch.tensor(x), torch.tensor(y)

def preparar_loaders(batch_size=32):
    dataset = METRLADataset()
    train_size = int(0.8 * len(dataset))
    test_ds_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_ds_size])
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False)