import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class METRLADataset(Dataset):
    def __init__(self, window_size=12, horizon=1):
        # Altere o nome de 'METR-LA_final.csv' para 'METR-LA_series.csv'
        path = r'C:\IC\Projeto_V2\data\tratados\METR-LA_series.csv'
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado em: {path}")

        df = pd.read_csv(path)
        
        # O restante do código permanece igual...
        self.data = df.values.astype(np.float32)
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std
        
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon

    def __getitem__(self, idx):
        # X: [Janela, Sensores, 1 feature (velocidade)]
        x = self.data[idx : idx + self.window_size]
        x = torch.tensor(x).unsqueeze(-1) # Shape: [12, 207, 1]
        
        # Y: [Sensores] - Velocidade no tempo futuro
        y = self.data[idx + self.window_size + self.horizon - 1]
        y = torch.tensor(y)
        
        return x, y

def preparar_loaders(batch_size=32):
    dataset = METRLADataset()
    # Dividir em treino e teste (ex: 80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader