import torch
import os
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self):
        base_dir = r'C:\IC\Projeto_V2\data\tratados'
        # Adicionamos weights_only=True para segurança
        self.x = torch.load(os.path.join(base_dir, 'x_features.pt'), weights_only=True)
        # O alvo (target) é a velocidade (coluna Traffic_Speed_kmh)
        # No seu tensor de 21 colunas, precisamos saber o índice exato (geralmente a 3ª ou última)
        self.y = self.x[:, 3] # Ajuste o índice se a velocidade não for a coluna 3

    def __len__(self):
        return 1 # Temos 1 grande snapshot da cidade

    def __getitem__(self, idx):
        # Pega o snapshot completo dos sensores
        x_snapshot = self.x.unsqueeze(0) # Adiciona dimensão de tempo (T=1)
        
        # Define a velocidade como alvo (ex: coluna 3)
        y_target = self.x[:, 3] 
        
        return x_snapshot, y_target

def preparar_loaders(batch_size=1):
    dataset = TrafficDataset()
    # Para grafos grandes, usamos batch_size=1 (o lote é o próprio grafo)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, loader