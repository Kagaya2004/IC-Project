import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class METRLADataset(Dataset):
    def __init__(self, window_size=12, horizon=1):
        # 1. Definição do caminho
        base_path = r'C:\IC\Projeto_V2\data\tratados'
        file_path = os.path.join(base_path, 'METR-LA_featured.csv')
        
        print(f"⏳ A carregar base de dados: {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Erro: O arquivo {file_path} não foi encontrado. Executa o featureEngineering.py primeiro!")

        # Leitura do CSV
        df = pd.read_csv(file_path, index_col=0)
        print(f"✅ Arquivo lido com sucesso! ({len(df)} registos encontrados)")

        # 2. Identificação de Colunas
        self.time_features_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.speed_cols = [c for c in df.columns if c not in self.time_features_cols]
        
        v_data = df[self.speed_cols].values.astype(np.float32)
        t_data = df[self.time_features_cols].values.astype(np.float32)

        # 3. Normalização Z-Score
        print("⚖️ A aplicar normalização Z-Score nas velocidades...")
        self.mean = v_data.mean()
        self.std = v_data.std()
        v_norm = (v_data - self.mean) / (self.std + 1e-8)

        # 4. Construção do Tensor Multidimensional [Tempo, Nós, Features]
        num_nodes = v_norm.shape[1]
        v_norm = np.expand_dims(v_norm, axis=-1)
        t_feats_expanded = np.tile(np.expand_dims(t_data, axis=1), (1, num_nodes, 1))
        
        # Concatenação final: [T, 207, 5]
        self.data = np.concatenate([v_norm, t_feats_expanded], axis=-1).astype(np.float32)
        
        self.window_size = window_size
        self.horizon = horizon

        print(f"📊 Processamento concluído: Tensor final com formato {self.data.shape}")
        print(f"   (Tempo: {self.data.shape[0]}, Sensores: {self.data.shape[1]}, Features: {self.data.shape[2]})")

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size + self.horizon - 1, :, 0]
        return torch.tensor(x), torch.tensor(y)

def preparar_loaders(batch_size=32):
    print("\n🚀 A preparar os DataLoaders para o treino...")
    
    dataset = METRLADataset()
    
    # Divisão 80/20
    train_size = int(0.8 * len(dataset))
    test_ds_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_ds_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"✨ DataLoaders prontos!")
    print(f"   - Treino: {len(train_ds)} amostras ({len(train_loader)} batches)")
    print(f"   - Teste: {len(test_ds)} amostras ({len(test_loader)} batches)\n")
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Teste rápido para ver as mensagens
    train_ld, test_ld = preparar_loaders()