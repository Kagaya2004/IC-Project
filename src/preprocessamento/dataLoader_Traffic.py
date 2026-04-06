import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, window_size=12, horizon=12):
        """
        Dataset Universal para comparação entre ST-GCN e LSTM.
        window_size: quantos passos do passado olhamos (ex: 12 = 1 hora)
        horizon: quantos passos do futuro queremos prever (ex: 12 = 1 hora)
        """
        base_path = r'C:\IC\Projeto_V2\data\tratados'
        file_path = os.path.join(base_path, 'METR-LA_featured.csv')
        
        print(f"⏳ Carregando base universal para comparação: {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Erro: {file_path} não encontrado. Execute o pipeline de pré-processamento primeiro!")

        df = pd.read_csv(file_path, index_col=0)
        
        # 1. Separar Velocidades de Features Temporais
        self.time_features_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.speed_cols = [c for c in df.columns if c not in self.time_features_cols]
        
        v_data = df[self.speed_cols].values.astype(np.float32)
        t_data = df[self.time_features_cols].values.astype(np.float32)

        # 2. Normalização Z-Score (Salvamos para desnormalizar no avaliar_modelo)
        self.mean = v_data.mean()
        self.std = v_data.std()
        v_norm = (v_data - self.mean) / (self.std + 1e-8)

        # 3. Construção do Tensor [Tempo, Nós, Features]
        num_nodes = v_norm.shape[1]
        v_norm = np.expand_dims(v_norm, axis=-1) # [T, N, 1]
        
        # Expandir features temporais para cada nó
        t_feats_expanded = np.tile(np.expand_dims(t_data, axis=1), (1, num_nodes, 1)) # [T, N, 4]
        
        # Concatenar: Velocidade + 4 Features Temporais = 5 Features por nó
        self.data = np.concatenate([v_norm, t_feats_expanded], axis=-1).astype(np.float32)
        
        self.window_size = window_size
        self.horizon = horizon

        print(f"✅ Processamento concluído: {self.data.shape[0]} timesteps x {num_nodes} sensores.")

    def __len__(self):
        # Garantimos que haja espaço para a janela de entrada E a de saída (horizonte)
        return len(self.data) - self.window_size - self.horizon

    def __getitem__(self, idx):
        # X: [Window_Size, Nodes, Features] -> Entrada do modelo
        x = self.data[idx : idx + self.window_size]
        
        # y: [Horizon, Nodes] -> Alvo (apenas a velocidade, índice 0)
        # Importante: Ambos LSTM e ST-GCN neste experimento focarão no primeiro passo (t+1)
        y = self.data[idx + self.window_size : idx + self.window_size + self.horizon, :, 0]
        
        return torch.tensor(x), torch.tensor(y)

def preparar_loaders(batch_size=32, window_size=12, horizon=12):
    print(f"\n🚀 Preparando DataLoaders (Janela: {window_size}, Horizonte: {horizon})...")
    
    dataset = TrafficDataset(window_size=window_size, horizon=horizon)
    
    # --- Modificação: Split Cronológico ao invés de Random Split ---
    # Em séries temporais, é fundamental que o treino ocorra no passado e o teste no futuro.
    train_size = int(0.8 * len(dataset))
    
    # Criamos índices sequenciais
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Usamos o Subset para separar os dados mantendo a ordem do tempo
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)
    
    # O shuffle=True no treino é aceitável pois as janelas já contêm a sequência temporal interna (window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Teste NUNCA deve ser embaralhado para podermos plotar os gráficos de forma contínua depois
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"✨ Concluído! Treino: {len(train_loader)} batches | Teste: {len(test_loader)} batches.\n")
    
    return train_loader, test_loader