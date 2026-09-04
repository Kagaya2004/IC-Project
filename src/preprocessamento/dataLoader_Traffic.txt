import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, window_size=12, horizon=12):
        # Raiz de projeto
        base_path = r'C:\IC\Projeto_V2\data\tratados'
        # Caminho do arquivo final após a aplicação da Engenharia de Atriibutos/Features Engineering
        file_path = os.path.join(base_path, 'METR-LA_featured.csv')
        
        print(f"A carregar base universal para comparação: {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Erro: {file_path} não encontrado. Execute o pré-processamento primeiro!")

        # Carrega o dataset com a coluna de timestamp como índice
        df = pd.read_csv(file_path, index_col=0)
        
        """
        Separação de Velocidades e Atributos Temporais:
        Identifica as colunas de velocidades e as colunas de atributos temmporais (hora e dia da semana)
        Converte tudo em matrizes NumPy do tipo float32, otimizando processamento
        """
        self.time_features_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.speed_cols = [c for c in df.columns if c not in self.time_features_cols]
        v_data = df[self.speed_cols].values.astype(np.float32)
        t_data = df[self.time_features_cols].values.astype(np.float32)

        """
        Normalização Z-Score (Salva para desnormalizar no avaliar_modelo.py):
        Centraliza as velocidades na média 0 com desvio padrão 1
        """
        self.mean = v_data.mean()
        self.std = v_data.std()
        v_norm = (v_data - self.mean) / (self.std + 1e-8)

        # Coleta a quantidade de sensores (nós) da matriz normalizada
        num_nodes = v_norm.shape[1]
        # Expande a dimensão de features para incluir a velocidade normalizada como o primeiro atributo de cada nó
        v_norm = np.expand_dims(v_norm, axis=-1) # [T, N, 1]
        
        # Expandir atributos temporais para cada nó, permitindo que cada sensor tenha acesso às informações de hora e dia da semana, essenciais para a previsão
        t_feats_expanded = np.tile(np.expand_dims(t_data, axis=1), (1, num_nodes, 1)) # [T, N, 4]
        
        # Concatena a Velocidade + 4 Features Temporais, formando um novo tensor com 5 atributos temporais
        self.data = np.concatenate([v_norm, t_feats_expanded], axis=-1).astype(np.float32)
        
        # Define os parâmetros de janela e horizonte para a construção das sequências de entrada e saída do modelo
        self.window_size = window_size
        self.horizon = horizon

        print(f"Processamento concluído: {self.data.shape[0]} timesteps x {num_nodes} sensores.")

    def __len__(self):
        # Garantimos que haja espaço para a janela de entrada E a de saída (horizonte)
        return len(self.data) - self.window_size - self.horizon

    def __getitem__(self, idx):
        # X: [Window_Size, Nodes, Features] -> Entrada do modelo
        x = self.data[idx : idx + self.window_size]
        
        # y: [Horizon, Nodes] -> Alvo (apenas a velocidade, índice 0)
        # Importante: DCRNN precisa da sequência, STGCN pode pegar apenas y[0]
        y = self.data[idx + self.window_size : idx + self.window_size + self.horizon, :, 0]
        
        return torch.tensor(x), torch.tensor(y)

def preparar_loaders(batch_size=32, window_size=12, horizon=12):
    print(f"\nPreparando DataLoaders (Janela: {window_size}, Horizonte: {horizon})...")
    
    dataset = TrafficDataset(window_size=window_size, horizon=horizon)
    
    # Divisão fixa para garantir comparação justa (80% treino, 20% teste)
    train_size = int(0.8 * len(dataset))
    
    # Substituição do random_split pelo Subset linear
    # Garante que o modelo treine estritamente com o passado e teste com o futuro
    train_ds = torch.utils.data.Subset(dataset, range(0, train_size))
    test_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # O shuffle=True no treino é mantido para embaralhar apenas os lotes dentro do 
    # período de passado isolado, o que ajuda na convergência sem causar vazamento.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"Concluído! Treino: {len(train_loader)} batches | Teste: {len(test_loader)} batches.\n")
    
    return train_loader, test_loader