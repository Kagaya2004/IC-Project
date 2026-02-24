import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCN_TrafficModel(nn.Module):
    def __init__(self, num_features, hidden_dim=64, output_dim=1, dropout=0.2):
        """
        Arquitetura ST-GCN adaptada para o METR-LA.
        Entrada esperada: [Batch, Time, Nodes, Features]
        """
        super(STGCN_TrafficModel, self).__init__()
        
        self.hidden_dim = hidden_dim

        # 1. Embedding: Projeta as features de entrada para a dimensão oculta
        self.embedding = nn.Linear(num_features, hidden_dim)
        
        # 2. Camada Espacial: Graph Convolutional Network
        self.spatial_gcn = GCNConv(hidden_dim, hidden_dim)
        
        # 3. Camada Temporal: Gated Recurrent Unit (GRU)
        # Batch_first=False é mais eficiente para o PyTorch internamente
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        
        # 4. Regressão: Saída final (previsão de velocidade)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: [Batch, T, Nodes, Features]
        batch_size, T, num_nodes, num_features = x.shape
        
        # --- PASSO 1: Embedding ---
        x = self.embedding(x) # [B, T, N, Hidden]
        
        # --- PASSO 2: Bloco Espacial (GCN aplicada em cada timestamp) ---
        out_spatial = []
        for t in range(T):
            # Extrai o frame t: [B, N, Hidden]
            xt = x[:, t, :, :] 
            # Achata o batch para processar como um grafo gigante [B*N, Hidden]
            xt_flat = xt.reshape(-1, self.hidden_dim)
            
            # Aplica GCN (vizinhança espacial)
            h = self.spatial_gcn(xt_flat, edge_index, edge_weight)
            
            # Retorna para o shape original e guarda
            out_spatial.append(h.reshape(batch_size, 1, num_nodes, self.hidden_dim))
        
        x = torch.cat(out_spatial, dim=1) # [B, T, N, Hidden]
        
        # --- PASSO 3: Bloco Temporal (GRU) ---
        # Reorganizamos para que a GRU veja a sequência temporal [T] para cada nó [B*N]
        # De [B, T, N, H] para [T, B*N, H]
        x = x.permute(1, 0, 2, 3).reshape(T, batch_size * num_nodes, self.hidden_dim)
        
        # h_n contém o "resumo" da sequência temporal de cada sensor
        _, h_n = self.temporal_gru(x) # h_n: [1, B*N, Hidden]
        
        # --- PASSO 4: Regressão Final ---
        # Pegamos o estado oculto final e passamos pela linear
        out = self.regressor(h_n.squeeze(0)) # [B*N, 1]
        
        # Retornamos para [Batch, Nodes] (previsão de velocidade para cada sensor)
        return out.reshape(batch_size, num_nodes)

if __name__ == "__main__":
    # Teste rápido de integridade
    model = STGCN_TrafficModel(num_features=1)
    dummy_x = torch.randn(16, 12, 207, 1) # Batch 16, 1h de dados (12*5min), 207 sensores
    dummy_edge = torch.randint(0, 207, (2, 500))
    output = model(dummy_x, dummy_edge)
    print(f"Sucesso! Shape da saída: {output.shape}") # Esperado: [16, 207]