import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCN_TrafficModel(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, output_dim=1, dropout=0.2):
        """
        Arquitetura ST-GCN Refinada.
        num_features: 5 (Velocidade + 4 Features Temporais)
        """
        super(STGCN_TrafficModel, self).__init__()
        
        self.hidden_dim = hidden_dim

        # 1. Embedding + Normalização
        # A LayerNorm ajuda a estabilizar as 5 features diferentes antes da GCN
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.ln_embed = nn.LayerNorm(hidden_dim)
        
        # 2. Camada Espacial: Graph Convolutional Network
        self.spatial_gcn = GCNConv(hidden_dim, hidden_dim)
        
        # 3. Camada Temporal: GRU
        # A GRU processa a sequência de 12 timesteps (1 hora)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.ln_temporal = nn.LayerNorm(hidden_dim)
        
        # 4. Regressão Final
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # x shape esperado: [Batch, Time, Nodes, Features]
        batch_size, T, num_nodes, num_features = x.shape
        
        # --- PASSO 1: Embedding & Normalização ---
        x = self.embedding(x)
        x = self.ln_embed(x)
        x = F.relu(x)
        
        # --- PASSO 2: Bloco Espacial (GCN) ---
        # Processamos cada passo de tempo T como um estado do grafo
        out_spatial = []
        for t in range(T):
            xt = x[:, t, :, :] # [B, N, Hidden]
            
            # Para a GCNConv processar o batch corretamente, 
            # tratamos o batch como subgrafos independentes
            # Shape para a GCN: [B*N, Hidden]
            xt_flat = xt.reshape(-1, self.hidden_dim)
            
            # Aplica GCN + Ativação
            h = self.spatial_gcn(xt_flat, edge_index, edge_weight)
            h = F.relu(h)
            
            # Retorna para o shape original [B, 1, N, Hidden]
            out_spatial.append(h.reshape(batch_size, 1, num_nodes, self.hidden_dim))
        
        x = torch.cat(out_spatial, dim=1) # [B, T, N, Hidden]
        
        # --- PASSO 3: Bloco Temporal (GRU) ---
        # Mudamos de [Batch, Time, Nodes, Hidden] para [Time, Batch*Nodes, Hidden]
        # Isso faz a GRU ver a evolução temporal de cada sensor separadamente
        x = x.permute(1, 0, 2, 3).reshape(T, batch_size * num_nodes, self.hidden_dim)
        
        # Pegamos apenas o último estado oculto (o resumo do tempo)
        _, h_n = self.temporal_gru(x) # h_n: [1, B*N, Hidden]
        h_n = self.ln_temporal(h_n.squeeze(0)) # [B*N, Hidden]
        
        # --- PASSO 4: Regressão Final ---
        out = self.regressor(h_n) # [B*N, 1]
        
        # Saída Final: [Batch, Nodes] (Velocidade prevista para cada sensor)
        return out.reshape(batch_size, num_nodes)

if __name__ == "__main__":
    # Teste de Integridade com as novas 5 features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulação: 16 batches, 12 timesteps, 207 sensores, 5 features
    model = STGCN_TrafficModel(num_features=5).to(device)
    dummy_x = torch.randn(16, 12, 207, 5).to(device)
    dummy_edge = torch.randint(0, 207, (2, 500)).to(device)
    
    print(f"🚀 Testando modelo no dispositivo: {device}")
    output = model(dummy_x, dummy_edge)
    
    print(f"✅ Sucesso! Shape da saída: {output.shape}") # Esperado: [16, 207]
    print(f"📊 Parâmetros totais: {sum(p.numel() for p in model.parameters()):,}")