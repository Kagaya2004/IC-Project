import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCN_TrafficModel_12h(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, horizon=12, dropout=0.2):
        """
        Arquitetura ST-GCN Profunda para Multi-Step Forecasting (Longo Prazo).
        - num_features: 5 (Velocidade + 4 Features Temporais)
        - horizon: 12 (Prevendo 1 hora completa)
        - Profundidade Espacial: 3 Camadas GCN (Visão de 3 saltos no grafo)
        """
        super(STGCN_TrafficModel_12h, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        # 1. Embedding + Normalização
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.ln_embed = nn.LayerNorm(hidden_dim)
        
        # 2. Camadas Espaciais (A GRANDE MUDANÇA: Aumentando o campo de visão para 3 saltos)
        self.spatial_gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.spatial_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.spatial_gcn3 = GCNConv(hidden_dim, hidden_dim)
        
        # 3. Camada Temporal: GRU
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.ln_temporal = nn.LayerNorm(hidden_dim)
        
        # 4. Regressão Final (Projeta os 12 passos)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon) 
        )

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, T, num_nodes, num_features = x.shape
        
        # --- PASSO 1: Embedding & Normalização ---
        x = self.embedding(x)
        x = self.ln_embed(x)
        x = F.relu(x)
        
        # --- PASSO 2: Bloco Espacial Profundo (GCN x3) ---
        out_spatial = []
        for t in range(T):
            xt = x[:, t, :, :] # [B, N, Hidden]
            xt_flat = xt.reshape(-1, self.hidden_dim)
            
            # Salto 1: Ouve os vizinhos imediatos
            h = self.spatial_gcn1(xt_flat, edge_index, edge_weight)
            h = F.relu(h)
            
            # Salto 2: Ouve os vizinhos dos vizinhos
            h = self.spatial_gcn2(h, edge_index, edge_weight)
            h = F.relu(h)

            # Salto 3: Visão de longo alcance
            h = self.spatial_gcn3(h, edge_index, edge_weight)
            h = F.relu(h)
            
            out_spatial.append(h.reshape(batch_size, 1, num_nodes, self.hidden_dim))
        
        x = torch.cat(out_spatial, dim=1) # [B, T, N, Hidden]
        
        # --- PASSO 3: Bloco Temporal (GRU) ---
        x = x.permute(1, 0, 2, 3).reshape(T, batch_size * num_nodes, self.hidden_dim)
        
        _, h_n = self.temporal_gru(x)
        h_n = self.ln_temporal(h_n.squeeze(0))
        
        # --- PASSO 4: Regressão Final ---
        out = self.regressor(h_n) 
        
        # --- PASSO 5: Formatação ---
        out = out.view(batch_size, num_nodes, self.horizon)
        out = out.permute(0, 2, 1).contiguous()
        
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = STGCN_TrafficModel_12h(num_features=5, horizon=12).to(device)
    dummy_x = torch.randn(16, 12, 207, 5).to(device)
    dummy_edge = torch.randint(0, 207, (2, 500)).to(device)
    
    print(f"🚀 Testando modelo 12h Profundo no dispositivo: {device}")
    output = model(dummy_x, dummy_edge)
    
    print(f"✅ Sucesso! Shape da saída: {output.shape}") 
    print(f"📊 Parâmetros totais: {sum(p.numel() for p in model.parameters()):,}")