import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """
    Camada de Convolução Temporal utilizando Gated Linear Units (GLU).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        # x: [Batch, Channels, Time, Nodes]
        x_conv = self.conv(x)
        # GLU: Divide os canais ao meio e aplica sigmoid em uma das partes
        p, q = torch.split(x_conv, x_conv.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatialConv(nn.Module):
    """
    Camada de Convolução Espacial (Graph Convolution).
    Utiliza o edge_index e edge_weight processados no preparar_grafo.py.
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # x: [Batch, Channels, Time, Nodes]
        # Ajustamos para processar a vizinhança no grafo
        batch, c, t, n = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, n, c) # [B*T, N, C]
        
        # Propagação simples de mensagens (pode ser substituído por GCNConv do PyG se preferir)
        # Aqui simulamos a agregação via matriz de adjacência
        adj = torch.zeros((n, n), device=x.device)
        adj[edge_index[0], edge_index[1]] = edge_weight
        
        # Agregação: D^-1 * A * X
        out = torch.matmul(adj, x_flat) # [B*T, N, C]
        out = self.theta(out) # Projeção linear
        
        # Volta para o formato original [Batch, Out_Channels, Time, Nodes]
        out = out.reshape(batch, t, n, -1).permute(0, 3, 1, 2)
        return F.relu(out)

class STConvBlock(nn.Module):
    """
    Bloco ST-Conv: Temporal -> Espacial -> Temporal
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.tmp_conv1(x)
        x = self.spa_conv(x, edge_index, edge_weight)
        x = self.tmp_conv2(x)
        # Normalização de camada para estabilidade
        batch, c, t, n = x.shape
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class STGCN_TrafficModel(nn.Module):
    def __init__(self, num_nodes=207, num_features=5, num_timesteps_input=12, num_timesteps_output=12):
        super(STGCN_TrafficModel, self).__init__()
        
        # Camada de entrada para ajustar as 5 features iniciais
        self.input_conv = nn.Conv2d(num_features, 64, kernel_size=(1, 1))
        
        # Blocos Espaciais-Temporais
        self.block1 = STConvBlock(64, 32, 64)
        self.block2 = STConvBlock(64, 32, 64)
        
        # Camada de saída para o Horizonte de 12 passos (1 hora)
        # Reduzimos a dimensão temporal e projetamos para a saída desejada
        self.fully_connected = nn.Sequential(
            nn.Linear(64 * num_timesteps_input, 128),
            nn.ReLU(),
            nn.Linear(128, num_timesteps_output)
        )

    def forward(self, x, edge_index, edge_weight):
        """
        x: [Batch, Time=12, Nodes=207, Features=5]
        """
        # 1. Ajuste de dimensões para Conv2D: [B, F, T, N]
        x = x.permute(0, 3, 1, 2) 
        
        # 2. Projeção inicial
        x = self.input_conv(x)
        
        # 3. Blocos ST-Conv
        x = self.block1(x, edge_index, edge_weight)
        x = self.block2(x, edge_index, edge_weight)
        
        # 4. Saída: [Batch, Channels, Time, Nodes] -> [Batch, Nodes, Channels * Time]
        batch, c, t, n = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, n, -1)
        
        # 5. Predição do Horizonte (12 passos)
        out = self.fully_connected(x) # [Batch, Nodes, Horizon=12]
        
        # 6. Formato final compatível com o y do DataLoader: [Batch, Horizon, Nodes]
        return out.permute(0, 2, 1)

if __name__ == "__main__":
    # Teste de shape
    dummy_x = torch.randn(32, 12, 207, 5)
    dummy_edge_idx = torch.randint(0, 207, (2, 500))
    dummy_edge_wt = torch.rand(500)
    
    model = STGCN_TrafficModel()
    pred = model(dummy_x, dummy_edge_idx, dummy_edge_wt)
    print(f"Shape da saída: {pred.shape}") # Deve ser [32, 12, 207]