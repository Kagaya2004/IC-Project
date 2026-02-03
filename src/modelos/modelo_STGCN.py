import torch
import torch.nn as nn
import torch.nn.functional as F
# Importamos a camada de convolução gráfica específica do PyG
from torch_geometric.nn import GCNConv

class STGCN_TrafficModel(nn.Module):
    def __init__(self, num_features, hidden_dim=64, output_dim=1, dropout=0.2):
        """
        Arquitetura ST-GCN para Previsão de Tráfego.
        
        Args:
            num_features: Número de variáveis de entrada por nó (do seu featureEngineering.py).
            hidden_dim: Tamanho da camada oculta (64 é um bom começo para 6GB VRAM).
            output_dim: 1, pois queremos prever apenas a velocidade.
            dropout: Taxa para evitar overfitting.
        """
        super(STGCN_TrafficModel, self).__init__()
        
        self.hidden_dim = hidden_dim

        # 1. Camada de Entrada (Embedding)
        # Transforma as features brutas em um espaço latente maior
        self.embedding = nn.Linear(num_features, hidden_dim)
        
        # 2. Bloco Espacial (Graph Convolution)
        # Processa a vizinhança geográfica usando o edge_index esparso
        self.spatial_gcn = GCNConv(hidden_dim, hidden_dim)
        
        # 3. Bloco Temporal (GRU - Gated Recurrent Unit)
        # Processa a sequência de tempo. batch_first=True espera (Batch, Seq, Features)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        
        # 4. Camada de Saída (Regressor)
        # Projeta o aprendizado final em um único valor de velocidade
        self.regressor = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        # Se os dados chegarem com 3 dimensões, expandimos para 4
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        B, T, N, F = x.shape # Agora o "unpack" funcionará sempre
        
        spatial_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :] # [Batch, Nodes, Features]
            x_t_emb = self.relu(self.embedding(x_t))
            
            # Agora a GCN encontrará todos os vizinhos de 0 a 4999!
            out_t = self.spatial_gcn(x_t_emb, edge_index, edge_weight)
            spatial_outputs.append(out_t)
            
        x_spatial = torch.stack(spatial_outputs, dim=1)
        x_temporal_input = x_spatial.view(B * N, T, self.hidden_dim)
        
        # A GRU processará a "sequência" de 1 elemento
        _, gru_output = self.temporal_gru(x_temporal_input)
        gru_final_state = gru_output.squeeze(0)
        
        prediction = self.regressor(gru_final_state)
        return prediction.view(B, N) # Retorna a velocidade de todos os 5000 nós

# =========================================
# Bloco de Teste Rápido da Arquitetura
# =========================================
if __name__ == "__main__":
    # Configurações para simular seus dados
    batch_size = 4
    window_size = 12  # Ex: 1 hora de dados
    num_nodes = 100   # Pequeno para teste
    num_features = 15 # Estimativa das suas colunas pós-engenharia
    
    # Verifica GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executando teste na: {device}")

    # 1. Instancia o Modelo
    model = STGCN_TrafficModel(num_features=num_features).to(device)
    print("✅ Modelo ST-GCN instanciado com sucesso.")

    # 2. Gera Dados Sintéticos (Simulando o DataLoader)
    # Entrada X: [Batch, T, N, F]
    dummy_x = torch.randn(batch_size, window_size, num_nodes, num_features).to(device)
    
    # Grafo Sintético (Edge Index aleatório)
    # Criando arestas aleatórias entre os 100 nós
    rows = torch.randint(0, num_nodes, (200,))
    cols = torch.randint(0, num_nodes, (200,))
    dummy_edge_index = torch.stack([rows, cols], dim=0).to(device)

    # 3. Forward Pass (Teste de Fogo)
    try:
        print("\nIniciando forward pass de teste...")
        output = model(dummy_x, dummy_edge_index)
        print("✅ Forward pass concluído!")
        print(f"Shape da Entrada: {dummy_x.shape}")
        print(f"Shape da Saída: {output.shape} (Esperado: [Batch, Nodes])")
        
        if output.shape == (batch_size, num_nodes):
            print("\n🎉 SUCESSO: A arquitetura está funcional e pronta para o treino!")
        else:
            print("\n⚠️ ATENÇÃO: O shape da saída está diferente do esperado.")
            
    except Exception as e:
        print(f"\n❌ ERRO durante o teste da arquitetura:\n{e}")